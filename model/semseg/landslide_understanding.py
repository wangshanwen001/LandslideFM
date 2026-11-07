import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import json
import re
import os
import sys
import time


def resolve_qwen_model_path(model_name, vlm_cfg=None):
    vlm_cfg = vlm_cfg or {}

    candidates = []
    explicit = vlm_cfg.get("qwen_model_path")
    if explicit:
        candidates.append(explicit)
    candidates.append(model_name)

    if "/" in model_name and not os.path.isdir(model_name):
        base = model_name.rstrip("/").split("/")[-1]
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates.append(os.path.join(repo_root, "pretrained", base))
        candidates.append(os.path.join("pretrained", base))

    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    return model_name


_QWEN_VL_CLASSES = {
    "qwen3": "Qwen3VLForConditionalGeneration",
    "qwen2.5": "Qwen2_5_VLForConditionalGeneration",
    "qwen2_5": "Qwen2_5_VLForConditionalGeneration",
}


def resolve_qwen_vl_model_class(model_name):
    name = (model_name or "").lower()
    for key, cls_name in _QWEN_VL_CLASSES.items():
        if key in name:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                raise ImportError(
                    f"'{model_name}' requires transformers.{cls_name}, but the "
                    f"installed transformers ({transformers.__version__}) does "
                    f"not provide it. Upgrade transformers (>=4.57 for Qwen3-VL)."
                )
            return cls
    return transformers.AutoModelForImageTextToText


def is_kimi_vl_model(model_name):
    return "kimi" in (model_name or "").lower()


def resolve_vlm_model_class(model_name):
    if is_kimi_vl_model(model_name):
        return transformers.AutoModelForCausalLM
    return resolve_qwen_vl_model_class(model_name)


def apply_kimi_transformers_compat():
    from transformers import activations
    from transformers.cache_utils import Cache

    if not hasattr(activations, "PytorchGELUTanh"):
        gelu_tanh = getattr(activations, "GELUTanh", None)
        if gelu_tanh is None:
            class PytorchGELUTanh(torch.nn.Module):
                def forward(self, x):
                    return F.gelu(x, approximate="tanh")
            gelu_tanh = PytorchGELUTanh
        activations.PytorchGELUTanh = gelu_tanh

    if not hasattr(Cache, "get_usable_length"):
        def get_usable_length(self, new_seq_length=None, layer_idx=0):
            return self.get_seq_length(layer_idx)
        Cache.get_usable_length = get_usable_length

    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda self: self.get_seq_length())


def apply_kimi_generation_compat(model):
    from transformers.cache_utils import Cache

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return
    cls = type(language_model)
    if getattr(cls, "_landslide_mask_length_fix", False):
        return

    original = cls.prepare_inputs_for_generation

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = original(
            self,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if not isinstance(past_key_values, Cache) or attention_mask is None:
            return model_inputs

        embeds = model_inputs.get("inputs_embeds")
        new_tokens = embeds.shape[1] if embeds is not None else model_inputs["input_ids"].shape[1]
        wanted = past_key_values.get_seq_length() + new_tokens
        if model_inputs["attention_mask"].shape[-1] != wanted and wanted <= attention_mask.shape[-1]:
            model_inputs["attention_mask"] = attention_mask[:, -wanted:]
        return model_inputs

    cls.prepare_inputs_for_generation = prepare_inputs_for_generation
    cls._landslide_mask_length_fix = True


ENVIRONMENTAL_CLASSES = [
    "landslide",
    "building",
    "road",
    "vegetation",
    "water",
    "bare_soil",
    "infrastructure",
]


class LandslideUnderstanding:
    def __init__(self, cfg=None, device="cuda", verbose=True, language="en", lora_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = cfg or {}
        self.verbose = verbose
        self.language = language
        self.lora_path = lora_path
        self._init_vlm()
        self._init_prompts()
        self.conversation_history = []

    def _log(self, msg, end="\n"):
        if self.verbose:
            print(msg, end=end)

    def _init_vlm(self):
        vlm_cfg = self.cfg.get("vlm", {})
        model_name = vlm_cfg.get("qwen_model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        resolved_name = resolve_qwen_model_path(model_name, vlm_cfg)
        use_4bit = vlm_cfg.get("use_4bit", True)

        self.is_kimi = is_kimi_vl_model(model_name)
        model_cls = resolve_vlm_model_class(model_name)
        if self.is_kimi:
            kimi_dtype = str(vlm_cfg.get("kimi_dtype", "bfloat16")).lower()
            model_dtype = torch.float16 if kimi_dtype == "float16" else torch.bfloat16
            apply_kimi_transformers_compat()
        else:
            model_dtype = torch.float16
        load_kwargs = {"trust_remote_code": True} if self.is_kimi else {}

        if resolved_name != model_name:
            self._log(f"  Using local VLM weights: {resolved_name}")
        lora_msg = f" + LoRA: {self.lora_path}" if self.lora_path else ""
        self._log(f"Loading VLM model: {resolved_name} (4-bit: {use_4bit}){lora_msg}... ", end="")
        start_time = time.time()

        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        else:
            device_map = "cpu"

        try:
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_dtype,
                    bnb_4bit_quant_type="nf4",
                )
                self.vlm_model = model_cls.from_pretrained(
                    resolved_name,
                    torch_dtype=model_dtype,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    **load_kwargs,
                )
            else:
                self.vlm_model = model_cls.from_pretrained(
                    resolved_name,
                    torch_dtype=model_dtype,
                    device_map=device_map,
                    **load_kwargs,
                )

            if self.is_kimi:
                apply_kimi_generation_compat(self.vlm_model)

            if self.lora_path:
                try:
                    from peft import PeftModel
                    self.vlm_model = PeftModel.from_pretrained(self.vlm_model, self.lora_path)
                    self._log(f"\n  ✓ LoRA adapter loaded from: {self.lora_path}")
                except ImportError as e:
                    print(f"\n  [FATAL] --lora-path='{self.lora_path}' was given, "
                          f"but peft could not be imported: {e}")
                    print("  The LoRA adapter was NOT applied. Install peft in the "
                          "image (add it to requirements.txt and rebuild), or drop "
                          "--lora-path to evaluate the base model on purpose.")
                    sys.exit(1)
                except Exception as e:
                    print(f"\n  [FATAL] Failed to load LoRA adapter from "
                          f"'{self.lora_path}': {e}")
                    print("  The LoRA adapter was NOT applied. Check that the adapter "
                          "directory exists and matches the base model, or drop "
                          "--lora-path to evaluate the base model on purpose.")
                    sys.exit(1)

            self.vlm_model.eval()
            if self.is_kimi:
                self.processor = AutoProcessor.from_pretrained(
                    resolved_name,
                    trust_remote_code=True,
                )
                self.processor.image_processor.in_token_limit = int(
                    vlm_cfg.get("kimi_in_token_limit", 1024)
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    resolved_name,
                    min_pixels=128 * 28 * 28,
                    max_pixels=256 * 28 * 28,
                )
            elapsed = time.time() - start_time
            self._log(f"done! ({elapsed:.1f}s)")
        except Exception as e:
            self._log(f"failed! Error: {str(e)}")
            raise

    def _init_prompts(self):
        self.prompts = {
            "environmental_context": {
                "en": """Analyze the environmental context of the landslide in this remote sensing image.

Image Information:
- The red area indicates the detected landslide location.

Please identify and analyze the following:
1. Topographic features around the landslide (slope, aspect, etc.)
2. Presence of infrastructure such as buildings and roads
3. Natural geographic elements such as rivers and water bodies
4. Vegetation coverage
5. Other factors that may affect landslide stability

Please output the analysis results in a structured format.""",
                "zh": """分析这张遥感图像中滑坡的环境背景。

图像信息：
- 红色区域标注了检测到的滑坡位置

请识别并分析以下内容：
1. 滑坡周围的地形特征（坡度、坡向等）
2. 是否存在建筑物、道路等基础设施
3. 是否有河流、水体等自然地理要素
4. 植被覆盖情况
5. 其他可能影响滑坡稳定性的环境因素

请以结构化方式输出分析结果。"""
            },
            "impact_assessment": {
                "en": """Assess the potential impact and risk level of this landslide.

Image Information:
- The red area indicates the detected landslide location.

Please assess:
1. Landslide scale (small/medium/large)
2. Areas and objects that may be threatened (buildings, roads, residents, etc.)
3. Potential risk level (low/medium/high/extremely high)
4. Recommended emergency response measures
5. Estimated potential economic losses

Please output the assessment results in a structured format.""",
                "zh": """评估此滑坡的潜在影响和风险等级。

图像信息：
- 红色区域标注了检测到的滑坡位置

请评估：
1. 滑坡规模（小型/中型/大型）
2. 可能受到威胁的区域和对象（建筑物、道路、居民等）
3. 潜在风险等级（低/中/高/极高）
4. 建议的应急响应措施
5. 可能的经济损失预估

请以结构化方式输出评估结果。"""
            },
            "structured_env": {
                "en": """You are a landslide analysis assistant. Analyze the remote sensing image pair.
The first image is the original, the second has a RED OVERLAY showing the detected landslide area.

Answer ONLY with a JSON object (no markdown, no extra text), following this schema:
{
  "landslide_exists": true or false,
  "has_building": true or false,
  "has_road": true or false,
  "has_water": true or false,
  "has_vegetation": true or false
}

Field definitions:
- landslide_exists : Is there actually a landslide visible in the image?
- has_building     : Are there buildings within or immediately around the red area?
- has_road         : Are there roads within or immediately around the red area?
- has_water        : Are there rivers / water bodies within or near the scene?
- has_vegetation   : Is there significant vegetation / forest cover around?

Output ONLY the JSON, nothing else.""",
                "zh": """你是一个滑坡分析助手。分析这组遥感图像。
第一张是原图，第二张红色区域是检测出的滑坡位置。

请只输出一个 JSON 对象（不要 markdown，不要额外文字），严格遵循如下 schema：
{
  "landslide_exists": true 或 false,
  "has_building": true 或 false,
  "has_road": true 或 false,
  "has_water": true 或 false,
  "has_vegetation": true 或 false
}

字段定义：
- landslide_exists : 图像中实际是否存在滑坡？
- has_building     : 红色区域内或紧邻处是否有建筑物？
- has_road         : 红色区域内或紧邻处是否有道路？
- has_water        : 场景内或附近是否有河流/水体？
- has_vegetation   : 周边是否有显著植被/森林覆盖？

只输出 JSON，其他内容不要写。"""
            },
            "qa_system": {
                "en": """You are a landslide disaster analysis expert assistant.

Known Information:
{context}

Please answer the user's question based on the above information and images.
If information is insufficient, clearly indicate what additional data is needed.""",
                "zh": """你是一个滑坡灾害分析专家助手。

已知信息：
{context}

请根据上述信息和图像，回答用户的问题。
如果信息不足，请明确说明需要哪些额外数据。"""
            },
        }

    def _generate_overlay(self, original_image, seg_mask, palette=None):
        if palette is None:
            palette = {1: (255, 0, 0)}

        overlay = original_image.copy()
        draw = ImageDraw.Draw(overlay)

        if isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.cpu().numpy()
        elif isinstance(seg_mask, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported mask type: {type(seg_mask)}")

        h, w = seg_mask.shape
        self._log(f"  Generating overlay ({h}x{w})... ", end="")
        start_time = time.time()

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        for class_id in palette:
            color = palette[class_id]
            binary_mask = (seg_mask == class_id).astype(np.uint8)

            if has_cv2:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) > 4:
                        contour = contour.squeeze(1).tolist()
                        draw.polygon(contour, fill=(*color, 128), outline=color, width=3)
            else:
                contours = self._find_contours_simple(binary_mask)
                for contour in contours:
                    if len(contour) > 4:
                        draw.polygon(contour, fill=(*color, 128), outline=color, width=3)

        elapsed = time.time() - start_time
        self._log(f"done! ({elapsed:.1f}s)")
        return overlay

    def _find_contours_simple(self, binary_mask):
        contours = []
        h, w = binary_mask.shape
        visited = np.zeros((h, w), dtype=np.bool_)

        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] == 1 and not visited[y, x]:
                    contour = []
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if cx < 0 or cx >= w or cy < 0 or cy >= h:
                            continue
                        if binary_mask[cy, cx] == 1 and not visited[cy, cx]:
                            visited[cy, cx] = True
                            contour.append((cx, cy))
                            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                    if len(contour) > 4:
                        contours.append(contour)
        return contours

    def _run_vlm(self, messages, max_new_tokens=512):
        self._log(f"  Running VLM inference (max_tokens={max_new_tokens})... ", end="")
        start_time = time.time()

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            generate_kwargs = {}
            if self.is_kimi:
                image_inputs = [
                    item["image"]
                    for msg in messages
                    if isinstance(msg.get("content"), list)
                    for item in msg["content"]
                    if isinstance(item, dict) and item.get("type") == "image"
                ]
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
                eos_ids = self.vlm_model.generation_config.eos_token_id
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                if im_end_id is not None and im_end_id != self.processor.tokenizer.unk_token_id:
                    eos_ids = list(eos_ids or []) + [im_end_id]
                generate_kwargs["eos_token_id"] = eos_ids
            else:
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(self.vlm_model.device)

            with torch.no_grad():
                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    **generate_kwargs,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )[0]

            elapsed = time.time() - start_time
            self._log(f"done! ({elapsed:.1f}s)")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            self._log(f"failed! ({elapsed:.1f}s) Error: {str(e)}")
            raise

    def analyze_environmental_context(self, original_image, seg_mask):
        step_log = "  Step 1/2: Analyzing environmental context..." if self.language == "en" else "  步骤 1/2: 分析环境背景..."
        self._log(step_log)
        overlay = self._generate_overlay(original_image, seg_mask)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": self.prompts["environmental_context"][self.language]},
                ],
            }
        ]

        result = self._run_vlm(messages, max_new_tokens=512)
        context_label = "Environmental analysis result:\n" if self.language == "en" else "环境分析结果：\n"
        self.conversation_history.append({
            "role": "system",
            "content": f"{context_label}{result}",
        })
        return result

    def assess_impact(self, original_image, seg_mask):
        step_log = "  Step 2/2: Assessing impact..." if self.language == "en" else "  步骤 2/2: 评估影响..."
        self._log(step_log)
        overlay = self._generate_overlay(original_image, seg_mask)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": self.prompts["impact_assessment"][self.language]},
                ],
            }
        ]

        result = self._run_vlm(messages, max_new_tokens=512)
        impact_label = "Impact assessment result:\n" if self.language == "en" else "影响评估结果：\n"
        self.conversation_history.append({
            "role": "system",
            "content": f"{impact_label}{result}",
        })
        return result

    def analyze_structured_env(self, original_image, seg_mask):
        """Simplified structured environmental analysis → JSON dict."""
        import json as _json
        step_log = "  Structured env analysis..." if self.language == "en" else "  结构化环境分析..."
        self._log(step_log)
        overlay = self._generate_overlay(original_image, seg_mask)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": self.prompts["structured_env"][self.language]},
                ],
            }
        ]

        raw = self._run_vlm(messages, max_new_tokens=256)

        try:
            # Robust JSON extraction (handles ```json fences, stray text, etc.)
            s = raw.strip()
            start = s.find("{")
            end = s.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON braces found")
            parsed = _json.loads(s[start:end + 1])
        except Exception as exc:
            self._log(f"  [WARN] Failed to parse structured JSON: {exc}. Raw: {raw[:120]}")
            parsed = {
                "landslide_exists": False,
                "has_building": False,
                "has_road": False,
                "has_water": False,
                "has_vegetation": False,
                "_parse_error": str(exc),
                "_raw": raw,
            }

        # Normalize booleans (VLM might return "true", 1, etc.)
        def _as_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "y", "t", "有", "是")
            return False

        fields = ["landslide_exists", "has_building", "has_road", "has_water", "has_vegetation"]
        out = {k: _as_bool(parsed.get(k, False)) for k in fields}
        if "_parse_error" in parsed:
            out["_parse_error"] = parsed["_parse_error"]
            out["_raw"] = parsed.get("_raw", raw)
        return out

    def ask_question(self, original_image, seg_mask, question):
        qa_label = "QA" if self.language == "en" else "问答"
        self._log(f"  {qa_label}: {question[:50]}...")
        overlay = self._generate_overlay(original_image, seg_mask)

        context = "\n".join(
            [msg["content"] for msg in self.conversation_history[-5:]]
        )
        if not context:
            context = "No environmental analysis and impact assessment have been conducted yet." if self.language == "en" else "尚未进行环境分析和影响评估。"

        system_prompt = self.prompts["qa_system"][self.language].format(context=context)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": question},
                ],
            },
        ]

        result = self._run_vlm(messages, max_new_tokens=768)
        self.conversation_history.append({
            "role": "user",
            "content": question,
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": result,
        })
        return result

    def reset_conversation(self):
        self.conversation_history = []

    def get_summary(self):
        summary = "对话历史摘要：\n"
        for msg in self.conversation_history:
            role = "系统" if msg["role"] == "system" else \
                   "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]
            if isinstance(content, str):
                summary += f"{role}: {content[:100]}...\n" if len(content) > 100 else f"{role}: {content}\n"
        return summary


class LandslideQAInterface:
    def __init__(self, config_path=None, checkpoint_path=None, device="cuda", verbose=True, language="en", auto_checkpoint=False, lora_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.seg_model = None
        self.understanding_model = None
        self.verbose = verbose
        self.language = language
        self.auto_checkpoint = auto_checkpoint
        self.lora_path = lora_path

    def _log(self, msg, end="\n"):
        if self.verbose:
            print(msg, end=end)

    @staticmethod
    def find_exp_checkpoint(config_path=None):
        exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exp")
        
        if not os.path.exists(exp_dir):
            return None

        import yaml
        cfg = None
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        best_checkpoint = None
        best_time = 0

        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file == "best.pth":
                    checkpoint_path = os.path.join(root, file)
                    mtime = os.path.getmtime(checkpoint_path)
                    
                    if cfg:
                        dataset = cfg.get("dataset", "")
                        backbone = cfg.get("backbone", "")
                        
                        if dataset and dataset in root and backbone and backbone in root:
                            if mtime > best_time:
                                best_time = mtime
                                best_checkpoint = checkpoint_path
                    else:
                        if mtime > best_time:
                            best_time = mtime
                            best_checkpoint = checkpoint_path

        return best_checkpoint

    def load_segmentation_model(self):
        import yaml
        from model.semseg.dpt_multitask import MultiTaskDPT

        if self.auto_checkpoint and not self.checkpoint_path:
            self._log("Auto-discovering trained checkpoint... ", end="")
            self.checkpoint_path = self.find_exp_checkpoint(self.config_path)
            if self.checkpoint_path:
                self._log(f"found: {self.checkpoint_path}")
            else:
                self._log("no checkpoint found, using default")

        self._log(f"Loading segmentation model... ", end="")
        start_time = time.time()

        with open(self.config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        model_configs = {
            "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
            "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
            "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        backbone_size = cfg["backbone"].split("_")[-1]
        model_config = {**model_configs[backbone_size], "nclass": cfg.get("nclass", 2)}

        self.seg_model = MultiTaskDPT(**model_config).to(self.device)

        if self.checkpoint_path:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    "Segmentation checkpoint not found: {}. "
                    "Check the --save-path used by sagt.py, or pass "
                    "--auto-checkpoint / a correct --checkpoint.".format(
                        os.path.abspath(self.checkpoint_path)))
            self._log(f"  Checkpoint: {os.path.abspath(self.checkpoint_path)} ")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model", checkpoint.get("model_ema", checkpoint))
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            self.seg_model.load_state_dict(new_state_dict, strict=False)

        self.seg_model.eval()
        self.cfg = cfg

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        elapsed = time.time() - start_time
        self._log(f"done! ({elapsed:.1f}s)")

    def load_understanding_model(self):
        load_log = "Loading VLM understanding model..." if self.language == "en" else "正在加载VLM理解模型..."
        self._log(load_log)
        self.understanding_model = LandslideUnderstanding(
            cfg=self.cfg if hasattr(self, "cfg") else {},
            device=self.device.type,
            verbose=self.verbose,
            language=self.language,
            lora_path=self.lora_path,
        )
        loaded_log = "✓ VLM model loaded" if self.language == "en" else "✓ VLM模型加载完成"
        self._log(loaded_log)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size[::-1]
        h, w = original_size
        patch_size = 14
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size

        if (new_h, new_w) != (h, w):
            image_resized = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            image_resized = image

        arr = np.array(image_resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor, original_size, image

    @torch.no_grad()
    def run_segmentation(self, image_path):
        self._log(f"  Running segmentation... ", end="")
        start_time = time.time()

        tensor, original_size, image = self.preprocess_image(image_path)
        seg_logits = self.seg_model(tensor, task="seg")

        if seg_logits.shape[-2:] != original_size:
            seg_logits = F.interpolate(seg_logits, size=original_size, mode="bilinear", align_corners=True)

        seg_mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        elapsed = time.time() - start_time
        self._log(f"done! ({elapsed:.1f}s)")
        return seg_mask, image

    def analyze_image(self, image_path):
        if self.seg_model is None:
            self.load_segmentation_model()
        if self.understanding_model is None:
            self.load_understanding_model()

        seg_mask, original_image = self.run_segmentation(image_path)

        environment_result = self.understanding_model.analyze_environmental_context(
            original_image, seg_mask
        )

        impact_result = self.understanding_model.assess_impact(
            original_image, seg_mask
        )

        return {
            "segmentation_mask": seg_mask,
            "original_image": original_image,
            "environmental_context": environment_result,
            "impact_assessment": impact_result,
        }

    def ask(self, image_path, question):
        if self.seg_model is None:
            self.load_segmentation_model()
        if self.understanding_model is None:
            self.load_understanding_model()

        seg_mask, original_image = self.run_segmentation(image_path)
        answer = self.understanding_model.ask_question(original_image, seg_mask, question)

        return {
            "segmentation_mask": seg_mask,
            "original_image": original_image,
            "question": question,
            "answer": answer,
        }

    def reset(self):
        if self.understanding_model:
            self.understanding_model.reset_conversation()