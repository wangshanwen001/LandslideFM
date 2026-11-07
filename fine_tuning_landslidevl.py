import argparse
import csv
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

STRUCTURED_FIELDS = [
    "landslide_exists", "has_building", "has_road",
    "has_water", "has_vegetation",
]


def read_training_csv(csv_path):
    """Return list of dicts: {image_name, text_description, ...structured}"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lstrip("\ufeff") for h in reader.fieldnames]
        for r in reader:
            name = (r.get("image_name") or r.get("image") or "").strip()
            if not name or name.startswith("#"):
                continue
            desc = (r.get("text_description") or "").strip()
            if not desc:
                continue   # skip rows without text description
            row = {"image_name": name, "text_description": desc}
            for f_ in STRUCTURED_FIELDS:
                v = (r.get(f_) or "").strip()
                row[f_] = v.lower() in ("1", "true", "yes", "y", "t") if v else None
            rows.append(row)
    return rows


def resolve_image_path(image_dir, image_name, subdirs=None):
    if subdirs is None:
        subdirs = ["JPEGImages", "None"]

    basename = os.path.basename(image_name)
    tried = []

    # 1. Try the path as-is (handles image_name containing a subpath)
    direct = os.path.join(image_dir, image_name)
    tried.append(direct)
    if os.path.exists(direct):
        return direct

    # 2. Try basename directly in image_dir
    root_candidate = os.path.join(image_dir, basename)
    tried.append(root_candidate)
    if os.path.exists(root_candidate):
        return root_candidate

    # 3. Try each subfolder under image_dir (for case a)
    for sub in subdirs:
        candidate = os.path.join(image_dir, sub, basename)
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate

    # 4. Try SIBLING folders of image_dir (for case b: image_dir is already a subfolder)
    parent_dir = os.path.dirname(os.path.abspath(image_dir))
    if parent_dir and os.path.basename(image_dir) in subdirs:
        for sub in subdirs:
            candidate = os.path.join(parent_dir, sub, basename)
            tried.append(candidate)
            if os.path.exists(candidate):
                return candidate

    # 5. Fallback: raise a clear error listing all tried paths
    raise FileNotFoundError(
        f"Image not found: {image_name}\n"
        f"Tried the following paths:\n  " + "\n  ".join(tried)
    )


def generate_overlay(original_image, seg_mask, color=(255, 0, 0), alpha=128):
    """Draw red contour overlay for landslide (class=1) using OpenCV if available."""
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()
    seg_mask = seg_mask.astype(np.uint8)

    overlay = original_image.copy()
    draw = ImageDraw.Draw(overlay)

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    binary_mask = (seg_mask == 1).astype(np.uint8)
    if has_cv2:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 4:
                pts = contour.squeeze(1).tolist()
                draw.polygon(pts, fill=(*color, alpha), outline=color, width=3)
    else:
        # Fallback: simple flood fill
        h, w = binary_mask.shape
        visited = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] == 1 and not visited[y, x]:
                    pts = []
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if 0 <= cx < w and 0 <= cy < h and binary_mask[cy, cx] == 1 and not visited[cy, cx]:
                            visited[cy, cx] = True
                            pts.append((cx, cy))
                            stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
                    if len(pts) > 4:
                        draw.polygon(pts, fill=(*color, alpha), outline=color, width=3)
    return overlay


STRUCTURED_ENV_PROMPT = {
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

字段说明：
- landslide_exists : 图像中是否真的存在滑坡？
- has_building     : 红色区域内或紧邻处是否有建筑物？
- has_road         : 红色区域内或紧邻处是否有道路？
- has_water        : 场景内或附近是否有河流/水体？
- has_vegetation   : 周围是否有显著植被/森林覆盖？

只输出 JSON，不要其他内容。""",
}

ENVIRONMENTAL_PROMPT = {
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

请以结构化方式输出分析结果。""",
}


def _build_json_target(row):
    """Build a compact JSON string from structured fields (lowcase booleans)."""
    import json as _json
    payload = {f: bool(row.get(f, False)) for f in STRUCTURED_FIELDS}
    return _json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class LoRALandslideDataset(torch.utils.data.Dataset):
    def __init__(self, rows, image_dir, seg_interface, language="en"):
        self.rows = rows
        self.image_dir = image_dir
        self.seg_interface = seg_interface
        self.language = language
        self.json_prompt = STRUCTURED_ENV_PROMPT[language]
        self.desc_prompt = ENVIRONMENTAL_PROMPT[language]
        self._cache = {}   # cache segmentation results to avoid re-running

    def __len__(self):
        # Two tasks per image
        return len(self.rows) * 2

    def _get_seg_and_overlay(self, image_path):
        """Run segmentation + overlay. Cache results by image path."""
        if image_path in self._cache:
            return self._cache[image_path]

        seg_mask, original_image = self.seg_interface.run_segmentation(image_path)
        overlay = generate_overlay(original_image, seg_mask)
        self._cache[image_path] = (original_image, overlay)
        return original_image, overlay

    def __getitem__(self, idx):
        row = self.rows[idx // 2]
        is_json_task = (idx % 2 == 0)
        img_path = resolve_image_path(self.image_dir, row["image_name"])
        original_image, overlay = self._get_seg_and_overlay(img_path)

        if is_json_task:
            prompt = self.json_prompt
            target = _build_json_target(row)
        else:
            prompt = self.desc_prompt
            target = row["text_description"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target},
                ],
            },
        ]
        return {
            "messages": messages,
            "image_path": img_path,
            "target_text": target,
            "task": "json" if is_json_task else "desc",
        }


def make_collate_fn(processor, max_length=1024):
    """Returns a collate function that tokenizes messages for training."""
    from qwen_vl_utils import process_vision_info

    def collate_fn(batch):
        # Use first sample (batch_size=1 typical for VLM LoRA)
        texts = []
        images_batch = []
        for item in batch:
            messages = item["messages"]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            images_batch.append(image_inputs)

        inputs = processor(
            text=texts,
            images=images_batch,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        # Build labels: mask prompt tokens, only learn from assistant response
        labels = inputs["input_ids"].clone()
        # Mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # For Qwen2-VL, image tokens also need masking (they are already handled)
        if "image_grid_thw" in inputs:
            image_mask = inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            labels[image_mask] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Qwen-VL on Landslide Understanding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--auto-checkpoint", action="store_true")
    parser.add_argument("--csv", type=str, required=True,
                        help="Training CSV with text_description column")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="lora_output/landslide_lora")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")

    # LoRA hyperparameters
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank (r=8 → ~20M params for 7B)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (typically 2*r)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of module names to apply LoRA")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("LoRA Fine-tuning for Qwen-VL on Landslide Understanding")
    print("=" * 70)
    print(f"Config:           {args.config}")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"CSV:              {args.csv}")
    print(f"Image dir:        {args.image_dir}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Language:         {args.language}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Grad accumulation:{args.gradient_accumulation}")
    print(f"LR:               {args.lr}")
    print(f"LoRA r/alpha:     {args.lora_r}/{args.lora_alpha}")
    print(f"LoRA target:      {args.lora_target}")
    print("=" * 70 + "\n")

    print("Loading segmentation backbone for overlay generation...")
    from model.semseg.landslide_understanding import LandslideQAInterface

    seg_interface = LandslideQAInterface(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        verbose=False,
        language=args.language,
        auto_checkpoint=args.auto_checkpoint,
    )
    seg_interface.load_segmentation_model()
    print(f"✓ Segmentation backbone loaded: {seg_interface.checkpoint_path or 'default'}")

    rows = read_training_csv(args.csv)
    print(f"Loaded {len(rows)} training samples (with text_description)")

    if len(rows) == 0:
        print("ERROR: No training samples with text_description found in CSV.")
        print("Add a 'text_description' column to your CSV with human-written descriptions.")
        sys.exit(1)

    # Print a sample
    print("\nSample training data:")
    print(f"  Image: {rows[0]['image_name']}")
    print(f"  Description: {rows[0]['text_description'][:200]}...")
    print(f"  JSON target: {_build_json_target(rows[0])}")
    print(f"  Total training samples: {len(rows) * 2} "
          f"({len(rows)} images x 2 tasks)")

    print("\nLoading base VLM model...")
    import yaml
    from transformers import AutoProcessor, BitsAndBytesConfig
    from model.semseg.landslide_understanding import resolve_qwen_vl_model_class

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    vlm_cfg = cfg.get("vlm", {})
    model_name = vlm_cfg.get("qwen_model_name", "Qwen/Qwen-VL-3B-Instruct")
    use_4bit = vlm_cfg.get("use_4bit", True)
    model_cls = resolve_qwen_vl_model_class(model_name)

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    print(f"✓ Loaded {model_name}")

    print("\nApplying LoRA adapter...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.lora_target.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

    model.print_trainable_parameters()
    model.train()

    print("\nPre-computing segmentation overlays for all training images...")
    dataset = LoRALandslideDataset(rows, args.image_dir, seg_interface, language=args.language)

    valid_rows = []
    skipped = 0
    for i, row in enumerate(rows):
        img_path = resolve_image_path(args.image_dir, row["image_name"])
        try:
            if img_path not in dataset._cache:
                dataset._get_seg_and_overlay(img_path)
            valid_rows.append(row)
        except (OSError, IOError) as e:
            skipped += 1
            print(f"  [SKIP] corrupted image: {img_path} ({e})")
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] overlays ready")
    dataset.rows = valid_rows
    print(f"✓ All overlays cached ({len(valid_rows)} usable, {skipped} skipped)")

    collate_fn = make_collate_fn(processor, max_length=args.max_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,   # VLM data loading is not picklable easily
    )

    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = (len(loader) * args.epochs) // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"Total optimization steps: {total_steps} (warmup: {warmup_steps})")

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    global_step = 0
    loss_history = []
    loss_history_path = os.path.join(args.output_dir, "loss_history.json")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            # Move to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation
            epoch_steps += 1

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0:
                    print(f"  Epoch {epoch+1}/{args.epochs} | "
                          f"Step {global_step}/{total_steps} | "
                          f"Loss: {epoch_loss/epoch_steps:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch+1}/{args.epochs} complete | Avg Loss: {avg_loss:.4f}")

        loss_history.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
            "steps": epoch_steps,
            "elapsed_sec": round(time.time() - epoch_start, 2),
        })
        with open(loss_history_path, "w") as f:
            json.dump(loss_history, f, indent=2)
        print(f"  Loss history → {loss_history_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"✓ Saved LoRA checkpoint → {ckpt_dir}")

    final_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\n✓ Final LoRA adapter saved → {final_dir}")

    # Save training metadata
    meta = {
        "base_model": model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lr": args.lr,
        "final_loss": avg_loss,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "n_samples": len(dataset.rows),
        "n_train_items": len(dataset),
        "multi_task": True,
        "language": args.language,
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nTo use the LoRA adapter in inference:")
    print(f"  python inference_vlm.py --config {args.config} \\")
    print(f"      --checkpoint <your_seg_ckpt> \\")
    print(f"      --image <image.jpg> --mode analyze \\")
    print(f"      --lora-path {final_dir}")
    print(f"\nTo use in evaluation:")
    print(f"  python evaluate_vlm.py --config {args.config} \\")
    print(f"      --checkpoint <your_seg_ckpt> \\")
    print(f"      --csv annotations/landslide_vlm_eval.csv \\")
    print(f"      --image-dir Landslide/JPEGImages \\")
    print(f"      --lora-path {final_dir}")


if __name__ == "__main__":
    main()
