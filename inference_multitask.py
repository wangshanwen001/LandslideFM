import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 优先导入多任务模型；如果你的仓库还没有 dpt_multitask.py，会退回原 DPT，只支持 seg。
try:
    from model.semseg.dpt_multitask import MultiTaskDPT
    HAS_MULTITASK = True
except Exception:
    MultiTaskDPT = None
    HAS_MULTITASK = False

from model.semseg.dpt import DPT


model_configs = {
    "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


DEFAULT_COLORS = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [255, 128, 0], [128, 0, 255], [0, 128, 255], [128, 255, 0], [255, 0, 128],
], dtype=np.uint8)


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ["model", "model_ema", "state_dict", "model_state"]:
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def remap_old_dpt_to_multitask(state_dict):
    """兼容原始 DPT checkpoint：head.xxx -> heads.seg.xxx。"""
    new_sd = dict(state_dict)
    for k, v in list(state_dict.items()):
        if k.startswith("head."):
            new_sd.setdefault("heads.seg." + k[len("head."):], v)
        if k.startswith("heads.seg."):
            new_sd.setdefault("head." + k[len("heads.seg."):], v)
    return new_sd


def add_text_labels_matplotlib(ax, seg_pred, class_names, min_area_threshold=200, safe_margin=50):
    unique_classes = np.unique(seg_pred)
    height, width = seg_pred.shape
    for class_id in unique_classes:
        if class_id >= len(class_names):
            continue
        class_mask = seg_pred == class_id
        if np.sum(class_mask) < min_area_threshold:
            continue
        labeled_array, num_features = ndimage.label(class_mask)
        if num_features <= 0:
            continue
        component_sizes = ndimage.sum(class_mask, labeled_array, range(1, num_features + 1))
        largest_component = int(np.argmax(component_sizes) + 1)
        largest_mask = labeled_array == largest_component
        y, x = ndimage.center_of_mass(largest_mask)
        if np.isnan(y) or np.isnan(x):
            continue
        name = class_names[class_id].split(",")[0]
        x = max(safe_margin, min(float(x), width - safe_margin))
        y = max(safe_margin, min(float(y), height - safe_margin))
        ax.text(
            x, y, name, color="white", fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8,
                      edgecolor="white", linewidth=1),
        )


class MultiTaskUniMatchInferencer:
    def __init__(self, config_path, checkpoint_path, device="cuda", task="seg",
                 num_cls_classes=None, num_det_classes=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task = task

        with open(config_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.dataset = self.cfg.get("dataset", "custom")
        self.num_seg_classes = int(self.cfg.get("nclass", 2))
        self.num_cls_classes = int(num_cls_classes or self.cfg.get("num_cls_classes", self.num_seg_classes))
        self.num_det_classes = int(num_det_classes or self.cfg.get("num_det_classes", self.num_seg_classes))

        self.class_names = self._get_class_names(max(self.num_seg_classes, self.num_cls_classes, self.num_det_classes))
        self.palette = self._get_palette(max(self.num_seg_classes, self.num_det_classes))

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        self.model = self._build_model()
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _build_model(self):
        backbone_size = self.cfg["backbone"].split("_")[-1]
        model_config = {**model_configs[backbone_size], "nclass": self.num_seg_classes}

        if HAS_MULTITASK:
            model_config.update({
                "num_cls_classes": self.num_cls_classes,
                "num_det_classes": self.num_det_classes,
            })
            print(f"Building MultiTaskDPT with config: {model_config}")
            model = MultiTaskDPT(**model_config)
        else:
            if self.task not in ["seg"]:
                raise RuntimeError(
                    "当前仓库没有 model/semseg/dpt_multitask.py，不能运行 cls/det/all。"
                    "请先把 dpt_multitask.py 放到 model/semseg/ 目录。"
                )
            print(f"Building original DPT with config: {model_config}")
            model = DPT(**model_config)

        return model.to(self.device)

    def _load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = extract_state_dict(checkpoint)
        state_dict = strip_module_prefix(state_dict)
        if HAS_MULTITASK:
            state_dict = remap_old_dpt_to_multitask(state_dict)

        # 如果用户传的是 backbone_xxx.pth，也支持只加载 backbone。
        model_sd = self.model.state_dict()
        if all(k.startswith("backbone.") or not isinstance(v, torch.Tensor) for k, v in state_dict.items()):
            pass
        elif not any(k.startswith("backbone.") or k.startswith("head.") or k.startswith("heads.") for k in state_dict.keys()):
            state_dict = {"backbone." + k: v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print("✓ Checkpoint loaded with strict=False")
        if missing:
            short_missing = [k for k in missing if k.startswith("heads.cls") or k.startswith("heads.det")]
            if short_missing:
                print("⚠ cls/det head 未在 checkpoint 中找到，当前 head 可能是随机初始化。")
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def _get_palette(self, n):
        if self.dataset == "loveda":
            palette = np.array([
                [0, 200, 0], [100, 149, 237], [102, 205, 170], [205, 133, 63],
                [160, 32, 240], [255, 64, 64], [139, 69, 19], [0, 255, 0],
            ], dtype=np.uint8)
        elif self.dataset == "potsdam":
            palette = np.array([
                [255, 255, 255], [0, 0, 255], [0, 255, 255],
                [0, 255, 0], [255, 255, 0], [255, 0, 0],
            ], dtype=np.uint8)
        elif self.dataset == "landslide":
            palette = np.array([
                [0, 0, 0], [255, 0, 0],
            ], dtype=np.uint8)
        else:
            reps = int(np.ceil(n / len(DEFAULT_COLORS)))
            palette = np.tile(DEFAULT_COLORS, (reps, 1))[:n]
        return palette

    def _get_class_names(self, n):
        if "class_names" in self.cfg and isinstance(self.cfg["class_names"], list):
            names = list(self.cfg["class_names"])
        elif self.dataset == "cityscapes":
            names = [
                "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                "truck", "bus", "train", "motorcycle", "bicycle",
            ]
        elif self.dataset == "potsdam":
            names = ["Impervious", "Building", "Low Veg", "Tree", "Car", "Clutter"]
        elif self.dataset == "loveda":
            names = ["Background", "Road", "Building", "Agriculture", "Water", "Barren", "Forest"]
        elif self.dataset == "landslide":
            names = ["Background", "Landslide"]
        else:
            names = [f"Class {i}" for i in range(n)]
        while len(names) < n:
            names.append(f"Class {len(names)}")
        return names

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size[::-1]  # H, W
        h, w = original_size
        patch_size = 14
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size
        if (new_h, new_w) != (h, w):
            print(f"  Resizing image from {original_size} to ({new_h}, {new_w})")
            image_resized = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            image_resized = image

        arr = np.array(image_resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor, original_size, image

    @torch.no_grad()
    def predict(self, image_path, task="seg"):
        image_tensor, original_size, original_image = self.preprocess(image_path)
        results = {}

        if task == "seg":
            seg_logits = self.model(image_tensor, task="seg") if HAS_MULTITASK else self.model(image_tensor)
            results["seg"] = self._postprocess_seg(seg_logits, original_size)
        elif task == "cls":
            cls_logits = self.model(image_tensor, task="cls")
            results["cls"] = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
        elif task == "det":
            results["det"] = self.model(image_tensor, task="det")
            results["det"] = {k: v.detach().cpu() for k, v in results["det"].items()}
        elif task == "all":
            outputs = self.model(image_tensor, task="all")
            if "seg" in outputs:
                results["seg"] = self._postprocess_seg(outputs["seg"], original_size)
            if "cls" in outputs:
                results["cls"] = torch.softmax(outputs["cls"], dim=1).squeeze(0).cpu().numpy()
            if "det" in outputs:
                results["det"] = {k: v.detach().cpu() for k, v in outputs["det"].items()}
        else:
            raise ValueError(f"Unsupported task: {task}")

        return results, original_image

    def _postprocess_seg(self, seg_logits, original_size):
        if seg_logits.shape[-2:] != original_size:
            seg_logits = F.interpolate(seg_logits, size=original_size, mode="bilinear", align_corners=True)
        return seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def visualize_mask(self, pred_mask):
        h, w = pred_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(min(len(self.palette), int(pred_mask.max()) + 1)):
            colored[pred_mask == i] = self.palette[i]
        return colored

    def get_font(self, image_size):
        font_size = max(12, min(48, min(image_size) // 25))
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for p in font_paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, font_size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def save_seg_outputs(self, image_name, original_image, pred_mask, output_dir,
                         save_overlay=True, save_index=True, save_binary=True,
                         show_labels=True, min_area_threshold=200,
                         use_matplotlib=False):
        color_dir = os.path.join(output_dir, "seg_color")
        index_dir = os.path.join(output_dir, "seg_index")
        binary_dir = os.path.join(output_dir, "seg_binary")
        overlay_dir = os.path.join(output_dir, "seg_overlay")
        os.makedirs(color_dir, exist_ok=True)

        color = self.visualize_mask(pred_mask)
        Image.fromarray(color).save(os.path.join(color_dir, f"{image_name}_seg_color.png"))
        print(f"✓ Saved seg color: {os.path.join(color_dir, f'{image_name}_seg_color.png')}")

        if save_binary:
            os.makedirs(binary_dir, exist_ok=True)
            binary_mask = np.where(pred_mask > 0, 255, 0).astype(np.uint8)
            Image.fromarray(binary_mask, mode="L").save(
                os.path.join(binary_dir, f"{image_name}_seg_binary.png"))
            print(f"✓ Saved seg binary: {os.path.join(binary_dir, f'{image_name}_seg_binary.png')}")

        if save_index:
            os.makedirs(index_dir, exist_ok=True)
            Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(index_dir, f"{image_name}_seg_index.png"))
            print(f"✓ Saved seg index: {os.path.join(index_dir, f'{image_name}_seg_index.png')}")

        if save_overlay:
            os.makedirs(overlay_dir, exist_ok=True)
            overlay_path = os.path.join(overlay_dir, f"{image_name}_seg_overlay.png")
            original_np = np.array(original_image)
            overlay = (original_np * 0.5 + color * 0.5).astype(np.uint8)

            if use_matplotlib and show_labels:
                h, w = pred_mask.shape
                dpi = 150
                fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
                ax.imshow(overlay)
                ax.axis("off")
                add_text_labels_matplotlib(ax, pred_mask, self.class_names, min_area_threshold)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
                plt.close(fig)
            else:
                overlay_img = Image.fromarray(overlay)
                if show_labels:
                    self._draw_seg_labels(overlay_img, pred_mask, min_area_threshold)
                overlay_img.save(overlay_path)
            print(f"✓ Saved seg overlay: {overlay_path}")

    def _draw_seg_labels(self, image, pred_mask, min_area_threshold):
        draw = ImageDraw.Draw(image)
        font = self.get_font(pred_mask.shape)
        h, w = pred_mask.shape
        for label_id in np.unique(pred_mask):
            if label_id >= len(self.class_names):
                continue
            mask = pred_mask == label_id
            if np.sum(mask) < min_area_threshold:
                continue
            labeled, n = ndimage.label(mask)
            if n <= 0:
                continue
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes) + 1)
            y, x = ndimage.center_of_mass(labeled == largest)
            if np.isnan(y) or np.isnan(x):
                continue
            name = self.class_names[int(label_id)].split(",")[0]
            x = max(50, min(float(x), w - 50))
            y = max(50, min(float(y), h - 50))
            bbox = draw.textbbox((0, 0), name, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = int(x - tw / 2), int(y - th / 2)
            draw.rectangle([tx - 4, ty - 4, tx + tw + 4, ty + th + 4], fill=(0, 0, 0), outline=(255, 255, 255))
            draw.text((tx, ty), name, font=font, fill=(255, 255, 255))

    def save_cls_outputs(self, image_name, cls_prob, output_dir, topk=5):
        os.makedirs(os.path.join(output_dir, "cls"), exist_ok=True)
        topk = min(topk, len(cls_prob))
        indices = np.argsort(cls_prob)[::-1][:topk]
        lines = []
        print("\nClassification Top-k:")
        for rank, idx in enumerate(indices, 1):
            name = self.class_names[int(idx)] if idx < len(self.class_names) else f"Class {idx}"
            line = f"{rank}. {name}: {float(cls_prob[idx]):.6f}"
            print("  " + line)
            lines.append(line)
        save_path = os.path.join(output_dir, "cls", f"{image_name}_cls_topk.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"✓ Saved cls result: {save_path}")

    @staticmethod
    def _box_iou(box, boxes):
        """计算单个框与一组框的 IoU，boxes 为 numpy array: [N, 4]。"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _nms(self, candidates, iou_thresh=0.5, max_boxes=10, class_aware=True):
        """简单 NMS：默认同类别之间抑制，避免密集重复框。"""
        if not candidates:
            return []

        final = []
        class_ids = sorted(set(c["class_id"] for c in candidates)) if class_aware else [None]

        for cid in class_ids:
            group = candidates if cid is None else [c for c in candidates if c["class_id"] == cid]
            group = sorted(group, key=lambda x: x["score"], reverse=True)

            while group and len(final) < max_boxes:
                best = group.pop(0)
                final.append(best)
                if not group:
                    break
                boxes = np.array([g["box"] for g in group], dtype=np.float32)
                ious = self._box_iou(np.array(best["box"], dtype=np.float32), boxes)
                group = [g for g, iou in zip(group, ious) if iou < iou_thresh]

        final = sorted(final, key=lambda x: x["score"], reverse=True)[:max_boxes]
        return final

    def decode_det(self, det_output, original_size, score_thresh=0.6, pre_topk=200,
                   max_boxes=10, nms_iou=0.45, min_box_size=20, det_class_id=None):
        logits = det_output["pred_logits"][0]  # C,h,w
        obj = det_output.get("pred_obj", torch.zeros(1, 1, *logits.shape[-2:]))[0, 0].sigmoid()  # h,w
        boxes_raw = det_output.get("pred_boxes", torch.zeros(1, 4, *logits.shape[-2:]))[0]  # 4,h,w

        cls_prob = logits.softmax(dim=0)
        cls_score, cls_id = cls_prob.max(dim=0)
        score = (cls_score * obj).flatten()

        k = min(pre_topk, score.numel())
        vals, inds = torch.topk(score, k=k)
        h0, w0 = original_size
        gh, gw = obj.shape
        candidates = []

        for s, ind in zip(vals, inds):
            s = float(s.item())
            if s < score_thresh:
                continue

            y = int(ind.item() // gw)
            x = int(ind.item() % gw)
            cid = int(cls_id[y, x].item())
            if det_class_id is not None and cid != int(det_class_id):
                continue

            raw = boxes_raw[:, y, x].sigmoid().numpy()
            cx = (x + 0.5) / gw * w0
            cy = (y + 0.5) / gh * h0

            bw = max(float(min_box_size), float(raw[2]) * w0 * 0.7)
            bh = max(float(min_box_size), float(raw[3]) * h0 * 0.7)

            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(float(w0 - 1), cx + bw / 2)
            y2 = min(float(h0 - 1), cy + bh / 2)

            if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                continue

            candidates.append({"box": [x1, y1, x2, y2], "score": s, "class_id": cid})

        candidates = self._nms(candidates, iou_thresh=nms_iou, max_boxes=max_boxes, class_aware=True)
        return candidates, obj.numpy()

    def save_det_outputs(self, image_name, original_image, det_output, output_dir,
                         score_thresh=0.6, pre_topk=200, max_boxes=10,
                         nms_iou=0.45, min_box_size=20, det_class_id=None):
        det_dir = os.path.join(output_dir, "det")
        os.makedirs(det_dir, exist_ok=True)
        original_size = original_image.size[::-1]
        candidates, heatmap = self.decode_det(
            det_output, original_size,
            score_thresh=score_thresh,
            pre_topk=pre_topk,
            max_boxes=max_boxes,
            nms_iou=nms_iou,
            min_box_size=min_box_size,
            det_class_id=det_class_id,
        )

        # heatmap
        heat = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(original_image.size, Image.BILINEAR)
        heat_path = os.path.join(det_dir, f"{image_name}_det_heatmap.png")
        heat_img.save(heat_path)
        print(f"✓ Saved det heatmap: {heat_path}")

        # boxes overlay
        vis = original_image.copy()
        draw = ImageDraw.Draw(vis)
        font = self.get_font(original_size)
        txt_lines = []
        for item in candidates:
            x1, y1, x2, y2 = item["box"]
            cid = item["class_id"]
            score = item["score"]
            name = self.class_names[cid] if cid < len(self.class_names) else f"Class {cid}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            label = f"{name} {score:.2f}"
            draw.text((x1, max(0, y1 - 16)), label, font=font, fill=(255, 0, 0))
            txt_lines.append(f"{name}\t{score:.6f}\t{x1:.1f}\t{y1:.1f}\t{x2:.1f}\t{y2:.1f}")

        box_path = os.path.join(det_dir, f"{image_name}_det_boxes.png")
        vis.save(box_path)
        print(f"✓ Saved det boxes: {box_path}")

        txt_path = os.path.join(det_dir, f"{image_name}_det_boxes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("class\tscore\tx1\ty1\tx2\ty2\n")
            f.write("\n".join(txt_lines))
            if txt_lines:
                f.write("\n")
        print(f"✓ Saved det txt: {txt_path}")
        if not candidates:
            print("⚠ No detection candidates above threshold. 如果 det head 没训练过，这是正常的。")

    def inference_single(self, image_path, output_dir, task="seg", save_overlay=True, save_index=True,
                         show_labels=True, min_area_threshold=200, use_matplotlib=False,
                         cls_topk=5, det_score_thresh=0.6, det_pre_topk=200,
                         det_max_boxes=10, det_nms_iou=0.45,
                         det_min_box_size=20, det_class_id=None):
        print("\n" + "=" * 60)
        print(f"Processing: {image_path}")
        print(f"Task: {task}")
        print("=" * 60)

        results, original_image = self.predict(image_path, task)
        image_name = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)

        if "seg" in results:
            self.save_seg_outputs(
                image_name, original_image, results["seg"], output_dir,
                save_overlay=save_overlay, save_index=save_index,
                show_labels=show_labels, min_area_threshold=min_area_threshold,
                use_matplotlib=use_matplotlib,
            )
        if "cls" in results:
            self.save_cls_outputs(image_name, results["cls"], output_dir, topk=cls_topk)
        if "det" in results:
            self.save_det_outputs(
                image_name, original_image, results["det"], output_dir,
                score_thresh=det_score_thresh,
                pre_topk=det_pre_topk,
                max_boxes=det_max_boxes,
                nms_iou=det_nms_iou,
                min_box_size=det_min_box_size,
                det_class_id=det_class_id,
            )

    def inference_batch(self, image_dir, output_dir, **kwargs):
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        image_files = []
        for ext in exts:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))
        print(f"Found {len(image_files)} images in {image_dir}")
        for i, p in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}]", end=" ")
            try:
                self.inference_single(str(p), output_dir, **kwargs)
            except Exception as e:
                print(f"✗ Error processing {p}: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task UniMatch/DPT Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./predictions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--task", type=str, default="seg", choices=["seg", "cls", "det", "all"])

    # 类别数；不填则从 yaml 中读取 num_cls_classes / num_det_classes，否则默认 nclass
    parser.add_argument("--num-cls-classes", type=int, default=None)
    parser.add_argument("--num-det-classes", type=int, default=None)

    # segmentation options，兼容你原来的命令
    parser.add_argument("--save-overlay", action="store_true", default=True)
    parser.add_argument("--no-save-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--save-index", action="store_true", default=True)
    parser.add_argument("--no-save-index", dest="save_index", action="store_false")
    parser.add_argument("--show-labels", action="store_true", default=True)
    parser.add_argument("--no-show-labels", dest="show_labels", action="store_false")
    parser.add_argument("--min-area-threshold", type=int, default=200)
    parser.add_argument("--use-matplotlib", action="store_true", default=False)

    # classification / detection options
    parser.add_argument("--cls-topk", type=int, default=5)
    parser.add_argument("--det-score-thresh", type=float, default=0.6,
                        help="检测框分数阈值；框太多就调大，例如 0.7/0.8")
    parser.add_argument("--det-pre-topk", type=int, default=200,
                        help="NMS 前最多保留多少个候选点")
    parser.add_argument("--det-max-boxes", type=int, default=10,
                        help="NMS 后最多画多少个框；框太多就调小")
    parser.add_argument("--det-nms-iou", type=float, default=0.45,
                        help="NMS IoU 阈值；重复框太多就调小，例如 0.3")
    parser.add_argument("--det-min-box-size", type=int, default=20,
                        help="过滤小框，单位像素")
    parser.add_argument("--det-class-id", type=int, default=None,
                        help="只显示指定类别 id 的检测框，例如只看滑坡类")

    args = parser.parse_args()
    if args.image is None and args.image_dir is None:
        parser.error("必须指定 --image 或 --image-dir 之一")
    if args.image is not None and args.image_dir is not None:
        parser.error("--image 和 --image-dir 不能同时指定")

    print("\n" + "=" * 60)
    print("Multi-task UniMatch/DPT Inference Configuration")
    print("=" * 60)
    print(f"Config:          {args.config}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Task:            {args.task}")
    print(f"Device:          {args.device}")
    print(f"Output dir:      {args.output_dir}")
    print("=" * 60 + "\n")

    inferencer = MultiTaskUniMatchInferencer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        task=args.task,
        num_cls_classes=args.num_cls_classes,
        num_det_classes=args.num_det_classes,
    )

    common_kwargs = dict(
        task=args.task,
        save_overlay=args.save_overlay,
        save_index=args.save_index,
        show_labels=args.show_labels,
        min_area_threshold=args.min_area_threshold,
        use_matplotlib=args.use_matplotlib,
        cls_topk=args.cls_topk,
        det_score_thresh=args.det_score_thresh,
        det_pre_topk=args.det_pre_topk,
        det_max_boxes=args.det_max_boxes,
        det_nms_iou=args.det_nms_iou,
        det_min_box_size=args.det_min_box_size,
        det_class_id=args.det_class_id,
    )
    if args.image:
        inferencer.inference_single(args.image, args.output_dir, **common_kwargs)
    else:
        inferencer.inference_batch(args.image_dir, args.output_dir, **common_kwargs)

    print("\n✓ Inference completed successfully!")
    print(f"✓ Results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
