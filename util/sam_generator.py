from __future__ import annotations

import os
import pickle
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (C,H,W) tensor → uint8 (H,W,3) numpy array."""
    img = tensor.cpu() * _IMAGENET_STD + _IMAGENET_MEAN
    img = (img.clamp(0.0, 1.0) * 255).byte().numpy()
    return img.transpose(1, 2, 0)                      # (H, W, 3)


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image → uint8 (H, W, 3) numpy array."""
    return np.array(img.convert('RGB'))


class _BaseSAMPseudoLabelGenerator:
    def __init__(
        self,
        nclass: int,
        conf_threshold: float = 0.70,
        purity_threshold: float = 0.60,
        min_mask_area: int = 100,
        device: str = 'cuda',
    ):
        self.nclass = nclass
        self.conf_threshold = conf_threshold
        self.purity_threshold = purity_threshold
        self.min_mask_area = min_mask_area
        self.device = device
        self.mask_generator = None

    def _generate_masks(self, img_np: np.ndarray) -> list[dict]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        teacher_model: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = images.shape

        teacher_probs = teacher_model(images).softmax(dim=1)
        teacher_conf, teacher_cls = teacher_probs.max(dim=1)

        pseudo_labels_list: list[torch.Tensor] = []
        quality_masks_list: list[torch.Tensor] = []

        for idx in range(B):
            img_np = _tensor_to_uint8(images[idx])
            t_conf = teacher_conf[idx].cpu()
            t_cls  = teacher_cls[idx].cpu()

            try:
                sam_masks = self._generate_masks(img_np)
                pl, qm = self._assign_classes(sam_masks, t_cls, t_conf, H, W)
            except Exception as exc:
                logger.warning(f'SAM failed on sample {idx}: {exc}. '
                               f'Falling back to teacher pseudo-labels.')
                pl, qm = self._teacher_fallback(t_cls, t_conf)

            pseudo_labels_list.append(pl)
            quality_masks_list.append(qm)

        pseudo_labels = torch.stack(pseudo_labels_list).to(images.device)
        quality_masks = torch.stack(quality_masks_list).to(images.device)
        return pseudo_labels, quality_masks

    def pregenerate(
        self,
        image_paths: list[str],
        output_dir: str,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for path in image_paths:
            img_np = np.array(Image.open(path).convert('RGB'))
            masks = self._generate_masks(img_np)
            payload = [
                {
                    'segmentation':    m['segmentation'],
                    'predicted_iou':   float(m.get('predicted_iou',   1.0)),
                    'stability_score': float(m.get('stability_score', 1.0)),
                    'area':            int(m['area']),
                }
                for m in masks
            ]
            name = os.path.splitext(os.path.basename(path))[0] + '.pkl'
            with open(os.path.join(output_dir, name), 'wb') as fh:
                pickle.dump(payload, fh)
        logger.info(f'Pre-generated SAM masks for {len(image_paths)} images → {output_dir}')

    @staticmethod
    def load_pregenerated(pkl_path: str) -> list[dict]:
        """Load a pre-generated mask file."""
        with open(pkl_path, 'rb') as fh:
            return pickle.load(fh)

    @torch.no_grad()
    def generate_from_pregenerated(
        self,
        images: torch.Tensor,
        teacher_model: torch.nn.Module,
        sam_mask_files: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = images.shape
        teacher_probs = teacher_model(images).softmax(dim=1)
        teacher_conf, teacher_cls = teacher_probs.max(dim=1)

        pseudo_labels_list, quality_masks_list = [], []
        for idx in range(B):
            t_conf = teacher_conf[idx].cpu()
            t_cls  = teacher_cls[idx].cpu()

            try:
                sam_masks = self.load_pregenerated(sam_mask_files[idx])
                pl, qm = self._assign_classes(sam_masks, t_cls, t_conf, H, W)
            except Exception as exc:
                logger.warning(f'Load/assign failed for {sam_mask_files[idx]}: {exc}')
                pl, qm = self._teacher_fallback(t_cls, t_conf)

            pseudo_labels_list.append(pl)
            quality_masks_list.append(qm)

        return (
            torch.stack(pseudo_labels_list).to(images.device),
            torch.stack(quality_masks_list).to(images.device),
        )

    def _assign_classes(
        self,
        sam_masks: list[dict],
        teacher_cls: torch.Tensor,
        teacher_conf: torch.Tensor,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pseudo_label = torch.full((H, W), 255, dtype=torch.long)
        quality_mask = torch.zeros((H, W), dtype=torch.bool)

        sam_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)

        for m in sam_masks:
            seg = torch.from_numpy(m['segmentation'].astype(bool))

            if seg.sum() == 0:
                continue

            seg_cls  = teacher_cls[seg]
            seg_conf = teacher_conf[seg]

            cls_hist = torch.bincount(seg_cls, minlength=self.nclass).float()
            dominant = int(cls_hist.argmax())
            purity   = (cls_hist[dominant] / cls_hist.sum()).item()
            mean_conf = seg_conf.mean().item()

            if purity >= self.purity_threshold and mean_conf >= self.conf_threshold:
                pseudo_label[seg] = dominant
                quality_mask[seg] = True

        return pseudo_label, quality_mask

    def _teacher_fallback(
        self,
        teacher_cls: torch.Tensor,
        teacher_conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """When SAM fails, fall back to raw teacher predictions with threshold."""
        quality = teacher_conf >= self.conf_threshold
        label   = torch.where(quality, teacher_cls, torch.full_like(teacher_cls, 255))
        return label, quality


class SAM2PseudoLabelGenerator(_BaseSAMPseudoLabelGenerator):
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str,
        nclass: int,
        conf_threshold: float = 0.70,
        purity_threshold: float = 0.60,
        sam2_iou_thresh: float = 0.80,
        sam2_stability_thresh: float = 0.85,
        min_mask_area: int = 100,
        device: str = 'cuda',
    ):
        super().__init__(
            nclass=nclass,
            conf_threshold=conf_threshold,
            purity_threshold=purity_threshold,
            min_mask_area=min_mask_area,
            device=device,
        )

        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            pred_iou_thresh=sam2_iou_thresh,
            stability_score_thresh=sam2_stability_thresh,
            min_mask_region_area=min_mask_area,
        )
        logger.info(
            f'SAM2PseudoLabelGenerator ready | conf≥{conf_threshold} '
            f'purity≥{purity_threshold} iou≥{sam2_iou_thresh} '
            f'stability≥{sam2_stability_thresh}'
        )

    def _generate_masks(self, img_np: np.ndarray) -> list[dict]:
        """Generate masks using SAM2 automatic mask generator."""
        return self.mask_generator.generate(img_np)


class SAM3HFPseudoLabelGenerator(_BaseSAMPseudoLabelGenerator):
    def __init__(
        self,
        nclass: int,
        conf_threshold: float = 0.70,
        purity_threshold: float = 0.60,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.85,
        min_mask_area: int = 100,
        device: str = 'cuda',
        sam3_model_name: str = 'facebook/sam3',
    ):
        super().__init__(
            nclass=nclass,
            conf_threshold=conf_threshold,
            purity_threshold=purity_threshold,
            min_mask_area=min_mask_area,
            device=device,
        )

        try:
            from transformers import pipeline

            self._points_per_batch = points_per_batch
            self._pred_iou_thresh = pred_iou_thresh
            self._stability_score_thresh = stability_score_thresh

            pipe_device = 0 if device == 'cuda' else -1
            self.mask_generator = pipeline(
                "mask-generation",
                model=sam3_model_name,
                device=pipe_device,
            )
            logger.info(
                f'SAM3HFPseudoLabelGenerator ready | model={sam3_model_name} '
                f'conf≥{conf_threshold} purity≥{purity_threshold}'
            )
        except ImportError:
            raise ImportError(
                'transformers package is required for SAM3 HF pipeline. '
                'Install it with: pip install transformers>=4.50.0'
            )

    def _generate_masks(self, img_np: np.ndarray) -> list[dict]:
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(img_np)
        results = self.mask_generator(
            pil_image,
            points_per_batch=self._points_per_batch,
            pred_iou_thresh=self._pred_iou_thresh,
            stability_score_thresh=self._stability_score_thresh,
            min_mask_region_area=self.min_mask_area,
        )

        masks = results.get('masks', [])
        if isinstance(masks, list) and len(masks) > 0:
            if isinstance(masks[0], dict):
                return masks
            elif isinstance(masks[0], np.ndarray):
                return [{'segmentation': m.astype(bool), 'area': int(m.sum())}
                        for m in masks]

        if isinstance(masks, np.ndarray):
            return [{'segmentation': m.astype(bool), 'area': int(m.sum())}
                    for m in masks]

        return []


class SAM3OfficialPseudoLabelGenerator(_BaseSAMPseudoLabelGenerator):
    def __init__(
        self,
        sam3_checkpoint: str,
        nclass: int,
        conf_threshold: float = 0.70,
        purity_threshold: float = 0.60,
        text_prompts: Optional[list[str]] = None,
        min_mask_area: int = 100,
        device: str = 'cuda',
    ):
        super().__init__(
            nclass=nclass,
            conf_threshold=conf_threshold,
            purity_threshold=purity_threshold,
            min_mask_area=min_mask_area,
            device=device,
        )
        self.text_prompts = text_prompts or []

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.model = build_sam3_image_model(
                checkpoint_path=sam3_checkpoint,
                device='cpu',
                eval_mode=True,
            )
            self.model = self.model.to(device=device, dtype=torch.bfloat16)
            param_dtype = next(self.model.parameters()).dtype
            first_param_device = next(self.model.parameters()).device
            logger.info(
                f'SAM3 model loaded: dtype={param_dtype}, device={first_param_device}'
            )
            self.processor = Sam3Processor(self.model)
            self._sam3_device = device
            logger.info(
                f'SAM3OfficialPseudoLabelGenerator ready | checkpoint={sam3_checkpoint}'
            )
        except ImportError:
            raise ImportError(
                'SAM3 package is required. Install from: '
                'pip install git+https://github.com/facebookresearch/sam3.git'
            )

    def _generate_masks(self, img_np: np.ndarray) -> list[dict]:
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(img_np)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    return self._generate_masks_impl(pil_image)
            else:
                return self._generate_masks_impl(pil_image)

    def _generate_masks_impl(self, pil_image: Image.Image) -> list[dict]:
        if self.text_prompts:
            return self._generate_with_text_prompts(pil_image)
        else:
            return self._generate_automatic(pil_image)

    def _generate_with_text_prompts(self, pil_image: Image.Image) -> list[dict]:
        """Generate masks using text prompts (concept segmentation)."""
        inference_state = self.processor.set_image(pil_image)
        all_masks = []

        for prompt in self.text_prompts:
            try:
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt,
                )
                masks = output.get('masks', [])
                boxes = output.get('boxes', [])
                scores = output.get('scores', [])

                if masks is not None and len(masks) > 0:
                    for i, mask in enumerate(masks):
                        if isinstance(mask, torch.Tensor):
                            mask_np = mask.cpu().numpy().astype(bool)
                        elif isinstance(mask, np.ndarray):
                            mask_np = mask.astype(bool)
                        else:
                            continue

                        if mask_np.sum() >= self.min_mask_area:
                            entry = {
                                'segmentation': mask_np,
                                'area': int(mask_np.sum()),
                                'predicted_iou': float(scores[i]) if scores is not None and i < len(scores) else 1.0,
                                'stability_score': float(scores[i]) if scores is not None and i < len(scores) else 1.0,
                            }
                            if boxes is not None and i < len(boxes):
                                entry['bbox'] = boxes[i].tolist() if isinstance(boxes[i], (list, tuple, np.ndarray)) else list(boxes[i])
                            all_masks.append(entry)
            except Exception as exc:
                logger.warning(f'Text prompt "{prompt}" failed: {exc}')

        return all_masks

    def _generate_automatic(self, pil_image: Image.Image) -> list[dict]:
        """Generate masks automatically by prompting with common categories."""
        inference_state = self.processor.set_image(pil_image)

        default_prompts = ['object', 'person', 'vehicle', 'building',
                          'tree', 'sky', 'road', 'animal']
        all_masks = []

        for prompt in default_prompts:
            try:
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt,
                )
                masks = output.get('masks', [])
                scores = output.get('scores', [])

                if masks is not None and len(masks) > 0:
                    for i, mask in enumerate(masks):
                        if isinstance(mask, torch.Tensor):
                            mask_np = mask.cpu().numpy().astype(bool)
                        elif isinstance(mask, np.ndarray):
                            mask_np = mask.astype(bool)
                        else:
                            continue

                        if mask_np.sum() >= self.min_mask_area:
                            entry = {
                                'segmentation': mask_np,
                                'area': int(mask_np.sum()),
                                'predicted_iou': float(scores[i]) if scores is not None and i < len(scores) else 1.0,
                                'stability_score': float(scores[i]) if scores is not None and i < len(scores) else 1.0,
                            }
                            all_masks.append(entry)
            except Exception:
                pass

        return self._deduplicate_masks(all_masks)

    @staticmethod
    def _deduplicate_masks(masks: list[dict]) -> list[dict]:
        """Remove duplicate/overlapping masks, keeping the larger one."""
        if len(masks) <= 1:
            return masks

        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        kept = []
        for m in masks:
            seg = m['segmentation']
            is_dup = False
            for k in kept:
                intersection = (seg & k['segmentation']).sum()
                min_area = min(seg.sum(), k['segmentation'].sum())
                if min_area > 0 and (intersection / min_area) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(m)
        return kept


def create_sam_generator(
    sam_version: str,
    nclass: int,
    sam_checkpoint: Optional[str] = None,
    sam_config: Optional[str] = None,
    conf_threshold: float = 0.70,
    purity_threshold: float = 0.60,
    min_mask_area: int = 100,
    device: str = 'cuda',
    **kwargs,
) -> _BaseSAMPseudoLabelGenerator:
    if sam_version == 'sam2':
        if sam_checkpoint is None:
            raise ValueError('sam_checkpoint is required for SAM2')
        if sam_config is None:
            raise ValueError('sam_config is required for SAM2')
        return SAM2PseudoLabelGenerator(
            sam2_checkpoint=sam_checkpoint,
            sam2_config=sam_config,
            nclass=nclass,
            conf_threshold=conf_threshold,
            purity_threshold=purity_threshold,
            min_mask_area=min_mask_area,
            device=device,
            **kwargs,
        )
    elif sam_version == 'sam3_hf':
        resolved_checkpoint = None
        if sam_checkpoint:
            if os.path.exists(sam_checkpoint):
                resolved_checkpoint = sam_checkpoint
            else:
                abs_path = os.path.abspath(sam_checkpoint)
                logger.warning(
                    f'Checkpoint path not found: {sam_checkpoint} '
                    f'(resolved: {abs_path}). '
                    f'Will try to download from HuggingFace.'
                )

        if resolved_checkpoint:
            if resolved_checkpoint.endswith('.pt') or resolved_checkpoint.endswith('.pth'):
                logger.info(
                    f'Detected local .pt checkpoint: {resolved_checkpoint}. '
                    f'Using SAM3OfficialPseudoLabelGenerator instead.'
                )
                return SAM3OfficialPseudoLabelGenerator(
                    sam3_checkpoint=resolved_checkpoint,
                    nclass=nclass,
                    conf_threshold=conf_threshold,
                    purity_threshold=purity_threshold,
                    min_mask_area=min_mask_area,
                    device=device,
                    **kwargs,
                )
            elif os.path.isdir(resolved_checkpoint):
                logger.info(
                    f'Detected local HF model directory: {resolved_checkpoint}'
                )
                return SAM3HFPseudoLabelGenerator(
                    nclass=nclass,
                    conf_threshold=conf_threshold,
                    purity_threshold=purity_threshold,
                    min_mask_area=min_mask_area,
                    device=device,
                    sam3_model_name=resolved_checkpoint,
                    **kwargs,
                )
        return SAM3HFPseudoLabelGenerator(
            nclass=nclass,
            conf_threshold=conf_threshold,
            purity_threshold=purity_threshold,
            min_mask_area=min_mask_area,
            device=device,
            **kwargs,
        )
    elif sam_version == 'sam3_official':
        if sam_checkpoint is None:
            raise ValueError('sam_checkpoint is required for SAM3 Official')
        return SAM3OfficialPseudoLabelGenerator(
            sam3_checkpoint=sam_checkpoint,
            nclass=nclass,
            conf_threshold=conf_threshold,
            purity_threshold=purity_threshold,
            min_mask_area=min_mask_area,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(
            f'Unknown SAM version: {sam_version}. '
            f'Choose from: sam2, sam3_hf, sam3_official'
        )


SAMPseudoLabelGenerator = SAM2PseudoLabelGenerator