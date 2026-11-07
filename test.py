import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from scipy import ndimage

from dataset.semi import SemiDataset

try:
    from model.semseg.dpt import DPT
except ImportError:
    print("Error: 找不到 model.semseg.dpt。请确保你在项目根目录下运行脚本。")
    exit(1)

def get_parser():
    parser = argparse.ArgumentParser(description='UniMatch-V2 Testing')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best.pth')
    parser.add_argument('--save-path', type=str, default=None, help='path to save visual results')
    parser.add_argument('--det-iou-thresh', type=float, default=0.5,
                        help='IoU threshold for box matching (default: 0.5)')
    parser.add_argument('--det-min-area', type=int, default=20,
                        help='min connected-component area (pixels) to form a box')
    parser.add_argument('--det-class-id', type=int, default=1,
                        help='class id treated as the detection target (1 = landslide)')
    parser.add_argument('--det-closing', type=int, default=0,
                        help='morphological closing kernel size applied to BOTH GT and '
                             'pred masks before box extraction (0 = disabled)')
    parser.add_argument('--det-single-box', action='store_true',
                        help='also report single-box-per-image detection metrics '
                             '(one global bbox over all class pixels, robust to fragmentation)')
    return parser

def intersectionAndUnion(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size).copy()

    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]

    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))

    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def mask_to_boxes(mask, cls_id, min_area=20, closing_kernel=0):
    binary = (mask == cls_id)
    if closing_kernel and closing_kernel > 0:
        binary = ndimage.binary_closing(
            binary, structure=np.ones((closing_kernel, closing_kernel), dtype=bool))
    labeled, n = ndimage.label(binary)
    if n == 0:
        return []
    areas = np.bincount(labeled.ravel())
    slices = ndimage.find_objects(labeled)
    boxes = []
    for k in range(1, n + 1):
        if areas[k] < min_area:
            continue
        sl = slices[k - 1]
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
    return boxes

def mask_to_single_box(mask, cls_id, min_area=20, closing_kernel=0):
    binary = (mask == cls_id)
    if closing_kernel and closing_kernel > 0:
        binary = ndimage.binary_closing(
            binary, structure=np.ones((closing_kernel, closing_kernel), dtype=bool))
    if int(binary.sum()) < min_area:
        return None
    ys, xs = np.where(binary)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def single_box_match(gt_box, pred_box, thresh):
    """Detection outcome for one image under single-box evaluation."""
    if gt_box is None and pred_box is None:
        return 0, 0, 0          # true negative, not counted
    if gt_box is None:
        return 0, 1, 0          # false positive
    if pred_box is None:
        return 0, 0, 1          # false negative
    iou = float(box_iou_matrix([gt_box], [pred_box])[0, 0])
    return (1, 0, 0) if iou >= thresh else (0, 1, 1)

def box_iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU matrix between two lists of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.array(boxes_a, dtype=np.float32)
    b = np.array(boxes_b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)

def greedy_match(iou, thresh):
    """One-to-one greedy matching by descending IoU. Returns (tp, fp, fn)."""
    n_gt, n_pred = iou.shape
    matched_gt = np.zeros(n_gt, dtype=bool)
    matched_pred = np.zeros(n_pred, dtype=bool)
    if n_gt > 0 and n_pred > 0:
        for idx in np.argsort(-iou, axis=None):
            gi, pi = divmod(int(idx), n_pred)
            if iou[gi, pi] < thresh:
                break
            if matched_gt[gi] or matched_pred[pi]:
                continue
            matched_gt[gi] = True
            matched_pred[pi] = True
    tp = int(matched_gt.sum())
    fn = int((~matched_gt).sum())
    fp = int((~matched_pred).sum())
    return tp, fp, fn

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f

def test(args):

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    val_dataset = SemiDataset(
        name=cfg['dataset'],
        root=cfg['data_root'],
        mode='val',
        size=cfg['crop_size']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Dataset: {cfg['dataset']}, Val images: {len(val_dataset)}")

    raw_backbone = cfg['backbone']

    features = 128
    out_channels = [96, 192, 384, 768]
    encoder_size = raw_backbone

    if 'small' in raw_backbone:
        encoder_size = 'small'
        features = 64
        out_channels = [48, 96, 192, 384]
        print("Detected Small backbone: reducing head channels (features=64).")

    elif 'base' in raw_backbone:
        encoder_size = 'base'
        features = 128
        out_channels = [96, 192, 384, 768]

    elif 'large' in raw_backbone:
        encoder_size = 'large'
        features = 256
        out_channels = [256, 512, 1024, 1024]

    elif 'giant' in raw_backbone:
        encoder_size = 'giant'

    print(f"Initializing DPT -> Size: {encoder_size}, Features: {features}, OutCh: {out_channels}")

    n_classes = cfg.get('nclass', cfg.get('num_classes'))
    model = DPT(encoder_size, n_classes, features=features, out_channels=out_channels)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded model weights (strict=True).")
    except Exception as e:
        print(f"Strict load failed: {e}")
        print("Trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)

    model.cuda()
    model.eval()

    intersection_meter = 0
    union_meter = 0
    target_meter = 0
    prediction_meter = 0

    LANDSLIDE_CLS = 1
    img_y_true = []   # 1 if GT mask contains class 1, else 0
    img_y_pred = []   # 1 if predicted mask contains class 1, else 0
    img_score  = []   # continuous score = max class-1 probability (for AUC)

    DET_CLS = args.det_class_id
    det_iou_sum = 0.0     # sum of best-match IoU over GT boxes (localization)
    det_iou_count = 0     # number of GT boxes contributing to det_iou_sum
    det50 = [0, 0, 0]     # [tp, fp, fn] at IoU >= 0.5
    det75 = [0, 0, 0]     # [tp, fp, fn] at IoU >= 0.75
    # Box-count diagnostics
    det_gt_total = 0      # total GT boxes over all images
    det_pred_total = 0    # total predicted boxes over all images
    det_img_total = 0     # number of images evaluated
    det_mismatch = []     # (name, n_gt, n_pred) for images whose counts differ
    # Single-box-per-image evaluation (optional, --det-single-box)
    sb_iou_sum = 0.0      # sum of per-image IoU where both GT and pred boxes exist
    sb_iou_count = 0
    sb50 = [0, 0, 0]      # [tp, fp, fn] at IoU >= 0.5
    sb75 = [0, 0, 0]      # [tp, fp, fn] at IoU >= 0.75

    print("Start evaluating...")
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with torch.no_grad():
        for i, (img, mask, filename) in enumerate(tqdm(val_loader)):
            img = img.cuda()
            mask = mask.cuda()

            h, w = img.shape[2], img.shape[3]
            patch_size = 14

            new_h = int(math.ceil(h / patch_size) * patch_size)
            new_w = int(math.ceil(w / patch_size) * patch_size)

            if new_h != h or new_w != w:
                img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=True)

            pred = model(img)

            if isinstance(pred, tuple):
                pred = pred[0]

            target_h, target_w = mask.shape[1], mask.shape[2]

            if pred.shape[2:] != (target_h, target_w):
                pred = F.interpolate(pred, size=(target_h, target_w), mode='bilinear', align_corners=True)

            pred = torch.argmax(pred, dim=1)

            pred_np = pred.cpu().numpy()
            mask_np = mask.cpu().numpy()

            valid_gt = mask_np[0] != 255
            has_ls_gt = bool((mask_np[0][valid_gt] == LANDSLIDE_CLS).any())
            has_ls_pred = bool((pred_np[0] == LANDSLIDE_CLS).any())
            img_y_true.append(int(has_ls_gt))
            img_y_pred.append(int(has_ls_pred))
            score = (pred_np[0] == LANDSLIDE_CLS).sum() / \
                    (valid_gt.sum() + 1e-10)
            img_score.append(float(score))

            gt_boxes = mask_to_boxes(mask_np[0], DET_CLS, args.det_min_area,
                                     closing_kernel=args.det_closing)
            pred_boxes = mask_to_boxes(pred_np[0], DET_CLS, args.det_min_area,
                                       closing_kernel=args.det_closing)
            iou = box_iou_matrix(gt_boxes, pred_boxes)
            # Localization quality: best-match IoU for each GT box
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                det_iou_sum += float(iou.max(axis=1).sum())
                det_iou_count += len(gt_boxes)
            # Detection quality at IoU thresholds
            tp, fp, fn = greedy_match(iou, 0.5)
            det50[0] += tp; det50[1] += fp; det50[2] += fn
            tp, fp, fn = greedy_match(iou, 0.75)
            det75[0] += tp; det75[1] += fp; det75[2] += fn
            # Box-count diagnostics
            det_gt_total += len(gt_boxes)
            det_pred_total += len(pred_boxes)
            det_img_total += 1
            if len(gt_boxes) != len(pred_boxes):
                _name = filename[0] if isinstance(filename, (list, tuple)) else f"{i}"
                det_mismatch.append((os.path.basename(str(_name)),
                                     len(gt_boxes), len(pred_boxes)))

            if args.det_single_box:
                gt_sb = mask_to_single_box(mask_np[0], DET_CLS, args.det_min_area,
                                           closing_kernel=args.det_closing)
                pred_sb = mask_to_single_box(pred_np[0], DET_CLS, args.det_min_area,
                                             closing_kernel=args.det_closing)
                if gt_sb is not None and pred_sb is not None:
                    sb_iou_sum += float(box_iou_matrix([gt_sb], [pred_sb])[0, 0])
                    sb_iou_count += 1
                _tp, _fp, _fn = single_box_match(gt_sb, pred_sb, 0.5)
                sb50[0] += _tp; sb50[1] += _fp; sb50[2] += _fn
                _tp, _fp, _fn = single_box_match(gt_sb, pred_sb, 0.75)
                sb75[0] += _tp; sb75[1] += _fp; sb75[2] += _fn

            if args.save_path:
                from PIL import Image
                name = filename[0] if isinstance(filename, (list, tuple)) else f"{i}"
                name = os.path.splitext(os.path.basename(name))[0]
                pred_img = Image.fromarray(pred_np[0].astype(np.uint8))
                pred_img.save(os.path.join(args.save_path, name + ".png"))

            intersection, union, target = intersectionAndUnion(
                pred_np, mask_np, n_classes, ignore_index=255
            )

            intersection_meter += intersection
            union_meter += union
            target_meter += target
            # prediction = TP + FP = union - target + intersection
            prediction_meter += union - target + intersection

    iou_class = intersection_meter / (union_meter + 1e-10)
    mIoU = np.mean(iou_class)

    precision_class = intersection_meter / (prediction_meter + 1e-10)
    recall_class = intersection_meter / (target_meter + 1e-10)
    f1_class = 2 * intersection_meter / (prediction_meter + target_meter + 1e-10)

    mPrecision = np.mean(precision_class)
    mRecall = np.mean(recall_class)
    mF1 = np.mean(f1_class)

    total_pixels = np.sum(target_meter)           # test.py 里是 numpy array，直接求和
    po = np.sum(intersection_meter) / (total_pixels + 1e-10)
    pe = np.sum(prediction_meter * target_meter) / (total_pixels ** 2 + 1e-10)
    kappa = (po - pe) / (1 - pe + 1e-10)

    print("------------------------------------------------")
    header = f"{'Class':<7}{'IoU':>9}{'Precision':>11}{'Recall':>9}{'F1':>9}"
    print(header)
    for i in range(n_classes):
        print(f"{i:<7}{iou_class[i] * 100:>9.2f}"
              f"{precision_class[i] * 100:>11.2f}"
              f"{recall_class[i] * 100:>9.2f}"
              f"{f1_class[i] * 100:>9.2f}")
    print("------------------------------------------------")
    print(f"{'mIoU':<7}{mIoU * 100:>9.2f}")
    print(f"{'mPrecision':<11}{mPrecision * 100:>9.2f}")
    print(f"{'mRecall':<8}{mRecall * 100:>9.2f}")
    print(f"{'mF1':<7}{mF1 * 100:>9.2f}")
    print(f"{'Kappa':<7}{kappa * 100:>9.2f}")
    print("------------------------------------------------")

    yt = np.array(img_y_true)
    yp = np.array(img_y_pred)
    sc = np.array(img_score)

    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    acc = (tp + tn) / len(yt) if len(yt) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    # Specificity: TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Balanced accuracy: (recall + specificity) / 2
    bal_acc = (rec + spec) / 2

    # AUC (binary, using img_score as the decision score)
    if yt.sum() > 0 and (1 - yt).sum() > 0:
        order = np.argsort(-sc)
        yt_sorted = yt[order]
        P = yt.sum()
        N = len(yt) - P
        tp_cumsum = np.cumsum(yt_sorted)
        fp_cumsum = np.cumsum(1 - yt_sorted)
        tpr = tp_cumsum / P
        fpr = fp_cumsum / N
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        auc = float(np.trapz(tpr, fpr))
    else:
        auc = float('nan')

    print()
    print("========================================================")
    print("  Image-level Landslide Detection (binary classification)")
    print("========================================================")
    print(f"  Total images   : {len(yt)}")
    print(f"  Positive (GT)  : {yt.sum()}")
    print(f"  Negative (GT)  : {(1 - yt).sum()}")
    print("--------------------------------------------------------")
    print(f"  {'':<14}{'Pred-Pos':>10}{'Pred-Neg':>10}")
    print(f"  {'GT-Pos':<14}{tp:>10}{fn:>10}")
    print(f"  {'GT-Neg':<14}{fp:>10}{tn:>10}")
    print("--------------------------------------------------------")
    print(f"  Accuracy       : {acc * 100:>6.2f}")
    print(f"  Balanced Acc   : {bal_acc * 100:>6.2f}")
    print(f"  Precision      : {prec * 100:>6.2f}")
    print(f"  Recall (Sens)  : {rec * 100:>6.2f}")
    print(f"  Specificity    : {spec * 100:>6.2f}")
    print(f"  F1-score       : {f1 * 100:>6.2f}")
    print(f"  AUC            : {auc:>6.4f}")
    print("========================================================")

    mean_box_iou = det_iou_sum / det_iou_count if det_iou_count > 0 else 0.0
    p50, r50, f50 = prf(*det50)
    p75, r75, f75 = prf(*det75)

    print()
    print("========================================================")
    print("  Box-level Landslide Detection (mask-derived, Plan A)")
    print("========================================================")
    print(f"  Target class   : {DET_CLS}")
    print(f"  Min area       : {args.det_min_area} px")
    print(f"  GT boxes       : {det50[0] + det50[2]}")
    print("--------------------------------------------------------")
    print(f"  Mean box IoU   : {mean_box_iou * 100:>6.2f}   (best-match, threshold-free)")
    print("--------------------------------------------------------")
    print(f"  {'IoU thr':<10}{'TP':>7}{'FP':>7}{'FN':>7}"
          f"{'Precision':>11}{'Recall':>9}{'F1':>9}")
    print(f"  {'0.50':<10}{det50[0]:>7}{det50[1]:>7}{det50[2]:>7}"
          f"{p50 * 100:>11.2f}{r50 * 100:>9.2f}{f50 * 100:>9.2f}")
    print(f"  {'0.75':<10}{det75[0]:>7}{det75[1]:>7}{det75[2]:>7}"
          f"{p75 * 100:>11.2f}{r75 * 100:>9.2f}{f75 * 100:>9.2f}")
    print("========================================================")

    n_mismatch = len(det_mismatch)
    n_more = sum(1 for _, g, p in det_mismatch if p > g)
    n_fewer = n_mismatch - n_more
    print()
    print("--------------------------------------------------------")
    print("  Box-count Diagnostics")
    print("--------------------------------------------------------")
    print(f"  Closing kernel       : {args.det_closing} px"
          f"{'  (disabled)' if args.det_closing == 0 else ''}")
    print(f"  Images evaluated     : {det_img_total}")
    print(f"  GT boxes total       : {det_gt_total}"
          f"   (avg {det_gt_total / max(det_img_total, 1):.2f}/image)")
    print(f"  Pred boxes total     : {det_pred_total}"
          f"   (avg {det_pred_total / max(det_img_total, 1):.2f}/image)")
    print(f"  Count mismatch       : {n_mismatch} images"
          f"  ({n_mismatch / max(det_img_total, 1) * 100:.1f}%)")
    print(f"     pred > GT (fragmented): {n_more}")
    print(f"     pred < GT (merged/lost): {n_fewer}")
    if det_mismatch:
        print("  First mismatched images (name, GT boxes, pred boxes):")
        for name, g, p in det_mismatch[:20]:
            print(f"     {name:<20} GT={g:<4} pred={p}")
        if n_mismatch > 20:
            print(f"     ... and {n_mismatch - 20} more")
    print("--------------------------------------------------------")

    if args.det_single_box:
        mean_sb_iou = sb_iou_sum / sb_iou_count if sb_iou_count > 0 else 0.0
        sp50, sr50, sf50 = prf(*sb50)
        sp75, sr75, sf75 = prf(*sb75)
        print()
        print("========================================================")
        print("  Single-box Landslide Detection (one bbox per image)")
        print("========================================================")
        print(f"  Target class   : {DET_CLS}")
        print(f"  Min area       : {args.det_min_area} px")
        print(f"  Images w/ both boxes : {sb_iou_count}")
        print("--------------------------------------------------------")
        print(f"  Mean box IoU   : {mean_sb_iou * 100:>6.2f}   (per-image, both boxes present)")
        print("--------------------------------------------------------")
        print(f"  {'IoU thr':<10}{'TP':>7}{'FP':>7}{'FN':>7}"
              f"{'Precision':>11}{'Recall':>9}{'F1':>9}")
        print(f"  {'0.50':<10}{sb50[0]:>7}{sb50[1]:>7}{sb50[2]:>7}"
              f"{sp50 * 100:>11.2f}{sr50 * 100:>9.2f}{sf50 * 100:>9.2f}")
        print(f"  {'0.75':<10}{sb75[0]:>7}{sb75[1]:>7}{sb75[2]:>7}"
              f"{sp75 * 100:>11.2f}{sr75 * 100:>9.2f}{sf75 * 100:>9.2f}")
        print("========================================================")


if __name__ == '__main__':
    args = get_parser().parse_args()
    test(args)