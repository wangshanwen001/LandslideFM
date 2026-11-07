import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


STRUCTURED_FIELDS = [
    "landslide_exists", "has_building", "has_road",
    "has_water", "has_vegetation",
]


def _read_annotation_csv(csv_path):
    """Return list of dicts: [{image_name, field: bool, ..., text_description: str}]"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Clean up header: remove leading/trailing spaces / BOM
        reader.fieldnames = [h.strip().lstrip("\ufeff") for h in reader.fieldnames]
        for r in reader:
            name = (r.get("image_name") or r.get("image") or "").strip()
            if not name or name.startswith("#"):
                continue
            row = {"image_name": name}
            for f in STRUCTURED_FIELDS:
                v = (r.get(f) or "").strip()
                if v.startswith("#"):
                    row[f] = None   # skip field if annotated as comment
                else:
                    row[f] = v.lower() in ("1", "true", "yes", "y", "t")
            # Free-text description (optional, used for --eval-mode description/both)
            row["text_description"] = (r.get("text_description") or "").strip()
            rows.append(row)
    return rows


def _binary_prf1(y_true, y_pred):
    tp = sum(int(t) and int(p) for t, p in zip(y_true, y_pred))
    fp = sum(not int(t) and int(p) for t, p in zip(y_true, y_pred))
    fn = sum(int(t) and not int(p) for t, p in zip(y_true, y_pred))
    tn = sum(not int(t) and not int(p) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp + fn, tn + fp, tp, fp, fn, tn


import re
from collections import Counter as _Counter


def _tokenize(text):
    """Simple whitespace + punctuation tokenizer (lowercased)."""
    text = text.lower().strip()
    # split on whitespace and non-alphanumeric, keep tokens
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return tokens


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_n(reference, hypothesis, n):
    """BLEU-n precision for a single (ref, hyp) pair."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if len(hyp_tokens) < n or len(ref_tokens) == 0:
        return 0.0
    ref_ngrams = _Counter(_ngrams(ref_tokens, n))
    hyp_ngrams = _Counter(_ngrams(hyp_tokens, n))
    overlap = sum((ref_ngrams & hyp_ngrams).values())
    total = sum(hyp_ngrams.values())
    return overlap / total if total > 0 else 0.0


def _bleu_score(reference, hypothesis, max_n=4):
    """Corpus-level BLEU-1..4 with brevity penalty (averaged over pairs)."""
    precisions = []
    for n in range(1, max_n + 1):
        p = _bleu_n(reference, hypothesis, n)
        precisions.append(p)
    if any(p == 0 for p in precisions):
        return 0.0, precisions
    # Brevity penalty
    ref_len = len(_tokenize(reference))
    hyp_len = len(_tokenize(hypothesis))
    if hyp_len == 0:
        return 0.0, precisions
    bp = 1.0 if hyp_len > ref_len else np.exp(1 - ref_len / hyp_len)
    geo_mean = float(np.exp(np.mean(np.log(precisions))))
    return bp * geo_mean, precisions


def _lcs_length(x, y):
    """Longest common subsequence length (DP)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l(reference, hypothesis):
    """ROUGE-L: F1 based on longest common subsequence of tokens."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _word_overlap_prf1(reference, hypothesis):
    """Word-level (unigram) overlap precision / recall / F1."""
    ref_set = set(_tokenize(reference))
    hyp_set = set(_tokenize(hypothesis))
    if not ref_set or not hyp_set:
        return 0.0, 0.0, 0.0
    inter = ref_set & hyp_set
    p = len(inter) / len(hyp_set)
    r = len(inter) / len(ref_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_text_metrics(reference, hypothesis):
    """Return a dict of text generation metrics for one (ref, hyp) pair."""
    bleu, bleu_ngram = _bleu_score(reference, hypothesis, max_n=4)
    rl_p, rl_r, rl_f1 = _rouge_l(reference, hypothesis)
    wo_p, wo_r, wo_f1 = _word_overlap_prf1(reference, hypothesis)
    return {
        "bleu": bleu,
        "bleu_1": bleu_ngram[0],
        "bleu_2": bleu_ngram[1],
        "bleu_3": bleu_ngram[2],
        "bleu_4": bleu_ngram[3],
        "rouge_l_p": rl_p,
        "rouge_l_r": rl_r,
        "rouge_l_f1": rl_f1,
        "word_overlap_p": wo_p,
        "word_overlap_r": wo_r,
        "word_overlap_f1": wo_f1,
    }


def _resolve_image_path(image_dir, image_name, subdirs=None):
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

    # 5. Fallback: return the first plausible path (will fail later with clear error)
    return os.path.join(image_dir, subdirs[0], basename) if subdirs else root_candidate


def main():
    parser = argparse.ArgumentParser(
        description="L2 Structured Evaluation of VLM Understanding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--auto-checkpoint", action="store_true")
    parser.add_argument("--csv", type=str, required=True,
                        help="Annotation CSV with header: image_name,"
                             "landslide_exists,has_building,has_road,"
                             "has_water,has_vegetation")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory where annotation image_names are resolved")
    parser.add_argument("--output-dir", type=str, default="vlm_eval_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit evaluation to first N images (for quick smoke test)")
    parser.add_argument("--ablate", type=str, default="full",
                        choices=["full", "no_seg", "random_mask"],
                        help="Ablation variant. 'full'=Ours (seg overlay), "
                             "'no_seg'=no overlay given, "
                             "'random_mask'=overlay is a random binary mask.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter directory for VLM (optional). "
                             "When set, the base VLM is loaded then the adapter "
                             "is applied via peft. Useful for comparing baseline vs "
                             "LoRA-finetuned VLM on L2 structured metrics.")
    parser.add_argument("--eval-mode", type=str, default="structured",
                        choices=["structured", "description", "both"],
                        help="What to evaluate: "
                             "'structured' = 5-field JSON P/R/F1 (default); "
                             "'description' = free-text generation quality (BLEU/ROUGE); "
                             "'both' = run both and save two reports.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bad_dir = os.path.join(args.output_dir, "bad_cases")
    os.makedirs(bad_dir, exist_ok=True)

    np.random.seed(args.seed)

    from model.semseg.landslide_understanding import LandslideQAInterface

    interface = LandslideQAInterface(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        verbose=True,
        language=args.language,
        auto_checkpoint=args.auto_checkpoint,
        lora_path=args.lora_path,
    )
    if args.lora_path:
        print(f"LoRA adapter will be loaded from: {args.lora_path}")
    else:
        print("LoRA adapter: none (--lora-path not set); using the base VLM model.")
    print("Loading segmentation model...")
    interface.load_segmentation_model()
    print("Loading VLM model...")
    interface.load_understanding_model()

    annotations = _read_annotation_csv(args.csv)
    if args.max_images is not None:
        annotations = annotations[:args.max_images]
    print(f"Loaded {len(annotations)} annotated images from {args.csv}")

    do_structured = args.eval_mode in ("structured", "both")
    do_description = args.eval_mode in ("description", "both")

    all_preds = []
    per_field_y_true = defaultdict(list)
    per_field_y_pred = defaultdict(list)
    bad_rows = []
    parse_errors = 0

    # Description evaluation state
    desc_predictions = []        # list of {image_name, generated, reference}
    desc_metric_keys = [
        "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "rouge_l_p", "rouge_l_r", "rouge_l_f1",
        "word_overlap_p", "word_overlap_r", "word_overlap_f1",
    ]
    desc_metric_sums = {k: 0.0 for k in desc_metric_keys}
    desc_count = 0
    desc_skipped_no_ref = 0

    for idx, ann in enumerate(annotations):
        name = ann["image_name"]
        img_path = _resolve_image_path(args.image_dir, name)
        print(f"\n[{idx+1}/{len(annotations)}] {name}")
        if not os.path.exists(img_path):
            print(f"  [SKIP] missing image: {img_path}")
            continue

        try:
            seg_mask, original_image = interface.run_segmentation(img_path)
        except (OSError, IOError) as e:
            print(f"  [SKIP] corrupted image: {img_path} ({e})")
            continue

        if args.ablate == "no_seg":
            # Feed a blank mask to generate_overlay so no overlay is visible
            effective_mask = np.zeros_like(seg_mask)
        elif args.ablate == "random_mask":
            effective_mask = (np.random.rand(*seg_mask.shape) > 0.9).astype(np.uint8)
        else:  # full
            effective_mask = seg_mask

        if do_structured:
            pred = interface.understanding_model.analyze_structured_env(
                original_image, effective_mask
            )
            if "_parse_error" in pred:
                parse_errors += 1

            rec = {
                "image_name": name,
                "prediction": {k: bool(pred.get(k, False)) for k in STRUCTURED_FIELDS},
                "annotation": {k: bool(ann.get(k, False)) for k in STRUCTURED_FIELDS},
            }
            if "_parse_error" in pred:
                rec["_parse_error"] = pred["_parse_error"]
                rec["_raw"] = pred.get("_raw", "")
            all_preds.append(rec)

            wrong_fields = []
            for f in STRUCTURED_FIELDS:
                gt = bool(ann.get(f, False))
                pr = bool(pred.get(f, False))
                per_field_y_true[f].append(gt)
                per_field_y_pred[f].append(pr)
                if gt != pr:
                    wrong_fields.append(f)

            if wrong_fields:
                bad_rows.append((name, wrong_fields))
                overlay = interface.understanding_model._generate_overlay(
                    original_image, seg_mask
                )
                safe_name = os.path.basename(name).replace("/", "_")
                miss_tag = "__".join(wrong_fields)
                out_file = f"{miss_tag}__{safe_name}.png" if safe_name.lower().endswith((".png",".jpg",".jpeg")) \
                           else f"{miss_tag}__{safe_name}.png"
                overlay.save(os.path.join(bad_dir, out_file))

        if do_description:
            reference_desc = ann.get("text_description", "")
            if not reference_desc:
                desc_skipped_no_ref += 1
                print(f"  [SKIP] no text_description annotation for {name}")
            else:
                generated = interface.understanding_model.analyze_environmental_context(
                    original_image, effective_mask
                )
                metrics = compute_text_metrics(reference_desc, generated)
                for k in desc_metric_keys:
                    desc_metric_sums[k] += metrics[k]
                desc_count += 1
                desc_predictions.append({
                    "image_name": name,
                    "generated": generated,
                    "reference": reference_desc,
                    "metrics": metrics,
                })
                print(f"  Description BLEU={metrics['bleu']:.3f} "
                      f"ROUGE-L={metrics['rouge_l_f1']:.3f}")

    if do_structured:
        rows_report = []
        all_p, all_r, all_f1 = [], [], []
        count_totals = {"n_pos": 0, "n_neg": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

        for f in STRUCTURED_FIELDS:
            yt = per_field_y_true[f]
            yp = per_field_y_pred[f]
            prec, rec, f1, n_pos, n_neg, tp, fp, fn, tn = _binary_prf1(yt, yp)
            rows_report.append((f, prec, rec, f1, n_pos, n_neg, tp, fp, fn, tn))
            all_p.append(prec); all_r.append(rec); all_f1.append(f1)
            for key, val in (("n_pos", n_pos), ("n_neg", n_neg), ("tp", tp),
                             ("fp", fp), ("fn", fn), ("tn", tn)):
                count_totals[key] += val

        macro_p = float(np.mean(all_p))
        macro_r = float(np.mean(all_r))
        macro_f1 = float(np.mean(all_f1))

        out_json = os.path.join(args.output_dir, "vlm_predictions.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "ablation": args.ablate,
                "lora_path": args.lora_path,
                "n_images": len(all_preds),
                "parse_errors": parse_errors,
                "predictions": all_preds,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nSaved predictions → {out_json}")

        header = ["Field", "Precision", "Recall", "F1",
                  "Support(Pos)", "Support(Neg)", "TP", "FP", "FN", "TN"]
        fmt = "{:<16} {:>10} {:>8} {:>8} {:>13} {:>13} {:>7} {:>7} {:>7} {:>7}"

        lines_txt = []
        lines_txt.append(f"LandslideFM VLM L2 Evaluation Report (Structured)")
        lines_txt.append(f"Ablation variant : {args.ablate}")
        lines_txt.append(f"LoRA adapter     : {args.lora_path if args.lora_path else 'none (baseline)'}")
        lines_txt.append(f"CSV file         : {args.csv}")
        lines_txt.append(f"Images evaluated : {len(all_preds)}")
        lines_txt.append(f"Parse errors     : {parse_errors}")
        lines_txt.append("")
        lines_txt.append(fmt.format(*header))
        lines_txt.append("-" * len(fmt.format(*([""] * len(header)))))

        rows_csv = [header]
        for row in rows_report:
            name, p, rc, f1, n_pos, n_neg, tp, fp, fn, tn = row
            lines_txt.append(fmt.format(name, f"{p:.4f}", f"{rc:.4f}", f"{f1:.4f}",
                                        f"{n_pos}", f"{n_neg}",
                                        f"{tp}", f"{fp}", f"{fn}", f"{tn}"))
            rows_csv.append([name, f"{p:.6f}", f"{rc:.6f}", f"{f1:.6f}",
                             f"{n_pos}", f"{n_neg}",
                             f"{tp}", f"{fp}", f"{fn}", f"{tn}"])

        lines_txt.append("")
        lines_txt.append(fmt.format("MACRO-AVG",
                                    f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}",
                                    count_totals["n_pos"], count_totals["n_neg"],
                                    count_totals["tp"], count_totals["fp"],
                                    count_totals["fn"], count_totals["tn"]))
        rows_csv.append(["MACRO-AVG", f"{macro_p:.6f}", f"{macro_r:.6f}", f"{macro_f1:.6f}",
                         count_totals["n_pos"], count_totals["n_neg"],
                         count_totals["tp"], count_totals["fp"],
                         count_totals["fn"], count_totals["tn"]])

        lines_txt.append("")
        lines_txt.append("Metric definitions (positive class = 1, negative class = 0):")
        lines_txt.append("  Precision     TP / (TP + FP)   of the predicted positives, how many are correct (PPV)")
        lines_txt.append("  Recall        TP / (TP + FN)   of the true positives, how many were recovered (sensitivity)")
        lines_txt.append("  F1            2PR / (P + R)    harmonic mean of precision and recall")
        lines_txt.append("  Support(Pos)  TP + FN          ground-truth positives (support of class 1)")
        lines_txt.append("  Support(Neg)  TN + FP          ground-truth negatives (support of class 0)")
        lines_txt.append("  TP FP FN TN                    confusion-matrix counts; TN = ground-truth")
        lines_txt.append("                                 negatives correctly predicted as 0")
        lines_txt.append("  Support(Pos) + Support(Neg) is the number of images scored for that field.")
        lines_txt.append("  MACRO-AVG averages the per-field P/R/F1 values and sums the counts,")
        lines_txt.append("  following the same convention as sklearn's classification_report.")
        lines_txt.append("  A metric with an empty denominator is reported as 0.0000 - check the")
        lines_txt.append("  matching support column to tell whether that value is meaningful.")

        report_txt = "\n".join(lines_txt)
        print("\n" + "=" * 80)
        print(report_txt)
        print("=" * 80 + "\n")

        report_txt_path = os.path.join(args.output_dir, "metrics_report.txt")
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report_txt + "\n")
        print(f"Saved text report → {report_txt_path}")

        report_csv_path = os.path.join(args.output_dir, "metrics_report.csv")
        with open(report_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_csv)
        print(f"Saved csv report  → {report_csv_path}")

        bad_csv_path = os.path.join(args.output_dir, "bad_cases_summary.csv")
        with open(bad_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "wrong_fields_count", "wrong_fields"])
            for name, wf in bad_rows:
                writer.writerow([name, len(wf), ",".join(wf)])
        print(f"Saved bad cases  → {bad_csv_path}  ({len(bad_rows)} images)")

    if do_description:
        desc_json = os.path.join(args.output_dir, "description_predictions.json")
        with open(desc_json, "w", encoding="utf-8") as f:
            json.dump({
                "ablation": args.ablate,
                "lora_path": args.lora_path,
                "n_images": desc_count,
                "skipped_no_reference": desc_skipped_no_ref,
                "predictions": desc_predictions,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nSaved description predictions → {desc_json}")

        if desc_count > 0:
            desc_avg = {k: desc_metric_sums[k] / desc_count for k in desc_metric_keys}
        else:
            desc_avg = {k: 0.0 for k in desc_metric_keys}

        desc_header = ["Metric", "Value"]
        desc_fmt = "{:<22} {:>10}"

        desc_lines = []
        desc_lines.append("LandslideFM VLM Text Description Generation Report")
        desc_lines.append(f"Ablation variant : {args.ablate}")
        desc_lines.append(f"LoRA adapter     : {args.lora_path if args.lora_path else 'none (baseline)'}")
        desc_lines.append(f"CSV file         : {args.csv}")
        desc_lines.append(f"Images evaluated : {desc_count}")
        desc_lines.append(f"Skipped (no ref) : {desc_skipped_no_ref}")
        desc_lines.append("")
        desc_lines.append(desc_fmt.format(*desc_header))
        desc_lines.append("-" * len(desc_fmt.format("", "")))

        desc_rows_csv = [desc_header]
        display_order = [
            "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
            "rouge_l_p", "rouge_l_r", "rouge_l_f1",
            "word_overlap_p", "word_overlap_r", "word_overlap_f1",
        ]
        for k in display_order:
            v = desc_avg[k]
            desc_lines.append(desc_fmt.format(k, f"{v:.4f}"))
            desc_rows_csv.append([k, f"{v:.6f}"])

        desc_report = "\n".join(desc_lines)
        print("\n" + "=" * 60)
        print(desc_report)
        print("=" * 60 + "\n")

        desc_txt_path = os.path.join(args.output_dir, "description_report.txt")
        with open(desc_txt_path, "w", encoding="utf-8") as f:
            f.write(desc_report + "\n")
        print(f"Saved description text report → {desc_txt_path}")

        desc_csv_path = os.path.join(args.output_dir, "description_report.csv")
        with open(desc_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(desc_rows_csv)
        print(f"Saved description csv report  → {desc_csv_path}")


if __name__ == "__main__":
    main()
