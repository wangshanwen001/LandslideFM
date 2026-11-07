import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Landslide Understanding Pipeline - Segmentation + VLM Analysis + QA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to segmentation checkpoint")
    parser.add_argument("--auto-checkpoint", action="store_true", default=False,
                        help="Automatically find the best trained checkpoint in exp/ directory")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="./vlm_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mode", type=str, default="analyze", choices=["analyze", "qa"],
                        help="Mode: 'analyze' for full analysis, 'qa' for question answering")
    parser.add_argument("--question", type=str, default=None, help="Question for QA mode")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"],
                        help="Output language: 'en' for English (default), 'zh' for Chinese")
    parser.add_argument("--structured", action="store_true", default=False,
                        help="Additionally run simplified structured (JSON) env analysis "
                             "(5 boolean fields). Does not affect the full analyze output.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter directory for VLM (optional). "
                             "When set, the base Qwen-VL is loaded then the adapter "
                             "is applied via peft.")

    args = parser.parse_args()

    if args.mode == "qa" and args.question is None:
        parser.error("--mode qa requires --question to be specified")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Landslide Understanding Pipeline")
    print("=" * 70)
    print(f"Config:          {args.config}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Image:           {args.image}")
    print(f"Mode:            {args.mode}")
    print(f"Device:          {args.device}")
    print(f"Language:        {args.language}")
    print(f"Output dir:      {args.output_dir}")
    print(f"LoRA adapter:    {args.lora_path if args.lora_path else 'none'}")
    print("=" * 70 + "\n")

    from model.semseg.landslide_understanding import LandslideQAInterface

    if args.auto_checkpoint and args.checkpoint:
        print("Warning: --auto-checkpoint is ignored because --checkpoint is specified")

    interface = LandslideQAInterface(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        language=args.language,
        auto_checkpoint=args.auto_checkpoint,
        lora_path=args.lora_path,
    )

    print("Loading segmentation model...")
    interface.load_segmentation_model()
    print(f"✓ Segmentation model loaded from: {interface.checkpoint_path or 'default'}")

    print("Loading VLM understanding model...")
    interface.load_understanding_model()
    print("✓ VLM model loaded\n")

    if args.mode == "analyze":
        run_log = "Running full analysis...\n" if args.language == "en" else "正在运行完整分析...\n"
        print(run_log)
        result = interface.analyze_image(args.image)

        env_title = "Environmental Context Analysis" if args.language == "en" else "环境背景分析"
        impact_title = "Impact Assessment" if args.language == "en" else "影响评估"

        print("\n" + "=" * 70)
        print(env_title)
        print("=" * 70)
        print(result["environmental_context"])

        print("\n" + "=" * 70)
        print(impact_title)
        print("=" * 70)
        print(result["impact_assessment"])

        if args.structured:
            structured_title = "Structured (Simplified) Env Analysis" if args.language == "en" else "结构化环境分析（简化版）"
            structured_result = interface.understanding_model.analyze_structured_env(
                result["original_image"], result["segmentation_mask"]
            )
            print("\n" + "=" * 70)
            print(structured_title)
            print("=" * 70)
            import json
            print(json.dumps(structured_result, indent=2, ensure_ascii=False))

        seg_mask = result["segmentation_mask"]
        original_image = result["original_image"]

        overlay = interface.understanding_model._generate_overlay(original_image, seg_mask)
        overlay_path = os.path.join(args.output_dir, "landslide_overlay.png")
        overlay.save(overlay_path)
        saved_log = "✓ Overlay saved to: {}" if args.language == "en" else "✓ 叠加图已保存至: {}"
        print(saved_log.format(overlay_path))

        mask_path = os.path.join(args.output_dir, "landslide_mask.png")
        Image.fromarray(seg_mask.astype(np.uint8)).save(mask_path)
        print(saved_log.format(mask_path))

        report_path = os.path.join(args.output_dir, "analysis_report.txt")
        report_title = "LANDSLIDE UNDERSTANDING REPORT" if args.language == "en" else "滑坡灾害理解报告"
        input_image_label = "Input Image: " if args.language == "en" else "输入图像: "
        analysis_time_label = "Analysis Time: " if args.language == "en" else "分析时间: "

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(report_title + "\n")
            f.write("=" * 70 + "\n\n")
            f.write(input_image_label + "{}\n".format(args.image))
            f.write(analysis_time_label + "{}\n\n".format(__import__("datetime").datetime.now()))
            f.write("=" * 70 + "\n")
            f.write(env_title.upper() if args.language == "en" else env_title + "\n")
            f.write("=" * 70 + "\n")
            f.write(result["environmental_context"] + "\n\n")
            f.write("=" * 70 + "\n")
            f.write(impact_title.upper() if args.language == "en" else impact_title + "\n")
            f.write("=" * 70 + "\n")
            f.write(result["impact_assessment"] + "\n")
            if args.structured:
                f.write("\n" + "=" * 70 + "\n")
                f.write(structured_title.upper() + "\n")
                f.write("=" * 70 + "\n")
                f.write(json.dumps(structured_result, indent=2, ensure_ascii=False) + "\n")
        print(saved_log.format(report_path))

    elif args.mode == "qa":
        question_label = "Question: " if args.language == "en" else "问题: "
        print(question_label + f"{args.question}\n")
        result = interface.ask(args.image, args.question)

        answer_label = "Answer" if args.language == "en" else "回答"
        print("\n" + "=" * 70)
        print(answer_label)
        print("=" * 70)
        print(result["answer"])

        overlay = interface.understanding_model._generate_overlay(
            result["original_image"], result["segmentation_mask"]
        )
        overlay_path = os.path.join(args.output_dir, "qa_overlay.png")
        overlay.save(overlay_path)
        saved_log = "✓ Overlay saved to: {}" if args.language == "en" else "✓ 叠加图已保存至: {}"
        print(saved_log.format(overlay_path))

        qa_path = os.path.join(args.output_dir, "qa_result.txt")
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write(question_label + "{}\n\n".format(args.question))
            f.write(answer_label + ": {}\n".format(result["answer"]))
        qa_result_log = "✓ QA result saved to: {}" if args.language == "en" else "✓ 问答结果已保存至: {}"
        print(qa_result_log.format(qa_path))

    complete_log = "✓ Analysis completed successfully!" if args.language == "en" else "✓ 分析完成！"
    print("\n" + "=" * 70)
    print(complete_log)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()