python fine_tuning_landslidevl.py \
    --config configs/landslide_Qwen3_8B.yaml \
    --checkpoint exp/landslide/sagt/dinov2_base/50_100_nocutmix/best.pth \
    --csv annotations/landslide_vlm_eval.csv \
    --image-dir Landslide/JPEGImages \
    --output-dir lora_output/landslide_lora_multitask_Qwen3_8B \
    --epochs 5 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --lora-r 8