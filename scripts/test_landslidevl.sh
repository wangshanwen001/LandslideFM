dataset='landslide'
method='sagt'
exp='dinov2_base'
split='50_100_nocutmix'

checkpoint="exp/$dataset/$method/$exp/$split/best.pth"

if [ ! -f "$checkpoint" ]; then
    echo "ERROR: segmentation checkpoint not found: $(pwd)/$checkpoint" >&2
    echo "       Run scripts/train_landslide_1.sh first, or point --checkpoint" >&2
    echo "       at an existing best.pth / use --auto-checkpoint." >&2
    exit 1
fi

python evaluate_vlm.py \
    --config configs/landslide_Qwen3_8B.yaml \
    --checkpoint "$checkpoint" \
    --csv annotations/LandslideVL.csv \
    --image-dir Landslide/JPEGImages \
    --output-dir vlm_eval_output/full_lora_Qwen3_8B \
    --ablate full \
    --lora-path lora_output/landslide_lora_multitask_Qwen3_8B_final/final_adapter \
    --eval-mode both