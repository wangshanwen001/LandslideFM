dataset='gvlm'
method='sagt'
exp='dinov2_base'
split='1_8'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 \
    --sam-version sam3_official \
    --sam-checkpoint pretrained/sam3.pt \
    --env-csv annotations/landslide_vlm_eval.csv \
    --env-loss-weight 0.1 \
    --consensus-weight 2>&1 | tee $save_path/out.log