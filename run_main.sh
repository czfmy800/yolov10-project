model_id=yolov10x
model_path=yolov10x.pt
datasets="tt100k_2021"
epoch=100
mkdir -p out_log

export CUDA_LAUNCH_BLOCKING=1

# 使用 nohup 运行训练脚本，并将输出重定向到 out_log/train.log
nohup python main.py \
    --model_id "$model_id" \
    --model_path "Yolov10/$model_path" \
    --epoch $epoch \
    --project "./result_file/$model_id-$datasets-$epoch" \
    --batch 16 \
    --imgsz 640 \
    --device '0' \
    --worker 8 \
    --train \
    --eval \
    --datasets $datasets \
    --lr0 0.01 \
    --eval_datasets "tt100k_2021" \
    --train_result_file "./result/$model_id-$datasets/result-$model_id-$datasets" \
    > out_log/train-$model_id-$datasets-$epoch-debug.log 2>&1 &

