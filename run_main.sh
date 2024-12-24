model_id=yolov10s
model_path=yolov10s.pt
datasets="coco"
mkdir -p out_log

# 使用 nohup 运行训练脚本，并将输出重定向到 out_log/train.log
nohup python main.py \
    --model_id "$model_id" \
    --model_path "Yolov10/$model_path" \
    --epoch 1 \
    --project "./$model_id-$datasets" \
    --batch 16 \
    --imgsz 640 \
    --device '0' \
    --worker 4 \
    --train \
    --datasets $datasets \
    --train_result_file "./$model_id-$datasets/result-$model_id-$datasets" \
    > out_log/train-$model_id-$datasets-.log 2>&1 &
