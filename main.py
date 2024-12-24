# 此部分未完成，目前只用于训练

from ultralytics import YOLOv10
import cv2
import tempfile
import os
import argparse
import json
import time

def get_model(model_id,model_path):
     model_yaml_path = f"ultralytics/cfg/models/v10/{model_id}.yaml"
     if (os.path.exists(model_yaml_path) is False):
         raise ValueError("Please check the model.yaml file")
    #  if(os.path.exists(model_path) is False):
    #      raise ValueError("Please check the model.pt file")
     model  = YOLOv10(model_yaml_path).load(model_path)
     return model

def print_training_metrics(train_result, log_path="out_log/train_result.json"):
    """
    从训练结果中提取并打印关键指标，并保存到日志文件中。
    """
    metrics_dict = {}
    try:
        metrics = train_result.metrics

        # 提取常用指标
        metrics_dict['mAP'] = metrics.get('map', None)
        metrics_dict['Precision'] = metrics.get('precision', None)
        metrics_dict['Recall'] = metrics.get('recall', None)
        metrics_dict['Loss'] = metrics.get('loss', None)

        # 打印指标
        print("### Training Metrics ###")
        for key, value in metrics_dict.items():
            if value is not None:
                print(f"{key}: {value}")

        # 保存到 JSON 文件
        with open(log_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Training metrics saved to {log_path}")
    except AttributeError:
        print("无法提取训练指标。请检查 train_result 的结构。")

def print_evaluation_metrics(eval_result, log_path="out_log/eval_metrics.json"):
    """
    从评估结果中提取并打印关键指标，并保存到日志文件中。
    """
    metrics_dict = {}
    try:
        metrics = eval_result.metrics

        # 提取常用指标
        metrics_dict['mAP'] = metrics.get('map', None)
        metrics_dict['Precision'] = metrics.get('precision', None)
        metrics_dict['Recall'] = metrics.get('recall', None)
        metrics_dict['AP50'] = metrics.get('map50', None)
        metrics_dict['AP75'] = metrics.get('map75', None)
        metrics_dict['AR'] = metrics.get('ar', None)

        # 打印指标
        print("### Evaluation Metrics ###")
        for key, value in metrics_dict.items():
            if value is not None:
                print(f"{key}: {value}")

        # 保存到 JSON 文件
        with open(log_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Evaluation metrics saved to {log_path}")
    except AttributeError as e:
        print(f"无法提取评估指标。错误: {e}")
        print("请检查 eval_result 的结构。")



def train_model(model_id,model_path,dataset,result_file = "out_log/train_result.json",imgsz = 640,epoch = 100,batch = 4,resume = False,worker = 8,optimizer = 'SGD',project = "./result",device = '0',cache = False,lr0 = 0.01):
    model = get_model(model_id,model_path)
    data_yaml = f"ultralytics/cfg/datasets/{dataset}.yaml"
    if (model is None):
        raise ValueError("Please load model first")
    if (os.path.exists(data_yaml) is False):
        raise ValueError("Please check the datasets.yaml file")
    name = dataset + "_result"

    start_time = time.time()
    print(f"Start training at dataset {dataset}")
    train_result = model.train(data = data_yaml,epochs = epoch,batch = batch,workers = worker,optimizer = optimizer,
                project = project, imgsz = imgsz,
                name = name,
                resume = resume,
                device = device,
                cache = cache,
                lr0 = lr0
                )
    end_time = time.time()
    print(f"Training at dataset {dataset} time: {end_time - start_time} seconds")
    if train_result is not None:
        print_training_metrics(train_result,result_file)
    return train_result

def eval_model(model_id, model_path, dataset, result_file = "out_log/eval_result.json",  imgsz=640, batch=4, worker=8, device='0', cache=False, project="./result"):
    model = get_model(model_id, model_path)
    data_yaml = f"ultralytics/cfg/datasets/{dataset}.yaml"
    if model is None:
        raise ValueError("Please load model first")
    if not os.path.exists(data_yaml):
        raise ValueError("Please check the datasets.yaml file")
    name = dataset + "_eval"
    
    start_time = time.time()
    print(f"Start evaluation on dataset: {dataset}")

    # 捕获评估结果
    eval_result = model.val(
        data=data_yaml,
        batch=batch,
        workers=worker,
        device=device,
        imgsz=imgsz,
        cache=cache,
        project=project,
        name=name
    )
    end_time = time.time()
    print(f"Evaluation on dataset {dataset} completed in {end_time - start_time:.2f} seconds")

    # 输出评估结果
    if eval_result:
        print_evaluation_metrics(eval_result, result_file)
    else:
        print("Evaluation failed.")
    return eval_result



    

def model_interferce(model,image,video,model_id,image_size,conf_threshold):
    if model is None:
         model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    elif model is str:
        model = get_model(model_id,model)
    else:
        model = model

    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path

parser = argparse.ArgumentParser(description="Train YOLOv10 model")

    # 需要接收值的参数
parser.add_argument("--model_id", type=str, required=True, help="模型大小导入")
parser.add_argument("--model_path", type=str, required=True, help="模型.pt文件")
parser.add_argument("--datasets", type=str, required=True, help="数据集名称")
parser.add_argument("--device", type=str, default="0", help="训练设备, CPU或GPU")
parser.add_argument("--imgsz", type=int, default=640, help="输入图片大小")
parser.add_argument("--epoch", type=int, default=200, help="训练epoch")
parser.add_argument("--batch", type=int, default=4, help="训练batch")
parser.add_argument("--worker", type=int, default=8, help="训练worker")
parser.add_argument("--optimizer", type=str, default="SGD", help="训练的优化器")
parser.add_argument("--project", type=str, default="./result", help="训练结果保存路径")
parser.add_argument("--conf_threshold", type=float, default=0.5, help="推理置信度阈值")
parser.add_argument("--lr", type=str, default=None, help="学习率")
parser.add_argument("--train_result_file", type=str, default="out_log/train_result.json", help="训练结果保存路径")

    # 布尔标志参数
parser.add_argument("--resume", default = False,action="store_true", help="是否从上次训练中断处开始训练")
parser.add_argument("--cache", action="store_true", help="是否缓存训练结果")
parser.add_argument("--train", action="store_true", help="是否训练模型")
parser.add_argument("--eval", action="store_true", help="评估模型")
parser.add_argument("--load_and_eval", action="store_true", help="加载模型并评估模型")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.train:
        print("Staring training the model!")
        datasets = ""
        train_result = train_model(model_id=args.model_id,model_path=args.model_path,dataset=args.datasets,result_file=args.train_result_file,imgsz=args.imgsz,epoch=args.epoch,
                    batch=args.batch,resume=args.resume,worker=args.worker,optimizer=args.optimizer,
                    project=args.project,device=args.device,cache=args.cache,)
    if args.eval:
        print("Staring eval the model!")
