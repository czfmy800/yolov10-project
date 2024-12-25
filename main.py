from ultralytics import YOLOv10
import cv2
import tempfile
import os
import argparse
import json
import time
import csv
from texttable import Texttable
import pandas as pd
import matplotlib.pyplot as plt



def get_model(model_id,model_path):
     model_yaml_path = f"ultralytics/cfg/models/v10/{model_id}.yaml"
     if (os.path.exists(model_yaml_path) is False):
         raise ValueError("Please check the model.yaml file")
    #  if(os.path.exists(model_path) is False):
    #      raise ValueError("Please check the model.pt file")
     model  = YOLOv10(model_yaml_path).load(model_path)
     return model

def get_best_model(result_path):
    pt_file = args.project+"/"+result_path + "/weights/best.pt"
    if (pt_file.exists() is False):
        raise ValueError("Please check the model.pt file")
    model  = YOLOv10(pt_file)
    return model


def find_column(df, keyword, exact=False):
    """
    在 DataFrame 的列名中搜索包含指定关键词的列。

    参数:
    df (pd.DataFrame): 数据框。
    keyword (str): 需要搜索的关键词。
    exact (bool): 是否进行精确匹配。

    返回:
    str: 匹配的列名。

    异常:
    ValueError: 如果未找到匹配列或找到多个匹配列。
    """
    if exact:
        matched_columns = [col for col in df.columns if col.lower() == keyword.lower()]
    else:
        matched_columns = [col for col in df.columns if keyword.lower() in col.lower()]
    
    if len(matched_columns) == 0:
        raise ValueError(f"未找到包含 '{keyword}' 的列。")
    elif len(matched_columns) > 1:
        raise ValueError(f"找到多个包含 '{keyword}' 的列: {matched_columns}。请确保每个关键词只有一个匹配列。")
    else:
        return matched_columns[0]

def print_training_metrics(csv_file):
    """
    读取 YOLOv10 训练日志的 CSV 文件，动态检测所需列，
    提取每个周期的指标，使用 Texttable 打印表格，并将表格写入 result.txt 文件。

    参数:
    csv_file (str): CSV 文件的路径。
    """
    try:
        # 尝试使用 UTF-8 编码读取 CSV 文件
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试使用 GBK 编码（常用于中文环境）
        try:
            df = pd.read_csv(csv_file, encoding='gbk')
        except Exception as e:
            print(f"读取 CSV 文件时发生错误: {e}")
            return
    except FileNotFoundError:
        print(f"错误：文件 '{csv_file}' 未找到。")
        return
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return

    # 定义每个指标对应的搜索关键词
    metrics_keywords = {
        'epoch': 'epoch',
        'cls_om': 'train/cls_om',
        'precision': 'metrics/precision(B)',
        'recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)'
    }

    detected_columns = {}

    try:
        for metric, keyword in metrics_keywords.items():
            # 对于 'epoch' 列，可以选择精确匹配
            if metric == 'epoch':
                col_name = find_column(df, keyword, exact=False)  # 如果需要精确匹配，设置 exact=True
            else:
                col_name = find_column(df, keyword, exact=False)
            detected_columns[metric] = col_name
    except ValueError as ve:
        print(f"错误：{ve}")
        return

    # 提取数据
    try:
        epochs = df[detected_columns['epoch']].tolist()
        cls_om_list = df[detected_columns['cls_om']].tolist()
        precision_list = df[detected_columns['precision']].tolist()
        recall_list = df[detected_columns['recall']].tolist()
        mAP50_list = df[detected_columns['mAP50']].tolist()
        mAP50_95_list = df[detected_columns['mAP50-95']].tolist()
    except Exception as e:
        print(f"提取数据时发生错误: {e}")
        return

    # 创建 Texttable 对象
    table = Texttable()
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.set_cols_align(["c", "c", "c", "c", "c", "c"])
    table.set_cols_valign(["m"] * 6)
    table.set_max_width(0)

    # 设置表头
    headers = [
        detected_columns['epoch'],
        detected_columns['cls_om'],
        detected_columns['precision'],
        detected_columns['recall'],
        detected_columns['mAP50'],
        detected_columns['mAP50-95']
    ]
    table.header(headers)

    # 添加数据行
    for i in range(len(epochs)):
        row = [
            epochs[i],
            cls_om_list[i],
            precision_list[i],
            recall_list[i],
            mAP50_list[i],
            mAP50_95_list[i]
        ]
        table.add_row(row)

    # 绘制表格字符串
    table_str = table.draw()

    # 打印表格到控制台
    print(table_str)

    # 确定 CSV 文件所在目录
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    result_file_path = os.path.join(csv_dir, 'result.txt')

    try:
        # 将表格写入 result.txt 文件
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(table_str)
        print(f"\n表格已成功写入 '{result_file_path}'。")
    except Exception as e:
        print(f"写入 '{result_file_path}' 时发生错误: {e}")
    # Plot the metrics
    plot_metrics(epochs, cls_om_list, precision_list, recall_list, mAP50_list, mAP50_95_list)

def plot_metrics(epochs, cls_om, precision, recall, mAP50, mAP50_95):
    """
    Plots the training metrics over epochs.

    Parameters:
    epochs (list): List of epoch numbers.
    cls_om (list): List of cls_om values.
    precision (list): List of precision values.
    recall (list): List of recall values.
    mAP50 (list): List of mAP50 values.
    mAP50_95 (list): List of mAP50-95 values.
    """
    plt.figure(figsize=(12, 8))

    plt.plot(epochs, cls_om, marker='o', label='cls_om')
    plt.plot(epochs, precision, marker='s', label='Precision(B)')
    plt.plot(epochs, recall, marker='^', label='Recall(B)')
    plt.plot(epochs, mAP50, marker='d', label='mAP50(B)')
    plt.plot(epochs, mAP50_95, marker='x', label='mAP50-95(B)')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Values')
    plt.title('YOLOv10 Training Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()






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
                # device = device,
                cache = cache,
                lr0 = lr0,
                pretrained=False
                )
    end_time = time.time()
    print(f"Training at dataset {dataset} time: {end_time - start_time} seconds")

    return model,name

def eval_model(model_path,dataset,result_file = "out_log/train_result.json",imgsz = 640,epoch = 100,batch = 4,resume = False,worker = 8,optimizer = 'SGD',project = "./result",device = '0',cache = False,lr0 = 0.01):
    model = get_best_model(model_path)
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
        data = data_yaml,epochs = epoch,batch = batch,workers = worker,optimizer = optimizer,
                project = project, imgsz = imgsz,
                name = name,
                resume = resume,
                # device = device,
                cache = cache,
                lr0 = lr0,
                pretrained=False
   
    )
    end_time = time.time()
    print(f"Evaluation on dataset {dataset} completed in {end_time - start_time:.2f} seconds")

    # 输出评估结果

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
parser.add_argument("--lr0", type=float, default=0.01, help="学习率")
parser.add_argument("--train_result_file", type=str, default="out_log/train_result.json", help="训练结果保存路径")
parser.add_argument("--eval_datasets", type=str, default="tt100k_2021", help="评估数据集")
parser.add_argument("--interferce", type=str,default=None,  help="是否推理模型")

    # 布尔标志参数
parser.add_argument("--resume", default = False,action="store_true", help="是否从上次训练中断处开始训练")
parser.add_argument("--cache", action="store_true", help="是否缓存训练结果")
parser.add_argument("--train", action="store_true", help="是否训练模型")
parser.add_argument("--eval", action="store_true", help="评估模型")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.train:
        print("Staring training the model!")
        train_result,best_dir = train_model(model_id=args.model_id,model_path=args.model_path,dataset=args.datasets,result_file=args.train_result_file,imgsz=args.imgsz,epoch=args.epoch,
                    batch=args.batch,resume=args.resume,worker=args.worker,optimizer=args.optimizer,
                    project=args.project,device=args.device,cache=args.cache,lr0=args.lr0)
        csv_file_path = args.project + "/" + args.datasets + "_result/results.csv"
        print_training_metrics(csv_file_path)

    if args.eval:
        print("Staring evaling the model!")
        eval_result = eval_model(model_path=best_dir,dataset=args.eval_datasets,result_file=args.train_result_file,imgsz=args.imgsz,epoch=args.epoch,
                    batch=args.batch,resume=args.resume,worker=args.worker,optimizer=args.optimizer,
                    project=args.project,device=args.device,cache=args.cache,lr0=args.lr0)
    if args.interferce:
        print("Staring inferencing the model!")
