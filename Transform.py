import json
import os
from PIL import Image
import numpy as np

json_file = 'annotations.json'
output_dir = '.'

class_names = [
    "i1", "i10", "i11", "i12", "i13", "i14", "i15", "i2", "i3", "i4", "i5",
    "il100", "il110", "il50", "il60", "il70", "il80", "il90", "io", "ip", "p1", "p10",
    "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p2", "p20", "p21",
    "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p3", "p4", "p5",
    "p6", "p7", "p8", "p9", "pa10", "pa12", "pa13", "pa14", "pa8", "pb", "pc",
    "pg", "ph1.5", "ph2", "ph2.1", "ph2.2", "ph2.4", "ph2.5", "ph2.6", "ph2.8", "ph2.9", "ph3",
    "ph3.2", "ph3.3", "ph3.5", "ph3.8", "ph4", "ph4.2", "ph4.3", "ph4.4", "ph4.5", "ph4.8", "ph5",
    "ph5.3", "ph5.5", "pl0", "pl10", "pl100", "pl110", "pl120", "pl15", "pl20", "pl25", "pl3",
    "pl30", "pl35", "pl4", "pl40", "pl5", "pl50", "pl60", "pl65", "pl70", "pl80", "pl90",
    "pm10", "pm13", "pm15", "pm1.5", "pm2", "pm2.5", "pm20", "pm25", "pm30", "pm35",
    "pm40", "pm46", "pm5", "pm50", "pm55", "pm8", "pn", "pne", "po", "pr10", "pr100",
    "pr20", "pr30", "pr40", "pr45", "pr50", "pr60", "pr70", "pr80", "ps", "pw2", "pw2.5",
    "pw3", "pw3.2", "pw3.5", "pw4", "pw4.2", "pw4.5", "w1", "w10", "w11", "w12", "w13",
    "w14", "w15", "w16", "w17", "w18", "w19", "w2", "w20", "w21", "w22", "w23", "w24",
    "w25", "w26", "w27", "w28", "w29", "w3", "w30", "w31", "w32", "w33", "w34", "w35",
    "w36", "w37", "w38", "w39", "w4", "w40", "w41", "w42", "w43", "w44", "w45", "w46",
    "w47", "w48", "w49", "w5", "w50", "w51", "w52", "w53", "w54", "w55", "w56", "w57",
    "w58", "w59", "w6", "w60", "w61", "w62", "w63", "w64", "w65", "w66", "w67", "w7",
    "w8", "w9", "pax", "pd", "pe", "phx", "plx", "pmx", "pnl", "prx", "pwx", "wo", "i6",
    "i7", "i8", "i9", "ilx", "w29", "w33", "w36", "w39", "w40", "w51", "w52", "w53", "w54",
    "w61", "w64", "w65", "w67", "w7", "w9", "pn40"
]

# 创建类别名称到ID的映射
object_class_id_map = {name: idx for idx, name in enumerate(class_names)}

datadir = "."

# 读取JSON数据
with open(json_file, 'r') as f:
    data = json.load(f)

# 遍历每个图像
for img_id, img_info in data['imgs'].items():

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构造输出文件的路径
    output_file = os.path.join(output_dir, f'{img_id}.txt')

    # 打开输出文件以写入
    with open(output_file, 'w') as out_f:
        # 遍历图像中的每个对象
        for obj in img_info['objects']:
            # 获取边界框信息
            bbox = obj['bbox']

            # 计算x, y, width, height (归一化为0到1之间)
            x_center = (bbox['xmin'] + bbox['xmax']) / 2 / 2048
            y_center = (bbox['ymin'] + bbox['ymax']) / 2 / 2048
            width = (bbox['xmax'] - bbox['xmin']) / 2048
            height = (bbox['ymax'] - bbox['ymin']) / 2048

            # 获取类别名称，并从映射中获取对应的ID
            category_name = obj['category']
            if category_name in object_class_id_map:
                object_class_id = object_class_id_map[category_name]
            else:
                print(f"Warning: Category '{category_name}' not found in mapping.")
                continue  # 如果没有找到对应的类别ID，则跳过该对象

            # 将结果写入文件
            out_f.write(f'{object_class_id} {x_center} {y_center} {width} {height}\n')

print("转换完成！")

