import json
import os

json_file = 'annotations_all.json'
output_dir = '.'
filedir = "annotations_all.json"
annos = json.loads(open(filedir).read())
annos.keys()
class_names = annos['types']

# 创建类别名称到ID的映射
object_class_id_map = {name: idx for idx, name in enumerate(class_names)}

datadir = "."

max_num = 0

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
                max_num = max(max_num, object_class_id)
            else:
                print(f"Warning: Category '{category_name}' not found in mapping.")
                continue  # 如果没有找到对应的类别ID，则跳过该对象

            # 将结果写入文件
            out_f.write(f'{object_class_id} {x_center} {y_center} {width} {height}\n')

print("转换完成！")
print(max_num)
