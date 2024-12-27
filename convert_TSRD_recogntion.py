def convert_to_yolov10_format(input_file, output_dir):
    # 确保输出目录存在
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取文件并转换每一行
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            # if len(parts) != 8:
            #     continue  # 跳过格式不正确的行

            # 解析数据
            image_name, img_width, img_height, x1, y1, w, h, class_id, waste = parts
            img_width, img_height, x1, y1, w, h, class_id = int(img_width), int(img_height), int(x1), int(y1), int(
                w), int(h), int(class_id)

            # 计算中心点坐标和目标框的宽度和高度
            x_center = (x1 + w / 2)
            y_center = (y1 + h / 2)
            width = w
            height = h

            # 计算归一化的坐标
            norm_x_center = x_center / img_width
            norm_y_center = y_center / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            image_name = image_name.replace('.png', '')

            # 创建输出文件名和路径
            output_file_name = f"{image_name}.txt"
            output_file_path = os.path.join(output_dir, output_file_name)

            # 写入YOLOv10格式的标签
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")


# 使用示例
convert_to_yolov10_format('./TSRD-Test Annotation/TsignRecgTest1994Annotation.txt', './TSRD-Test/labels')
convert_to_yolov10_format('./TSRD-Train Annotation/TsignRecgTrain4170Annotation.txt', './tsrd-train/labels')