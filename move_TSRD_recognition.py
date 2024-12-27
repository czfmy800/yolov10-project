import os
import random
import shutil

def move_to_validation(train_img_dir, train_label_dir, validation_img_dir, validation_label_dir, move_percentage=0.2):
    """
    将一定比例的训练集图片和标签文件移动到验证集目录。

    :param train_img_dir: 训练集图片所在目录
    :param train_label_dir: 训练集标签文件所在目录
    :param validation_img_dir: 验证集图片目录
    :param validation_label_dir: 验证集标签目录
    :param move_percentage: 移动到验证集的图片和标签比例，默认是0.2（20%）
    """
    os.makedirs(validation_img_dir, exist_ok=True)
    os.makedirs(validation_label_dir, exist_ok=True)

    img_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    label_files = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]

    total_files = len(img_files)
    num_files_to_move = int(total_files * move_percentage)

    selected_files = random.sample(img_files, num_files_to_move)

    for img_file in selected_files:
        label_file = img_file.replace('.png', '.txt')

        img_src = os.path.join(train_img_dir, img_file)
        label_src = os.path.join(train_label_dir, label_file)

        img_dst = os.path.join(validation_img_dir, img_file)
        label_dst = os.path.join(validation_label_dir, label_file)

        shutil.move(img_src, img_dst)
        shutil.move(label_src, label_dst)

    print(f"已将{num_files_to_move}个图片和标签文件从训练集移动到验证集。")

# 使用示例
move_to_validation(
    train_img_dir='./TSRD-Train/images',
    train_label_dir='./TSRD-Train/labels',
    validation_img_dir='./TSRD-Val/images',
    validation_label_dir='./TSRD-Val/labels',
    move_percentage=0.2
)
