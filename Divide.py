import os
import shutil

images_train_dir = 'train/images'
images_test_dir = 'test/images'
images_other_dir = 'other/images'
txts_train_dir = 'train/labels'
txts_test_dir = 'test/labels'
txts_other_dir = 'other/labels'

os.makedirs(txts_train_dir, exist_ok=True)
os.makedirs(txts_test_dir, exist_ok=True)
os.makedirs(txts_other_dir, exist_ok=True)

# 获取训练集和测试集的图像文件名（不带扩展名）
train_image_files = [f[:-4] for f in os.listdir(images_train_dir) if f.endswith('.jpg') or f.endswith('.png')]
test_image_files = [f[:-4] for f in os.listdir(images_test_dir) if f.endswith('.jpg') or f.endswith('.png')]
other_image_files = [f[:-4] for f in os.listdir(images_other_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 遍历训练集的图像文件名，找到对应的txt文件并移动
for img_name in train_image_files:
    txt_file = f'{img_name}.txt'
    if os.path.exists(f'{txt_file}'):# 假设txt文件在当前目录下
        shutil.move(f'{txt_file}', os.path.join(txts_train_dir, txt_file))

# 遍历测试集的图像文件名，找到对应的txt文件并移动
for img_name in test_image_files:
    txt_file = f'{img_name}.txt'
    if os.path.exists(f'{txt_file}'):  # 假设txt文件在当前目录下
        shutil.move(f'{txt_file}', os.path.join(txts_test_dir, txt_file))

for img_name in other_image_files:
    txt_file = f'{img_name}.txt'
    if os.path.exists(f'{txt_file}'):# 假设txt文件在当前目录下
        shutil.move(f'{txt_file}', os.path.join(txts_other_dir, txt_file))

print("TXT files have been successfully sorted into train and test directories.")