# yolov10-project

## Start

Download the yolov10 project and create the environment:

```bash
git clone https://github.com/THU-MIG/yolov10
cd yolov10
conda create -n yolov10 python=3.9
pip install -r requirments.txt

```

## Datasets preparation

In the dir of yolov10,implement the following command:

```bash
mkdir -p datasets
cd datasets
wget https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip  # get the tt100k_2016
unzip data.zip
mv data tt100k_2016
```
