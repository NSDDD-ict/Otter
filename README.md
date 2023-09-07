# Otter nsddd
## 安装环境
1. git clone https://github.com/NSDDD-ict/Otter
2. conda create -n Otter python=3.10
3. pip install -r requirements.txt
## 下载数据
1. 数据获取链接
```
初赛数据集

    百度网盘
    链接：https://pan.baidu.com/s/1rQol1MkYoQJFpQcD971FfA?pwd=x6rq
    提取码：x6rq

    夸克网盘
    链接：https://pan.quark.cn/s/a862dc0d898f
    提取码：UZBV
```
## 处理数据
1. 解压数据
```
cd Otter
mkdir data

#拷贝数据
cp -r /data/dwq_data/* data  

#解压数据
cd data
unzip test.zip
unzip train.zip
unzip annotation_with_ID.zip 
```
2. 提取视频特征
```
mkdir all_videos

python3 unittest/mv_videos.py --raw_data_path /data/chengshuang/Otter/data --all_videos_path /data/chengshuang/Otter/data/all_videos

# 生成帧 json 文件
python3 mimic-it/convert-it/main.py --name video.DenseCaptions --num_threads 32 --image_path /data/chengshuang/Otter/data/all_videos 
```
## 生成训练数据
1. 生成 FunQA128_instructions_train.json
```
python3 anno4otter.py --train_video_path /data/chengshuang/Otter/data/train --annotation_json_path /data/chengshuang/Otter/data/annotation_with_ID/ --output_name FunQAf128_instructions_train.json --all_video_path /data/chengshuang/Otter/data/all_videos/
```
2. 生成 FunQA128_train_train.json
```
python3 generate_train_json.py --input_file '/data/chengshuang/Otter/output/FunQAf128_instructions_train.json' --output_file /data/chengshuang/Otter/output/FunQAf128_train_train.json
```


## 训练
1. fsdp训练
```
bash script/train_fsdp.sh
```
2. zero2训练
```
bash script/train_zeros.sh
```

## 测试
1. 转换成 huggingface checkpoint
```
bash script/convert_otter_to_hf.sh
```
2. 推理
```
python3 pipeline/demo/otter_batch_infer.py
```

## 计算分数
参考 使用colab 
https://colab.research.google.com/drive/1Sqc5w-ws9fG-7S5WGrzZcy8syqSidUeK?usp=sharing

