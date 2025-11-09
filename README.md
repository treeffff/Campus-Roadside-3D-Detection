# 校园路侧3D目标检测

## 解决方案

本项目采用BEVHeight算法, 是传统的LSS(Lift-Splat-Shoot)方案的模型, 该模型主要由4个部分组成:

1. 图像特征提取模块: 项目采用官方预训练的ResNet101.
2. 像素级高度预测模块: 项目通过卷积网络预测每一个像素点的高度热力图
3. 视角转换模块: 通过数学关系将像素的高度转化为深度, 利用编译好的voxel_pooling模块实现从图像特征到BEV特征的投影
4. 目标检测头: 项目采用CenterNet风格的检测头, 对输入的BEV特征进行3D属性提取



## 环境安装

本项目采用传统自动驾驶的环境风格, 需要下载mmdetection3d库, 编译voxel_pooiling等算子

本项目的训练测试均在单张3090上进行, PyTorch 1.9.0  Python 3.8(ubuntu18.04)  CUDA 11.1

```
pip install pytorch-lightening==1.5.10
pip install torch==1.9.0+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchmetrics==0.7.0

pip install opencv-python-headless

# 需要安装mmcv-full==1.4.0 mmdet==2.19.0 mmdet3d==0.18.1
pip install -U openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.19.0

# 在官网下载mmdetection3d对应版本的库并解压编译
tar -zxvf mmdetection3d-0.18.1.tar.gz
cd mmdetection3d
pip install numba
pip install -v -e .

# 下载pypcd并编译
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install

# 在projcet/code文件夹下运行并编译
pip install -r requirements.txt
python setup.py develop
```

## 模型训练过程

### 数据预处理

假定文件的组织如下:

```
xfdata
-train
	-train_images
	-train_calibs
	-train_labels
-test
	-images
	-calibs
```

将原始数据转化为nuscenes风格的data_info文件, 便于在训练时直接使用NuscenesDataset

```python
# 请确保在project文件目录下
python ./code/data_converter/gen_info.py 
# 在user_data/data_info目录下生成nuscenes风格的info文件
```

### 模型训练

```python
# 请确保在project文件目录下运行
python ./code/exps/bev_height_lss_r101_864_1536_256x256_140_train.py --amp_backend native -b 4 --gpus 1

# 或者在project/code文件目录下运行
chmod +x train.sh
./train.sh
```

### 模型预测

因为时间有限, 项目没有写专用的推理代码, 而是直接转化evaluation过程的结果

```python
# 请确保在project文件目录下运行
python ./code/exps/bev_height_lss_r101_864_1536_256x256_140_test.py --ckpt_path ./user_data/model_weights -e -b 4 --gpus 1

# 或者在project/code文件目录下运行
chmod +x test.sh
./test.sh
# 代码会在目标目录prediction_result/result下生成结果
```

### 注
本仓库基于 [ADLab-AutoDrive/BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight) Fork，用于学习/研究/个人项目。

