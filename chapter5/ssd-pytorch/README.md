# 第五章 SSD

## 简介

该代码主要参考了[amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)的PyTorch复现工程，如在学习时遇到问题，可前往[amdegroot的问题区](https://github.com/amdegroot/ssd.pytorch/issues)查看是否有解决方法。

## 数据集
代码提供了COCO与PASCAL VOC两种数据集的使用方法，在此以VOC2012为例，利用下面脚本可以自动完成下载，当然也可以手动把数据集放到对应文件夹。
```Shell
# 默认数据路径为data/VOCdevkit
sh data/scripts/VOC2012.sh 
```

## 训练
* 利用如下指令下载VGG的预训练权重，并放到默认创建的weights文件夹内，当然，也可以手动从下列网址下载再放到weights文件夹内。
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

* 利用如下指令可以进行模型训练：
```Shell
python train.py
```
根据所需修改脚本中的超参数。

## 前向计算
利用下面脚本进行前向计算：
```Shell
python eval.py
```
根据所需修改脚本中的超参数

