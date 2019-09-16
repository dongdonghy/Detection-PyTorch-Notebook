# 第六章 YOLO v2

## 简介

该代码主要参考了[longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)的PyTorch复现工程，如在学习时遇到问题，可前往[longcw的问题区](https://github.com/longcw/yolo2-pytorch/issues)查看是否有解决方法。

## 准备工作

### 1 编译
* 编译 reorg 模块，修改mask.sh中的arch，具体可参考第四章。
    ```bash
    cd yolo2-pytorch
    ./make.sh
    ```
### 2 数据集
* 以VOC2012为例，将数据集建立软链接到data文件夹下：
    
    ```bash
    cd yolo2-pytorch
    mkdir data
    cd data
    ln -s "your VOCdevkit path" VOCdevkit2012
    ```
### 3 预训练权重    
* 下载预训练权重[darknet19](https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view?usp=sharing)
* 然后在`yolo2-pytorch/cfgs/exps/darknet19_exp1.py`中修改权重的路径。

## 训练
* 运行如下指令：
    ```python
    python train.py
    ```

## 前向计算

* 在`yolo2-pytorch/cfgs/config.py`中修改trained_model的路径。
    ```bash
    mkdir output
    python test.py
    ```
