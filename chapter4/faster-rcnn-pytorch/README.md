# 第四章 Faster RCNN

## 简介

该代码主要参考了[jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)的PyTorch复现工程，如在学习时遇到问题，可前往[jwyang的问题区](https://github.com/jwyang/faster-rcnn.pytorch/issues)查看是否有解决方法。

## 准备工作

首先clone本书代码到本地：
```
git clone https://github.com/dongdonghy/Detection-PyTorch-Notebook.git
```

然后切换到本代码：
```
cd Detection-PyTorch-Notebook/faster-rcnn.pytorch
```

### 依赖

* Python 2.7或者3.6
* Pytorch 0.4.0
* CUDA 8.0或者更高

### 数据准备

* **PASCAL_VOC 07+12**: 如果是VOC的数据集，按照[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)的方法准备VOC数据集，并创建软连接到data文件夹。

* **COCO**: 如果是COCO数据集，则按照[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)的方法准备COCO数据集，并创建软连接到data文件夹。

### 预训练权重

作者提供了VGG与ResNet101两个不同的预训练权重:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

下载相应的预训练权重，并放到data/pretrained_model文件夹下，从实验发现caffe得到的预训练权重模型精度更高，因此使用了caffe的预训练权重。

### 编译

由于NMS、RoI Pooling、RoI Align等模块依赖于自己实现的CUDA C代码，因此这部分需要单独进行编译，首先在lib/make.sh中将CUDA_ARCH改为自己GPU对应的arch，对应表如下：

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

更多关于arch的介绍可以参考官方介绍：[cuda-gpus](https://developer.nvidia.com/cuda-gpus) 或者[sm-architectures](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

使用pip安装Python的依赖库：
```
pip install -r requirements.txt
```

编译依赖CUDA的库：

```
cd lib
sh make.sh
```

## 训练

训练Faster RCNN指令如下：这里默认使用VOC数据集、VGG16的预训练模型，众多超参可根据实际情况修改。
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py \

这里使用了train_net.py中的默认参数，众多超参可根据实际情况修改。Batch Size以及Worker Number可根据GPU能力合理选择。

```
## 前向测试

训练完想要测试模型在测试集上的前向效果，运行如下指令：
```
python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
SESSION、EPOCH、CHECKPOINT修改为自己想要前向测试的模型


