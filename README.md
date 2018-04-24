# 未来杯AI挑战赛区域赛 AI专业组-图像场景分类 baseline

## 概述

组委会提供baseline程序，以边参赛者更好的理解赛题和使用数据。

baseline程序仅提供了简单的模型，未经优化，但可以正常完成数据集训练和测试。

##  数据集下载和GPU使用

数据集下载和GPU使用，请参考 [GPU.md]

## 系统需求
* Python 2.7
* Tensorflow 1.4
* Pillow (可使用pip安装)

## 运行Baseline程序

### 训练模型

使用命令行执行如下操作：

```
python image_scene_classification.py --mode=train --dataset_dir=<dir> --max_steps=<max_steps> --checkpoint_dir=<checkpoint_dir>
```

参数说明

* --mode 设置执行模式，训练时应为train
* --dataset_dir 训练数据集所在目录, 例如： /home/ubuntu/image_scene_training_v1
* --checkpoint_dir checkpoint目录，训练后的模型参数。如果模型有调整或重新训练，需要设置新的目录。
* --max_steps 最大训练次数，到达最大训练次数，程序将自动停止。

### 执行测试

使用命令行执行如下操作：

```
python scene.py --mode=test --dataset_dir=<testset_dir> --checkpoint_dir=<checkpoint_dir> --target_file=<target_file>
```

 --mode 置执行模式，测试时应为 test
 --dataset_dir 测试数据集所在目录, e.g. /home/ubuntu/image_scene_test_v1
 --checkpoint_dir checkpoint目录，训练后的模型参数，必须与训练时指定的参数保持一致。
 --target_file 结果文件存放路径，应该指向一个csv文件地址。

### 关于测试集

测试集未发放期间，参数选手可以自筹数据或从训练集中切出部分数据调试测试程序。

测试集规格参考大赛官网。
