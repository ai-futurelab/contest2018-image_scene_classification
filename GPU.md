# GPU服务器使用教程

## 硬件型号和配置

本次大赛初赛免费为选手提供GPU计算资源。GPU服务器为金山云P3I.14B1，配置如下：
* GPU：Tesla P4 x 1
* vCPU：Xeon E5 v4 14核
* 内存：DDR4 120GB
* 数据盘：本地SSD 500GB

## 软件环境

GPU服务器默认安装：
* Ubuntu Linux 16.04-64位
* CUDA 8.0
* Python 2.7
* Tensorflow 1.4

我们也可提供 CentOS 7.2/3镜像，如有需求，请联系小助手申请重置。

我们暂不提供桌面版本Linux，请使用命令行进行操作。

## 如何登陆

* Windows 环境，可以使用 PuTTY 等SSH 终端软件登录，配置host, 填入IP地址、用户名（ubuntu）和密码即可登录。
* Linux、Mac OS 可以使用系统自带的"终端"直接登录。命令如下：

```bash
# IP地址请替换为申请时下发的IP，输入正确的用户名和密码即可登录。
ssh ubuntu@111.222.10.123
```

* Mac OSX 系统推荐使用Terminus 终端，可获得更好的操作体验。

## 挂载数据盘

系统提供的数据盘需要自行操作挂载，命令如下：

```bash
sudo mkfs.ext4  /dev/vdb
sudo mkdir /data
sudo mount /dev/vdb /data
# 执行 df -h 可以查看到磁盘空间被挂载到 /data目录下
```

## 下载和解压数据集

使用如下命令即可通过内网环境告诉下载数据集：

```base
# 声纹识别训练集
wget http://10.0.0.4/voiceprint_training.zip
# 声纹识别训练集 SHA1SUM
wget http://10.0.0.4/voiceprint_training.zip.sha1

# 图像场景识别训练集
wget http://10.0.0.4/image_scene_training_v1.zip
# 图像场景识别训练集 SHA1SUM
wget http://10.0.0.4/image_scene_training_v1.zip.sha1
```

比如数据集完整性：

```bash
# 以声纹数据集为例
sha1sum voiceprint_training.zip
cat voiceprint_training.zip.sha1
# 比对sha1sum 值一致，即成功完整下载。
```

解压数据集：

```base
# 以声纹数据集为例
unzip voiceprint_training.zip
# 系统会提示输入密码，解压密码请与微信小助手联系取得。
```

## 上传和下载程序和模型

您可以通过SCP命令远程拷贝你开发的程序。

在本地打开终端，执行如下命令：

```bash
# 从本地拷贝到服务器, 目录和IP地址根据实际情况填入。
scp -r /myprojects/mycodes ubuntu@111.222.10.123:/data/

# 从远程拷贝到本地：
scp -r ubuntu@111.222.10.123:/data/mycodes /myprojects
```

您也可以考虑使用gitlab一类的代码托管平台，使用git传递代码，考虑到竞赛期间的代码保密，建议使用private仓库，竞赛结束后再考虑是否转为public仓库。

数据训练后，记得下载保存训练好的模型用于测试集的计算，否则您的计算成果有可能丢失。

## 安装所需的其他程序

您可以使用 apt-get 命令安装所需的软件包。

```
# 例如安装 g++
sudo apt-get install g++
```

您也可以选择自行下载源码编译，具体请查询所需软件包的文档。

## 安装Python软件包

推荐使用pip 安装所需的Python软件包

```
# 例如安装keras
pip install keras
```

有时，pip和apt-get 需要交替操作，例如安装cv2的时候。具体可查询所需软件包的文档。

## 在GPU运行计算框架

Tensorflow等计算框架，均提供了GPU和CPU版本。

默认镜像安装了CUDA驱动程序，Python 2.7 和 Tensorflow 1.4 GPU版。

在命令行输入python，交互式编程界面中录入如下代码：
```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

如果系统提示如下信息，即认为您的tensorflow 程序可使用GPU。如运行于CPU下，很可能达不到预期的提速效果。

```
Found device 0 with properties:
name: Tesla P4 major: 6 minor: 1 memoryClockRate(GHz): 1.1135
pciBusID: 0000:00:07.0
totalMemory: 7.43GiB freeMemory: 7.31GiB

I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla P4, pci bus id: 0000:00:07.0, compute capability: 6.1)
```
