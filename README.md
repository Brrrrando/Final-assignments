# Final-assignments
# 神经网络和深度学习-期末作业
## 问题1：对比监督学习与自监督学习在图像分类任务中的性能表现
所需环境：pytorch

数据集：CIFAR-10数据集

训练步骤：
* 准备数据集，存放于dataset中

* 可根据自己计算机性能config.py中的batchsize参数

* baseline模型：运行baseline.py，模型训练结果、训练过程中每个epoch的loss、acc会存入weights文件夹中
* 加入自监督学习后的模型：运行trainstage1.py进行第一阶段的预训练，预训练模型训练结果和loss变化会存入weights文件夹中。完成预训练后运行trainstage2.py进行第二阶段训练，第二阶段的模型训练结果、训练过程中每个epoch的loss、acc同样会存入weights文件夹中
* 绘制loss、acc曲线：运行showlossacc.py，可对训练集loss和测试集acc进行画图，图片会保存在loss_acc_plots文件夹中


## 问题2：设计Transformer网络模型，与期中作业1的结果对比
epoch:70
alpha:0.5
Cutmix_prob:0.3
lr:1e-4
Weight_decay:1e-3
batch_size:128


## 问题3：使用具有泛化能力的NeRF 模型进行三维物体重建
所需环境：
#torch==1.11.0
torchvision>=0.9.1
imageio
imageio-ffmpeg
matplotlib
configargparse
tensorboard>=2.0
tqdm
opencv-python
*colmap下载地址：https://github.com/colmap/colmap/releases
*1、先在上面的地址下载colmap
*2、将收集好的数据集放在nerf-pytorch\data\nerf_llff_data\CrystalIO\images该目录下
*3、利用colmap对数据集提取特征点并输出结果、配准并重建
*4、运行nerf模型（run_nerf.py）训练并输出重建结果，结果保存在nerf-pytorch\logs\CrystalIO目录下
*5、github上传的模型因为文件数量限制有所删减，如果运行有问题的话可以用百度网盘的那个
*6、重建后的视频上传在网盘部分
