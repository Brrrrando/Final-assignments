## 问题1：对比监督学习与自监督学习在图像分类任务中的性能表现
所需环境：pytorch

数据集：CIFAR-10数据集

训练步骤：
* 准备数据集，存放于dataset中

* 可根据自己计算机性能config.py中的batchsize参数

* baseline模型：运行baseline.py，模型训练结果、训练过程中每个epoch的loss、acc会存入weights文件夹中
* 加入自监督学习后的模型：运行trainstage1.py进行第一阶段的预训练，预训练模型训练结果和loss变化会存入weights文件夹中。完成预训练后运行trainstage2.py进行第二阶段训练，第二阶段的模型训练结果、训练过程中每个epoch的loss、acc同样会存入weights文件夹中
* 绘制loss、acc曲线：运行showlossacc.py，可对训练集loss和测试集acc进行画图，图片会保存在loss_acc_plots文件夹中
