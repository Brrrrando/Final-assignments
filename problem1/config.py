# config.py
import os
from torchvision import transforms

use_gpu=True



########################################################################
# 以下参数可修改
gpu_name=0  # gpu编号

baseline_batchsize = 800  # baseline的batchsize

stage1_batchsize = 400  # 自监督（第一阶段）训练的batchsize
stage2_batchsize = 800  # 有监督（第二阶段）训练的batchsize


download_dataset_root = "dataset"     # 存放cifar-10-python.tar.gz的文件夹路径

save_path="weights"   # 保存权重和txt的文件夹路径


########################################################################

########################################################################
# 下面的不用改
pre_model=os.path.join(save_path,'model_stage1_epoch.pth')


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
########################################################################