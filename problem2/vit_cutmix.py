import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from config import *
from model import VisionTransformer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs')

transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=4)

t = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=t)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=4)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            n_classes=n_classes,
            embedding_dimension=embedding_dimension,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_p=proj_p,
            attn_p=attn_p,
        )
model.cuda()

from torchsummary import summary
summary(model, input_size=(3, 32, 32))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-3)

num_epochs = 70
alpha = 0.5
cutmix_prob = 0.3

for epoch in range(num_epochs):
    
    total_right = 0
    total = 0
    total_loss = 0
    
    for data in trainloader:
        inputs, labels = data
        r = np.random.rand(1)
        inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
        
        optimizer.zero_grad()
        
        if alpha > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(inputs.size()[0])
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            loss = mixup_criterion(loss_fn, outputs, target_a, target_b, lam)
        else:
            # compute output
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)


        loss.backward()
        optimizer.step()
        
        predicted = outputs.data.max(1)[1]
        total += labels.size(0)
        total_loss += loss.item()
        total_right += predicted.eq(labels.data).cpu().sum().float()
    writer.add_scalar(tag="train_accuracy", # 可以暂时理解为图像的名字
                scalar_value=total_right/total,  # 纵坐标的值
                global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                )
    writer.add_scalar(tag="train_loss", # 可以暂时理解为图像的名字
                scalar_value=total_loss/total,  # 纵坐标的值
                global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                )
    print("Training accuracy for epoch {} : {}".format(epoch+1,total_right/total))
    
    if (epoch+1)%5==0:
        torch.save(model,'vit_para_cutmix.ckpt')

    # if (epoch+1)%1==0:
    #     my_model = torch.load('hw4_para_cutmix.ckpt')

    total_right = 0
    test_loss = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images,labels = data
            images, labels = Variable(images).cuda(),Variable(labels).cuda()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predicted = outputs.data.max(1)[1]
            total += labels.size(0)
            total_right += (predicted == labels.data).float().sum()
            test_loss += loss.item()
    writer.add_scalar(tag="test_accuracy", # 可以暂时理解为图像的名字
                scalar_value=total_right/total,  # 纵坐标的值
                global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                )
    writer.add_scalar(tag="test_loss", # 可以暂时理解为图像的名字
                scalar_value=test_loss/total,  # 纵坐标的值
                global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                )
    
    print("Test accuracy: %d" % (100*total_right/total))
