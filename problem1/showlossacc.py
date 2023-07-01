import matplotlib.pyplot as plt

# baseline的acc、loss画图
file_path = r'weights\baseline_loss_top1-acc_top5-acc.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
# 提取每列数据
loss = []
acc_top1 = []
acc_top5 = []
for line in lines:
    values = line.split()
    loss.append(float(values[0]))
    acc_top1.append(float(values[1])/100.0)
    acc_top5.append(float(values[2])/100.0)
# loss画图
plt.figure()
plt.plot(loss)
plt.xlabel('epoch',fontsize=12)
plt.ylabel('loss',fontsize=12)
plt.title('train loss of baseline',fontsize=15)
plt.savefig(( r'loss_acc_plots\baseline_loss.png'))
# acc画图
plt.figure()
plt.plot(acc_top1,label='top1-acc')
plt.plot(acc_top5,label='top5-acc')
plt.xlabel('epoch',fontsize=12)
plt.ylabel('accuracy',fontsize=12)
plt.ylim(0.4, 1)
plt.title('test accuracy of baseline',fontsize=15)
plt.legend()
plt.savefig(( r'loss_acc_plots\baseline_acc.png'))



# 1阶段loss画图
with open('weights\stage1_loss.txt', 'r') as file:
    data = [float(num) for num in file.read().split()]

plt.figure()
plt.plot(data)
plt.xlabel('epoch',fontsize=12)
plt.ylabel('loss',fontsize=12)
plt.title('train loss of stage1 (SimCLR)',fontsize=15)
plt.savefig(( 'loss_acc_plots\stage1_loss_curve.png'))



# 2阶段acc、loss画图
file_path = r'weights\stage2_loss_top1-acc_top5-acc.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
# 提取每列数据
loss = []
acc_top1 = []
acc_top5 = []
for line in lines:
    values = line.split()
    loss.append(float(values[0]))
    acc_top1.append(float(values[1])/100.0)
    acc_top5.append(float(values[2])/100.0)
# loss画图
plt.figure()
plt.plot(loss)
plt.xlabel('epoch',fontsize=12)
plt.ylabel('loss',fontsize=12)
plt.title('train loss of stage2 (resnet-18)',fontsize=15)
plt.savefig(( 'loss_acc_plots\stage2_loss_curve.png'))
# acc画图
plt.figure()
plt.plot(acc_top1,label='top1-acc')
plt.plot(acc_top5,label='top5-acc')
plt.xlabel('epoch',fontsize=12)
plt.ylabel('accuracy',fontsize=12)
plt.ylim(0.4, 1)
plt.title('test accuracy of stage2 (resnet-18)',fontsize=15)
plt.legend()
plt.savefig(( 'loss_acc_plots\stage2_acc_curve.png'))
