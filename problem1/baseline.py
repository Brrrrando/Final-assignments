import torch,argparse,os
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import config

# train stage two
def train(args):
    if os.path.exists(os.path.join(config.save_path, "baseline_loss_top1-acc_top5-acc.txt")):
        os.remove(os.path.join(config.save_path, "baseline_loss_top1-acc_top5-acc.txt"))
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    # load dataset for train and eval
    train_dataset = CIFAR10(root = 'dataset', train=True, transform=config.train_transform, download=True)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    eval_dataset = CIFAR10(root = 'dataset', train=False, transform=config.test_transform, download=True)
    eval_data = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = resnet18(pretrained=False, num_classes=len(train_dataset.classes)).to(DEVICE)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1,args.max_epoch+1):
        model.train()
        total_loss=0
        for batch, (data, target) in enumerate(train_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            loss = loss_criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        save_loss = total_loss / len(train_dataset)*args.batch_size
        print("epoch",epoch,"loss:", save_loss)

        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_baseline_epoch.pth'))

            model.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
                for batch, (data, target) in enumerate(eval_data):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    pred = model(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc

                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top5_acc / data.size(0) * 100))

                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                      "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))
                
                with open(os.path.join(config.save_path, "baseline_loss_top1-acc_top5-acc.txt"), "a") as f:
                    f.write(str(save_loss) + " " + 
                            str(total_correct_1 / total_num * 100) + " " + 
                            str(total_correct_5 / total_num * 100) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=config.baseline_batchsize, type=int, help='')
    parser.add_argument('--max_epoch', default=200, type=int, help='')

    args = parser.parse_args()
    train(args)