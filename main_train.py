import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import TrainDatasetFromFolder
from draw import draw
from matplotlib import pyplot as plt

#全局参数设置
parser=argparse.ArgumentParser(description="VDSR,全局参数")
parser.add_argument("--batch_size","-b", type=int, default=64, help="batch_size set,默认：128")
parser.add_argument("--nEpochs",     "-e", type=int, default=100, help="总训练次数，默认：50")
parser.add_argument("--lr",        "-l", type=float, default=0.1, help="learning rate, 默认：0.1")
parser.add_argument("--step",      "-s", type=int, default=20, help="步长，默认：20")
parser.add_argument("--cuda",      "-c", action="store_true", help="是否使用cuda，默认：false")
parser.add_argument("--resume",    "-r", default="", help="继续上次训练的路径，默认不使用")
parser.add_argument("--start_epoch",'-se', type=int,default=1,help="开始的epoch，默认：1")
parser.add_argument("--clip",      '-cl', type=float, default=0.9,help="梯度出现的最大值,防止梯度爆炸")
parser.add_argument("--momentum",  "-m", default=0.9, type=float, help="动量，默认: 0.9")
parser.add_argument("--weight_decay", "-wd", default=0.0001, type=float, help="权重衰减，默认: 1e-4")

def main():
    global par, model
    par = parser.parse_args()
    print(par)

    print("===> 建立模型")
    # model=Net() #模型
    model=torch.load("./checkpoint/pre_model.pth")["model"]
    criterion=nn.MSELoss(reduction='sum') #损失函数reduction='sum'
    print("===> 加载数据集")
    train_set = TrainDatasetFromFolder("./data/train_set.h5")
    train_loader = DataLoader(dataset=train_set,num_workers=1,batch_size=par.batch_size, shuffle=True)
    print("===> 设置 GPU")
    cuda = par.cuda
    if cuda :
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()
        else:raise Exception("没有可用的显卡设备")

    # optionally resume from a checkpoint
    if par.resume:
        if os.path.isfile(par.resume):
            checkpoint=torch.load(par.resume)
            par.start_epoch=checkpoint['epoch']
            model.load_state_dict(checkpoint["model"].statedict())

    print("===> 设置 优化器")
    optimizer = optim.SGD(model.parameters(), lr=par.lr, momentum=par.momentum, weight_decay=par.weight_decay)

    print("===> 进行训练")
    plt.figure(figsize=(8, 6), dpi=80)
    draw_list = []
    for epoch in range(par.start_epoch, par.nEpochs + 1):
        draw_list=train(train_loader, optimizer, model, criterion, epoch,draw_list)
        save_checkpoint(model, epoch)
        draw(range(1,len(draw_list)+1), 10, draw_list, 10,{"EPOCH:":epoch,"LR:":round(optimizer.param_groups[0]["lr"],4)})
    plt.show()





def adjust_learning_rate(optimizer, epoch,step,end_step, rate):# 学习率修改函数
    if epoch > end_step:
        return
    if epoch == 1 or epoch % step-1 != 0:
        return
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * rate


def train(training_data_loader, optimizer, model, criterion, epoch,draw_list=[]):
    adjust_learning_rate(optimizer, epoch, par.step,70,0.1)
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    loss_sum=0

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0],batch[1]
        if par.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),par.clip)#/optimizer.param_groups[0]["lr"]
        optimizer.step()
        loss_sum+=loss.item()
        print("iteration:",iteration)
    print("===> Epoch[{}](iterations:{}): Loss: {:.10f}".format(epoch, len(training_data_loader), loss_sum))
    jilu = open("./jilu.txt", "a")
    jilu.writelines("\n" + str(epoch) + ":  lr:" + str(optimizer.param_groups[0]["lr"])[0:7] + "     " + str(loss_sum))
    jilu.close()
    draw_list.append(loss_sum)
    return draw_list

def save_checkpoint(model, epoch):
    if epoch % 10 == 0:
        model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch ,"model": model}
        if not os.path.exists("checkpoint/"):
            os.makedirs("checkpoint/")

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()