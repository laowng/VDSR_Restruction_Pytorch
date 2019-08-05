# 测试集测试
import PSNR
import torch
import random
import torch.utils.data as Data
from torchvision.transforms import ToTensor, ToPILImage
from dataset import TrainDatasetFromFolder



tran_im=ToPILImage()
tran_ten=ToTensor()


test_set = TrainDatasetFromFolder("./data/test_set_s3.h5")#h5数据集制作工具在data中，可自己制作
test_loader = Data.DataLoader(dataset=test_set,num_workers=1,batch_size=40, shuffle=False)
net=torch.load("./checkpoint/model_epoch_100.pth")["model"]

net.cpu()
i=random.randint(0,100)

aa,bb=test_loader.dataset[i]

lim1=tran_im(aa)
label=tran_im(bb)



lim1.show()

print("CUBIC_PSNR:",PSNR.psnr(lim1,label))

T_X=torch.unsqueeze(tran_ten(lim1),0)


prediction=net(T_X)
prediction[prediction<0]=0
prediction[prediction>1]=1



lim2=tran_im(prediction[0])
lim2.show()

print("NET_PSNR:",PSNR.psnr(lim2,label))
label.show()


