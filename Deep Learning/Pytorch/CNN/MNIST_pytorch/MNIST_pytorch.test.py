import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 数据集的预处理
#torchvision.transforms.Compose：将多个transform组合起来使用
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]
)

data_path = r'./MNIST_DATA_PyTorch'
# 获取数据集
test_data = mnist.MNIST(data_path,train=False,transform=data_tf,download=False)

#获取迭代数据
test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True)

# 定义网络结构 （28+1-3)/2+1=14
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            #Conv2d： 输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out
            #torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
            #BatchNorm2d：对小批量(mini-batch)3d数据组成的4d输入进行批标准化(Batch Normalization)操作
            torch.nn.Conv2d(in_channels=1,  
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),  #输入的每一条边补充0的层数,这儿1层
            torch.nn.BatchNorm2d(num_features = 16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64,100)
        self.mlp2 = torch.nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x
model = CNNnet()
#print(model)


# 测试网络
model = torch.load(r'./MNIST_Weight_PyTorch/log_CNN')
accuracy_sum = []
for i,(test_x,test_y) in enumerate(test_loader):
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    out = model(test_x)
    # print('test_out:\t',torch.max(out,1)[1])
    # print('test_y:\t',test_y)
    accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
    accuracy_sum.append(accuracy.mean())
    print('accuracy:\t',accuracy.mean())

print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
# 精确率图
print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
plt.figure('Accuracy')
plt.plot(accuracy_sum,'o',label='accuracy')
plt.title('Pytorch_CNN_Accuracy')
plt.legend()
plt.show()