import torch
import torch.nn as nn
from torchvision.models import resnet,alexnet


class Attention(nn.Module):
    def __init__(self, dim, hdim, r=8):
        super(Attention, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.layer2 = nn.Sequential(
            nn.Linear(self.dim, self.r),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.r, self.hdim),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.layer2(input)
        x = self.layer3(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return x


class Local(nn.Module):
    def __init__(self, dim, hdim, r=16):
        super(Local, self).__init__()
        self.dim = dim
        self.hdim = hdim
        self.r = r
        self.layer = Attention(dim, hdim, self.r)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, p):
        a = self.avgpool(p).squeeze()
        a = self.layer(a)
        a = a.view(a.shape[0], a.shape[1], 1, 1)
        a = a * p
        a = a.sum(1).view(a.shape[0], -1)
        a = torch.softmax(a, dim=1)
        a = a.view(a.shape[0], 1, p.shape[2], p.shape[3])
        return self.avgpool(a * p).squeeze()

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.num_classes=num_classes
        ResNet=resnet.resnet18(pretrained=True)
        self.conv1=ResNet.conv1
        self.bn1=ResNet.bn1
        self.relu=ResNet.relu
        self.maxpool=ResNet.maxpool
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.layer3=ResNet.layer3
        self.layer4=ResNet.layer4
        self.avgpool=ResNet.avgpool

        self.attibute1 = Local(512, 512)
        self.attibute2 = Local(512, 512)
        self.attibute3 = Local(512, 512)
        self.attibute4 = Local(512, 512)
        self.attibute5 = Local(512, 512)
        self.attibute6 = Local(512, 512)
        self.classifier=nn.Linear(512*4,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_4 = self.layer4(x)
        att1 = self.attibute1(x_4)
        att2 = self.attibute2(x_4)
        att3 = self.attibute3(x_4)
        att4 = self.attibute4(x_4)
        att5 = self.attibute5(x_4)
        att6 = self.attibute6(x_4)

        x = self.avgpool(x_4)
        x = torch.flatten(x, 1)
        x=torch.cat((x,att1,att2,att3),dim=1)
        x = self.classifier(x)
        return x

# model_test=Model()
# img=torch.rand(2,3,256,256)
# x=model_test(img)


