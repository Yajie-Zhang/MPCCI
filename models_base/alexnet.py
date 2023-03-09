import torch
import torch.nn as nn
from torchvision.models import alexnet,vgg

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.num_classes=num_classes
        AlexNet=vgg.vgg16(pretrained=True)
        self.features = AlexNet.features
        self.avgpool=AlexNet.avgpool
        self.classifier=AlexNet.classifier[:-1]
        self.classifier_add=nn.Linear(4096,num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x=self.classifier_add(x)
        return x

# MODEL=Model(100)
# a=torch.rand(2,3,128,128)
# b=MODEL(a)
# print()