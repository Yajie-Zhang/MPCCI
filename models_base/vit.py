import torch
import torch.nn as nn
# from vit_pytorch import ViT
from pytorch_pretrained_vit import ViT

from einops import rearrange, repeat

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.num_classes=num_classes
        VIT=ViT('B_16_imagenet1k',num_classes=num_classes,pretrained=True)
        self.patch_embedding=VIT.patch_embedding
        self.class_token=VIT.class_token
        self.positional_embedding=VIT.positional_embedding
        self.transformer=VIT.transformer
        # self.pre_logits=VIT.pre_logits
        self.norm=VIT.norm
        self.fc=VIT.fc

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        # x = self.pre_logits(x)
        # x = torch.tanh(x)
        x = self.norm(x)[:, 0]  # b,d
        x = self.fc(x)  # b,num_classes
        return x

# MODEL=Model(2)
# a=torch.rand(2,3,384,384)
# b=MODEL(a)
# print(b)
