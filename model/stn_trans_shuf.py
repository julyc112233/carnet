import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2
import math
from x_transformers import ViTransformerWrapper, Encoder
from utils.options import args_parser
from torch.autograd import Variable
from model.efficient import ViT
args = args_parser()
from ssdaugumentations import *
transforms=Basetransform(size=args.img_size)
class stn_trans_shufflenet(nn.Module):
    def __init__(self, num_classes = 2):
        super(stn_trans_shufflenet, self).__init__()
        self.input_channel = 3
        self.num_output = num_classes
        self.shufflenet=torchvision.models.shufflenet_v2_x1_0(pretrained=False,num_classes=num_classes)
        self.Vit= ViT(
                        dim = 192,
                        image_size = args.img_size,
                        patch_size = 8,
                        num_classes = 2,
                        transformer = Encoder(
                            dim = 192,                  # set to be the same as the wrapper
                            depth = 12,
                            heads = 8,
                            ff_glu = True,              # ex. feed forward GLU variant https://arxiv.org/abs/2002.05202
                            residual_attn = True        # ex. residual attention https://arxiv.org/abs/2012.11747
                        )
                    )
        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(24, 36, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(36 * 10 * 10, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 使用身份转换初始化权重/偏差
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # 空间变换器网络转发功能
    def stn(self, x):
        xs = self.localization(x)

        xs = xs.view(-1, 10 * 10 * 36)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid,align_corners=True)

        return x

    def forward(self, x):
        x=self.stn(x)
        x = self.Vit(x)
        x=x.view(x.shape[0],int(math.sqrt(x.shape[1])),-1,x.shape[-1]).permute(0,3,1,2)
        x=x.contiguous()
        x=x.view(x.shape[0],3,56,56)

        x=self.shufflenet(x)
        return x

