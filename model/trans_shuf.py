import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2
import math
from model.shuffle_net_se import ShuffleNetV2SE
from model.shufflenet_v2_k5_liteconv import ShuffleNetV2K5Lite
from model.shufflenet_v2_k5 import ShuffleNetV2K5
from model.shufflenet_v2_liteconv import ShuffleNetV2LiteConv
from model.shufflenet_v2_sk_attention import ShuffleNetV2SK
from model.shufflenet_v2_csp import ShuffleNetV2CSP
from x_transformers import ViTransformerWrapper, Encoder
from utils.options import args_parser
from torch.autograd import Variable
from model.efficient import ViT
import torchvision.transforms as transforms
args = args_parser()
from ssdaugumentations import *
# transforms=Basetransform(size=args.img_size)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class UP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UP, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self,x):
        x=self.up(x)
        x=self.conv(x)
        return x

class trans_shufflenet(nn.Module):
    def __init__(self, num_classes = 2,shuff_type="shuf",trans_dim=192):
        super(trans_shufflenet, self).__init__()
        self.trans_dim=trans_dim
        self.input_channel = 3
        self.num_output = num_classes
        param={"class_num":num_classes,
                    "channel_ratio":1}
        self.upsampel=nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv
        if shuff_type=="shuf":
            self.shufflenet=torchvision.models.shufflenet_v2_x1_0()
            print(shuff_type)
        elif shuff_type=="shuf_se":
            self.shufflenet=ShuffleNetV2SE(param)
            print(shuff_type)
        elif shuff_type=="shuf_k5_liteconv":
            self.shufflenet=ShuffleNetV2K5Lite(param)
            print(shuff_type)
        elif shuff_type=="shuf_liteconv":
            self.shufflenet=ShuffleNetV2LiteConv(param)
            print(shuff_type)
        elif shuff_type=="shuf_k5":
            self.shufflenet=ShuffleNetV2K5(param)
            print(shuff_type)
        elif shuff_type=="shuf_csp":
            self.shufflenet=ShuffleNetV2CSP(param)
            print(shuff_type)
        elif shuff_type=="shuf_sk":
            self.shufflenet=ShuffleNetV2SK(param)
            print(shuff_type)
        self.up1=UP(self.trans_dim,self.trans_dim//2)
        # self.up2=UP(self.trans_dim//2,self.trans_dim//4)
        # self.up3=UP(self.trans_dim//16,self.trans_dim//64)
        # self.up4 = UP(self.trans_dim // 64, self.trans_dim // 256)
        # self.shufflenet=ShuffleNetV2SE()
        self.Vit= ViT(
                        dim = self.trans_dim,
                        image_size = 256,
                        patch_size = 32,
                        num_classes = 2,
                        transformer = Encoder(
                            dim = self.trans_dim,                  # set to be the same as the wrapper
                            depth = 12,
                            heads = 8,
                            ff_glu = True,              # ex. feed forward GLU variant https://arxiv.org/abs/2002.05202
                            residual_attn = True        # ex. residual attention https://arxiv.org/abs/2012.11747
                        )
                    )
    def forward(self, x):
        tmp=x.clone()
        tmp=tmp.resize_(tmp.shape[0],tmp.shape[1],64,64)
        # print(tmp.shape)
        # exit()
        x = self.Vit(x)
        # print(x.shape)
        # exit()
        x=x.view(x.shape[0],int(math.sqrt(x.shape[1])),-1,x.shape[-1]).permute(0,3,1,2)
        x=x.contiguous()
        x=self.up1(x)
        # print(x.shape)
        # x=self.up2(x)
        # # print(x.shape)
        # x = self.up3(x)
        # # print(x.shape)
        # x=self.up4(x)
        # print(x.shape)
        # exit()
        x=x.view(x.shape[0],6,64,64)
        # x=slf
        x=torch.cat([x,tmp],dim=1)
        x=self.shufflenet(x)
        return x

#
# minetransforms = torchvision.transforms.Compose([
#         transforms.ToTensor(),  # normalize to [0, 1]
#         transforms.Resize((256,256)),
#         # transforms.ColorJitter(),
#         transforms.Normalize(mean=[0.5,0.5,0.5],
#                              std=[0.5, 0.5, 0.5])
#     ])
# net=trans_shufflenet(shuff_type="shuf_k5_liteconv")
# path="/home/zengweijia/project/pklot_detection/img.png"
# img=cv2.imread(path)
# x=minetransforms(img)
# x=Variable(x.unsqueeze(0))
# print(x.shape)
# out=net(x)
# print(out)
