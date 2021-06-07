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
class trans_shufflenet(nn.Module):
    def __init__(self, num_classes = 2):
        super(trans_shufflenet, self).__init__()
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

    def forward(self, x):
        x = self.Vit(x)
        x=x.view(x.shape[0],int(math.sqrt(x.shape[1])),-1,x.shape[-1]).permute(0,3,1,2)
        x=x.contiguous()
        x=x.view(x.shape[0],3,56,56)

        x=self.shufflenet(x)
        return x

