#!/usr/bin/python3
import os

# import model.SqueezeNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['MASTER_PORT'] = '8888'
# os.environ['MASTER_ADDR'] = '120.27.217.156'
from utils.options import args_parser
from model.stn_shuf import stn_shufflenet
from model.stn_trans_shuf import stn_trans_shufflenet
from utils.imshow import imshow
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.carnet import carNet
import torchvision.models as models
import torch.distributed as dist
from utils.dataloader import selfData
from utils.train import train
from utils.test import test
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init
import torch.multiprocessing as mp
import torch.distributed as dist

args = args_parser()

# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
#初试代码
# transforms = transforms.Compose([
#         transforms.ToTensor(),  # normalize to [0, 1]
#         transforms.Resize(256),
#         transforms.RandomResizedCrop(args.img_size),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#预训练代码
transforms = transforms.Compose([
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
    ])



if __name__=="__main__":

    print("load_net...")
    if args.model == 'mAlexNet':
        net = mAlexNet()
    elif args.model == 'AlexNet':
        net = AlexNet()
    elif args.model == "carnet":
        net=carNet()
    elif args.model=="googlenet":
        net=models.googlenet()
        num_fc = net.fc.in_features
        net.fc = nn.Linear(num_fc, 2)
    elif args.model=="vgg16":
        net=models.vgg16(pretrained=True)
        num_fc = net.classifier[6].in_features
        net.classifier[6] = torch.nn.Linear(num_fc, 2)
        for param in net.parameters():
            param.requires_grad = False
        # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
        for param in net.classifier[6].parameters():
            param.requires_grad = True
    elif args.model=="Inception_v3":
        net=models.inception_v3()
        net.AuxLogits.fc=nn.Linear(768,2)
        net.fc = nn.Linear(2048, 2)
        net.aux_logits=False
        # net=net.cuda()
    elif args.model=="mobilenet_v3_small":
        net=models.mobilenet_v3_small()
        net.classifier[3]=nn.Linear(1024,2)
        # net.fc = nn.Linear(num_fc, 2)
    elif args.model=="mobilenet_v3_large":
        net = models.mobilenet_v3_large()
        net.classifier[3] = nn.Linear(1280, 2)
    elif args.model=="ShuffleNet_v2":
        net=models.shufflenet_v2_x1_0()
        net.fc=nn.Linear(1024,2)
    elif args.model == "mobilenet_v2":
        net = models.mobilenet_v2()
        net.classifier[1] = nn.Linear(1280, 2)
    elif args.model=='stn_shuf':
        net=stn_shufflenet()
    elif args.model=='stn_trans_shuf':
        net=stn_trans_shufflenet()
    elif args.model=='shuf':
        net=torchvision.models.shufflenet_v2_x1_0(pretrained=False,num_classes=2)



    # net = net.cuda()
    # for name, parameters in net.named_parameters():
    #     print(name, ':', parameters.size())
    # exit()

    print("weight init..")

    # weights_init(net)
    criterion = nn.CrossEntropyLoss()
    if args.path == '':
        print(args.model)
        net=train(args.epochs, args.train_img, args.train_lab, transforms, net, criterion)
        # net=train(args.epochs, args.train_img, args.train_lab, transforms, net, criterion)
        PATH = './weights/'+args.model+'.pth'
        torch.save(net.state_dict(), PATH)
    else:
        PATH = args.path
        if args.model == 'mAlexNet':
            net = mAlexNet()
        elif args.model == 'AlexNet':
            net = AlexNet()
        net.cuda()
        net.load_state_dict(torch.load(PATH))
    # accuracy = test(args.test_img, args.test_lab, transforms, net)
    # print("\nThe accuracy of training on '{}' and testing on '{}' is {:.3f}.".format(args.train_lab.split('.')[0], args.test_lab.split('.')[0], accuracy))
