#!/usr/bin/python3
import os
# os.environ['MASTER_PORT'] = '8888'
# os.environ['MASTER_ADDR'] = '120.27.217.156'
from utils.options import args_parser
from utils.imshow import imshow
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.carnet import carNet
from model.stn_shuf import stn_shufflenet
from model.stn_trans_shuf import stn_trans_shufflenet
# from ssdaugumentations import *
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

transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Resize(256),
        transforms.ColorJitter(),
        # transforms.RandomPerspective(),
        # transforms.RandomRotation(degrees=180),
        transforms.RandomResizedCrop(args.img_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])




if __name__=="__main__":

    print("load_net...")

    if args.model == 'mAlexNet':
        net = mAlexNet()
    elif args.model == 'AlexNet':
        net = AlexNet()
    elif args.model == "carnet":
        net=carNet()
    elif args.model=='stn_shuf':
        net=stn_shufflenet()
    elif args.model=='stn_trans_shuf':
        net=stn_trans_shufflenet()
    # net = net.cuda()
    # for name, parameters in net.named_parameters():
    #     print(name, ':', parameters.size())
    # exit()
    print("weight init..")
    weights_init(net)
    criterion = nn.CrossEntropyLoss()

    # weight_path = "/home/zengweijia/.jupyter/cnrpark/parking_lot_occupancy_detection/weights/carnet_60_0.001.pth"
    # net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_path, map_location="cpu").items()})
    if args.path == '':
        net=train(args.epochs, args.train_img, args.train_lab, transforms=transforms, net=net, criterion=criterion)
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
