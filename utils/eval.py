import os.path as osp
from utils.dataloader import selfData, collate_fn
import torch
import torch.nn as nn
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.carnet import carNet
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import numpy as np
from tqdm import tqdm
import os
import cv2
import torch.nn.functional as F
from utils.options import args_parser
from torch.autograd import Variable
args = args_parser()
from ssdaugumentations import *
transforms=Basetransform(size=args.img_size)
def eval(img_path,target_path, net,str="rainy"):
    print("\nTesting starts now...")

    net.eval()
    test_dataset = selfData(img_path, target_path, transforms)
    test_size=len(test_dataset)
    correct = 0
    total = 0
    TP=0
    FP=0
    FN=0
    TN=0
    with torch.no_grad():
        for i in tqdm(range(test_size)):
            split_path = args.split_path
            split_path = osp.join(split_path, str)
            image=test_dataset.pull_img(i)
            if image is None:
                continue
            tmp_image=test_dataset.pull_img(i)
            label=test_dataset.pull_label(i)
            label=int(label)

            x=transforms(image)
            x=Variable(x.unsqueeze(0))
            if torch.cuda.is_available():
                device = torch.device(args.cuda_device)
                x = x.to(device)
                # print(x.shape)
            y=net(x)
            _,predicted=torch.max(y,1)
            # predicted=torch.
            # print(predicted==label)
            # exit()
            total += 1
            correct+=(predicted==label)
            if predicted == 1 and label == 1:
                TP += 1
                split_path = osp.join(split_path, "TP")
            elif predicted == 1 and label == 0:
                FP += 1
                split_path = osp.join(split_path, "FP")
            elif predicted == 0 and label == 1:
                FN += 1
                split_path = osp.join(split_path, "FN")
            elif predicted==0 and label ==0:
                TN += 1
                split_path = osp.join(split_path, "TN")
            if not os.path.exists(split_path):
                os.makedirs(split_path)
            split_path=osp.join(split_path,repr(i)+".jpg")
            cv2.imwrite(split_path, tmp_image)
    print("Acc:{}\tTP:{}\tFP:{}\tFN:{}\tTN:{}".format((correct/total),TP, FP, FN, TN))
    return (correct / total)



if __name__=="__main__":

    net=carNet()
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    if torch.cuda.is_available():
        net.cuda(args.cuda_device)
    acc=eval(args.test_img,args.test_lab,net,args.eval_data)
    # print(args.path)

