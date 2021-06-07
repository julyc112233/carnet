from torch.autograd import Variable
import os.path as osp
from utils.dataloader import selfData, collate_fn
import torch
import torch.nn as nn
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.carnet import carNet
from model.stn_shuf import stn_shufflenet
from model.stn_trans_shuf import stn_trans_shufflenet
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from utils.options import args_parser
import numpy as np
from tqdm import tqdm
import os
import cv2
import torch.nn.functional as F
from ssdaugumentations import *
from tqdm import tqdm
import torch.nn.functional as F
import cv2
args = args_parser()
device=int(args.cuda_device)
# if args.eval_data=="PKLot":
#     minetransforms = torchvision.transforms.Compose([
#         transforms.ToTensor(),  # normalize to [0, 1]
#         transforms.Resize((args.img_size,args.img_size)),
#         # transforms.ColorJitter(),
#         transforms.Normalize(mean=[0.3708, 0.3936, 0.3976],
#                              std=[0.0152, 0.0115, 0.0250]),
#         # transforms.Normalize(mean=[0.3518, 0.4025, 0.4123],
#         #                      std=[0.0744, 0.0760, 0.0752)])
#     ])
# elif args.eval_data=='cnrext':
#     minetransforms = torchvision.transforms.Compose([
#         transforms.ToTensor(),  # normalize to [0, 1]
#         transforms.Resize((args.img_size, args.img_size)),
#         # transforms.ColorJitter(),
#         # transforms.Normalize(mean=[0.4993, 0.5231, 0.5268],
#         #                      std=[0.0744, 0.0785, 0.0776]),
#         transforms.Normalize(mean=[0.3450, 0.3949, 0.4050],
#                              std=[0.0209, 0.0064, 0.0152]),
#     ])
minetransforms = torchvision.transforms.Compose([
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Resize((args.img_size,args.img_size)),
        # transforms.ColorJitter(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5, 0.5, 0.5])
    ])
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

            x=minetransforms(image)
            x=Variable(x.unsqueeze(0))
            if torch.cuda.is_available():
                x=x.cuda(device)
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

    if args.model=="carnet":
        net=carNet()
    elif args.model=="mAlexNet":
        net=mAlexNet()
    elif args.model=='stn_shuf':
        net=stn_shufflenet()
    elif args.model=='stn_trans_shuf':
        net=stn_trans_shufflenet()
    elif args.model=='shuf':
        net=torchvision.models.shufflenet_v2_x1_0(pretrained=False,num_classes=2)
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    if torch.cuda.is_available():
        net.cuda(device)
    acc=eval(args.test_img,args.test_lab,net,args.eval_data)
    # print(args.path)

