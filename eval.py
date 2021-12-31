from torch.autograd import Variable
import os.path as osp
from utils.dataloader import selfData, collate_fn
import torch
import torch.nn as nn
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.trans_shuf import trans_shufflenet
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

minetransforms = torchvision.transforms.Compose([
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Resize((args.img_size,args.img_size)),
        # transforms.ColorJitter(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
    count = [0 for x in range(0,200)]
    # count['_88']=1
    # count['_88']=count['_88']+1
    # print(count)
    # exit()
    # f=open('89.250.150.72.txt','w')
    with torch.no_grad():
        for i in range(test_size):
            # for ind,val in enumerate(count):
            #     print(ind,val)
            split_path = args.split_path
            split_path = osp.join(split_path, str)
            image=test_dataset.pull_img(i)
            # print(test_dataset.img_list[i])
            # if test_dataset.img_list[i].find("_25")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_26")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_27")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_30")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_119") >= 0:
            #     continue
            # if test_dataset.img_list[i].find("_111")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_122")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_150")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_107")>=0:
            #     continue
            # if test_dataset.img_list[i].find("_105")>=0:
            #     continue
            # print(test_dataset.img_list[i].find("_76.jpg"))
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

            # if predicted!=label:
            #     print(predicted, test_dataset.img_list[i])
            #     index=int(test_dataset.img_list[i].split('_')[-1].split('.')[0])
            #     count[index]=count[index]+1
                # cv2.imshow('test',tmp_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            if predicted == 1 and label == 1:
                TP += 1
                split_path = osp.join(split_path, "TP")
            elif predicted == 1 and label == 0:
                FP += 1
                split_path = osp.join(split_path, "FP")
                # f.write(test_dataset.img_list[i]+'\n')
                # print(predicted,test_dataset.img_list[i])
            elif predicted == 0 and label == 1:
                FN += 1
                split_path = osp.join(split_path, "FN")
                # print(predicted,test_dataset.img_list[i])
            elif predicted==0 and label ==0:
                TN += 1
                split_path = osp.join(split_path, "TN")
            # if not os.path.exists(split_path):
            #     os.makedirs(split_path)
            # split_path=osp.join(split_path,repr(i)+".jpg")
            # cv2.imwrite(split_path, tmp_image)
    print("Acc:{}\tTP:{}\tFP:{}\tFN:{}\tTN:{}".format((correct/total),TP, FP, FN, TN))
    # for ind, val in enumerate(count):
    #     print(ind, val)
    return (correct / total)
    # print(count)


# test
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
    elif args.model == 'trans_shuf':
        print("shuf_type:", args.shuf_type)
        net = trans_shufflenet(shuff_type=args.shuf_type)
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    if torch.cuda.is_available():
        net.cuda(device)
    acc=eval(args.test_img,args.test_lab,net,args.eval_data)
    # print(args.path)

