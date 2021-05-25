import os
import random
import torch
from torch.autograd import Variable
import cv2
import matplotlib
from utils.dataloader import *
import argparse
import pandas as pd
import os.path as osp
import numpy as np
from model.carnet import carNet
from model.malexnet import mAlexNet

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_img_path',type=str,default="/Users/julyc/Downloads/CNR-EXT_FULL_IMAGE_1000x750",help="full img file path")
    parser.add_argument('--test_img',type=str,default="/Users/julyc/Downloads/CNR-EXT-Patches-150x150/PATCHES")
    parser.add_argument('--test_lab', type=str, default="/Users/julyc/Downloads/splits/CNRPark-EXT/all.txt")
    parser.add_argument('--save_path',type=str,default="./data/visualize/")
    parser.add_argument('--net',type=str,default="carnet")
    parser.add_argument('--weight_path',type=str,default="/Users/julyc/PycharmProjects/parking_lot_occupancy_detection/carnet (1).pth")
    parser.add_argument("--days",type=int,default=2)
    parser.add_argument("--img_perday",type=int,default=2)
    parser.add_argument('--cuda_device', type=int, default=3)
    parser.add_argument('--img_size',type=int,default=54)
    args = parser.parse_args()
    return args

args=arg_parser()
transforms = transforms.Compose([
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Resize((args.img_size,args.img_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
weather=["SUNNY","RAINY","OVERCAST"]
def change_path(seg_img_id):
    # print(seg_img_id)
    seg_img_id=seg_img_id.split('/')[-1]
    seg_img_id=seg_img_id.split('_')
    seg_img_id=seg_img_id[1]+"_"+seg_img_id[2]
    seg_img_id=seg_img_id.replace('.','')
    return seg_img_id

def judge(full_img_id,seg_img_id):
    str1=full_img_id.split('/')[-2]
    str2=seg_img_id.split('/')[-2]
    return str1==str2 and full_img_id.find(change_path(seg_img_id))>=0

def check(net,img,label,transform=transforms):
    net.eval()
    with torch.no_grad():
        label = int(label)
        x = transform(img)
        x = Variable(x.unsqueeze(0))
        if torch.cuda.is_available():
            device = torch.device(args.cuda_device)
            x = x.to(device)
            # print(x.shape)
        y = net(x)
        _, predicted = torch.max(y, 1)
    return predicted==label


def solve(cameras,net,img_loader,save_path):
    count=0
    wi_ratio = 1000/2592
    hi_ratio = 750/1944
    file_path=args.full_img_path
    annot_path=osp.join(file_path,cameras+'.csv')
    img_path=osp.join(file_path,"FULL_IMAGE_1000x750")
    df=pd.read_csv(annot_path)
    test_data=img_loader
    for wt in weather:
        print("weather:",wt)
        img_dir1=osp.join(img_path,wt)
        file_dir=os.listdir(img_dir1)
        # file_dir=[  for path in file_dir]
        file_dir=random.sample(file_dir,args.days)
        print(file_dir)
        for dir in file_dir:
            if dir.startswith('.'):
                continue
            img_dir2=osp.join(img_dir1,dir,cameras)#/Users/julyc/Downloads/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-22/camera1
            img_dir3=os.listdir(img_dir2)
            img_dir3=random.sample(img_dir3,args.img_perday)
            # print(img_dir3)
            # exit()
            for full_img_path in img_dir3:
                print(full_img_path)
                count+=1
                full_img_path = osp.join(img_dir2, full_img_path)
                full_img = cv2.imread(full_img_path)
                print("find image...")
                for i,img in enumerate(test_data.img_list):
                    if judge(full_img_path,img):
                        img=test_data.pull_img(i)
                        label=test_data.pull_label(i)
                        result=check(net,img,label,transforms)

                        # print(full_img.shape)
                        slot_id=test_data.slot_id[i]
                        tmp=df.loc[lambda df:df['SlotId']==slot_id]
                        tmp=np.array(tmp)
                        if tmp.size==0:
                            print(i)
                            exit()
                        # print(tmp)
                        # exit()
                        _,x,y,w,h=tmp[0]
                        x=int(x*wi_ratio)
                        w=int(w*wi_ratio)
                        y=int(y*hi_ratio)
                        h=int(h*hi_ratio)

                        if result:
                            if label=='1':
                                full_img=cv2.rectangle(full_img,(x,y),(x+w,y+w),color=(0,255,0))#green
                            else:
                                full_img = cv2.rectangle(full_img, (x, y), (x + w, y + w), color=(0,0 , 255))#red
                        else:
                            full_img = cv2.rectangle(full_img, (x, y), (x + w, y + w), color=(255, 0, 0))#blue
                targ_path=osp.join(save_path,cameras+"_"+repr(count)+'.jpg')
                print("save image:",cameras+"_"+repr(count)+'.jpg')
                cv2.imwrite(targ_path,full_img)
                # cv2.imshow("test",full_img)
                # cv2.waitKey(0)


if __name__=="__main__":
    test_img=selfData(args.test_img,args.test_lab,slot_id=True)
    save_path=args.save_path+args.net

    if args.net=="carnet":
        net=carNet()
    elif args.net=="mAlexNet":
        net=mAlexNet()
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weight_path, map_location="cpu").items()})
    if torch.cuda.is_available():
        net.cuda(args.cuda_device)
    for index in range(1,10):
        str="camera"+repr(index)
        print("camera:", index)
        solve(str,net,test_img,save_path)
