from torch.autograd import Variable
import numpy as np
from utils.dataloader import selfData, collate_fn
import torch
import torch.nn as nn
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.carnet import carNet
from model.stn_shuf import stn_shufflenet
from model.trans_shuf import trans_shufflenet
from model.stn_trans_shuf import stn_trans_shufflenet
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from utils.options import args_parser
from ssdaugumentations import *
from tqdm import tqdm
import torch.nn.functional as F
import cv2
args = args_parser()
# transforms=Basetransform(size=args.img_size)
# if args.eval_data=="PKLot":
#     print(args.eval_data)
#     minetransforms = torchvision.transforms.Compose([
#         transforms.ToTensor(),  # normalize to [0, 1]
#         transforms.Resize((args.img_size,args.img_size)),
#         # transforms.ColorJitter(),
#         transforms.Normalize(mean=[0.3708, 0.3936, 0.3976],
#                              std=[0.0152, 0.0115, 0.0250])
#         # transforms.Normalize(mean=[0.3518, 0.4025, 0.4123],
#         #                      std=[0.0744, 0.0760, 0.0752)])
#     ])
# elif args.eval_data=='cnrext':
#     print(args.eval_data)
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
def test(img_path,target_path, net,transforms=minetransforms,device=int(args.cuda_device)):
    print("\nTesting starts now...",type(device),device)

    net.eval()
    test_dataset = selfData(img_path, target_path, transforms)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 16,drop_last= False,pin_memory=True,collate_fn=collate_fn)
    # test_loader=list(test_loader)
    test_size=len(test_dataset)
    data_size=test_size//64
    if test_size%64 !=0:
        data_size+=1
    correct = 0
    total = 0
    item = 1
    test_iter=iter(test_loader)
    with torch.no_grad():
        for i in range(data_size):

            images,labels=next(test_iter)
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            item += 1
    return (correct/total)

if __name__=="__main__":
    if args.model == 'mAlexNet':
        net = mAlexNet()
    elif args.model == 'AlexNet':
        net = AlexNet()
    elif args.model == "carnet":
        net = carNet()
    elif args.model == 'stn_shuf':
        net = stn_shufflenet()
    elif args.model == 'stn_trans_shuf':
        net = stn_trans_shufflenet()
    elif args.model=='shuf':
        net=torchvision.models.shufflenet_v2_x1_0(pretrained=False,num_classes=2)
    elif args.model=='trans_shuf':
        net=trans_shufflenet()
    torch.set_default_tensor_type('torch.FloatTensor')
    print("test net:carNet..")
    # print({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    # exit()
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    if torch.cuda.is_available():
        net.cuda(int(args.cuda_device))
    # exit()
    acc=test(args.test_img,args.test_lab,net)
    print(args.test_lab,acc)