import copy

from utils.dataloader import selfData, collate_fn,data_prefetcher
from utils import options
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from utils.test import test
from utils.options import args_parser

args = args_parser()
# learning_rates = np.array([0.7, 0.8, 0.9, 1, 1.1,10,100,1000])*1e-3
learning_rates=np.array([10])*1e-5*args.factor
# learning_rates=np.array([0.2])
# print(learning_rates)
# exit()

def train(epoch, img_path, target_path, transforms, net, criterion):
    batch_size=args.batch_size
    train_dataset = selfData(img_path, target_path, transforms)
    data_size=len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size = batch_size,  num_workers =16,drop_last= False,pin_memory=True,shuffle=True,collate_fn=collate_fn)
    # train_loader=list(train_loader)
    epoch_size=data_size//batch_size
    print(args.cuda_device)
    # exit()
    net = torch.nn.DataParallel(net)
    net=net.cuda()
    if data_size%batch_size !=0:
        epoch_size+=1

    # torch.multiprocessing.set_start_method('spawn')
    best_net=copy.deepcopy(net)
    best_acc=0
    best_ep=0
    tmp_net=copy.deepcopy(net)
    net.train()
    best_lr=0
    for lr in learning_rates:
        net=copy.deepcopy(tmp_net)
        tmp_lr = lr.copy()
        for ep in range(epoch):
            net.train()

            # if ep %args.ep_lr_decay==0 and ep>0:
            #     lr*=0.75
            running_loss = 0.0
            # print("Epoch {}.".format(ep+1))
            prefetcher=data_prefetcher(train_loader)
            # batch_iter=iter(train_loader)
            # sum=0
            for i in range(epoch_size):
                # inputs,labels=next(batch_iter)
                inputs, labels = prefetcher.next()
                labels = list(map(int, labels))
                # sum+=len(labels)
                labels = torch.Tensor(labels)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
                optimizer.zero_grad()
                outputs = net(inputs)
                # print(outputs.shape,labels.shape)
                # exit()
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Epoch {}.\tLoss = {:.3f}.\tlr={}\t....".format(ep + 1, running_loss, tmp_lr))
            if (ep+1) % 2==0:

                accuracy = test(args.test_img, args.test_lab, net)
                running_loss = 0.0
                if accuracy>best_acc:
                    best_net=copy.deepcopy(net)
                    best_acc=accuracy
                    best_ep=ep
                    best_lr=tmp_lr
                # if dist.get_rank()==0:

#                 PATH = './weights/' + args.model  + "_" + repr(ep + 1) + "_" + repr(
#                         tmp_lr) + '.pth'
#                 torch.save(net.state_dict(), PATH)
                print("accuracy: {}...".format(accuracy))
                print("best_ep:{}\tbest_lr:{}\tbest_acc:{}".format(best_ep, best_lr, best_acc))
    # if dist.get_rank() == 0:
    net=best_net
    print("best_ep:{}\tbest_lr:{}\tbest_acc:{}".format(best_ep,best_lr,best_acc))
    print('Finished Training.')
    return best_net
