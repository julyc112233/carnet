import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image
import cv2


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # print(len(batch))
    return torch.utils.data.dataloader.default_collate(batch)

class selfData:
    def __init__(self, img_path, target_path, transforms = None,slot_id=None):
        with open(target_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [os.path.join(img_path, i.split()[0]) for i in lines]
            self.label_list = [i.split()[1] for i in lines]
            if slot_id:
                self.slot_id = list()
                for line in lines:
                    id=line.split()[0]
                    slot=id.split('_')[-1]
                    slot=slot.split('.')[0]
                    self.slot_id.append(int(slot))
            # print(self.slot_id)
            self.transforms = transforms
    
    def __getitem__(self, index):
        try:
            img_path = self.img_list[index]
            # img = jpeg.JPEG(img_path).decode()
            img=cv2.imread(img_path)
            # if img is not None:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            #     img[:, :, 0] = 100
            img = self.transforms(img)
            label = self.label_list[index]
        except:
            return None
        return img, label
    
    def __len__(self):
        return len(self.label_list)

    def pull_img(self,index):
        im=cv2.imread(self.img_list[index],cv2.IMREAD_COLOR)
        # if im is not None:
        #     im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        #     im[:, :, 0] = 100
        # print(self.img_list[index])
        return im
    def pull_label(self,index):
        return self.label_list[index]


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

# class Compose(object):
#     def __init__(self,transforms):
#         self.transforms=transforms
#
#     def __call__(self, img):
#         for t in self.transforms:
#             img=t(img)
#         return img
