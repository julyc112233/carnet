import torch
import torch.nn as nn

class carNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(carNet, self).__init__()
        self.feature=nn.Sequential(nn.Conv2d(3,96,kernel_size=11,dilation=2,padding=10),#[64, 96, 54, 54]
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,stride=2),
                                   nn.Conv2d(96,192,kernel_size=11,dilation=2,padding=10),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,stride=2),
                                   nn.Conv2d(192,384,kernel_size=11,dilation=2,padding=10),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,stride=2)
                                   )

        self.classifier=nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(6*6*384,4096),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(4096,4096),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(4096,2))
    def forward(self,x):
        x=self.feature(x)
        # print(x.shape)
        # exit()
        x = torch.flatten(x,1)
        x=self.classifier(x)
        m = nn.Softmax(dim=1)
        x = m(x)
        return x