import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # ConvBlock 1
        self.block1 = Model.conv_block([3,32,32,32],last_stride=2,dilation=2, padding=2)

        # ConvBlock 2
        self.block2 = Model.conv_block([32,32,64],last_stride=1, dilation=2, padding=2)
        
        # ConvBlock 3
        self.block3 = Model.conv_block([64,64,64],last_stride=1, depthwise=True)
        
        # ConvBlock 4
        self.block4 = Model.conv_block([64,96,110],last_stride=2, depthwise=True)

        # OutBlock 1
        self.out_block1 = nn.Sequential(
            nn.Conv2d(110, 32, 1, bias=False),  # out: 1
            nn.AdaptiveAvgPool2d(1) # GAP
            
        )
        self.fc = nn.Linear(32, 10)
        
    @staticmethod
    def conv_block(channel_arr, last_stride=1, dilation=1, padding=1, depthwise=False):
        modules = []
        for i in range(len(channel_arr)-1):
            if(i == len(channel_arr)-2):
                modules.append(nn.Conv2d(channel_arr[i], channel_arr[i+1], 3, bias=False, stride=last_stride))
            else:
                if(depthwise):
                    # add a pointwise seperable conv to complete depthwise seperable conv
                    modules.append(nn.Conv2d(channel_arr[i], channel_arr[i] , 3, bias=False, padding=1, groups=channel_arr[i]))
                    modules.append(nn.Conv2d(channel_arr[i], channel_arr[i+1], 1, bias=False))
                else:
                    modules.append(nn.Conv2d(channel_arr[i], channel_arr[i+1] , 3, bias=False, padding=padding, dilation=dilation))
            modules.append(nn.BatchNorm2d(channel_arr[i+1]))
            modules.append(nn.ReLU())
        return nn.Sequential(*modules)
    
           
    def forward(self, x):        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out_block1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x) # FC layer
        return F.log_softmax(x, dim=-1)
    
    