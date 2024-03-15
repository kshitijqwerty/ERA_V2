import torch
from torch import nn
from torch.nn import functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        # ConvBlock 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, bias=False),  # out: 26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 22
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # TransBlock 1
        self.trans_block1 = nn.Sequential(
            nn.MaxPool2d(2, 2), # out: 11
        )

        # ConvBlock 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, bias=False),  # out: 9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # OutBlock 1
        self.out_block1 = nn.Sequential(
            nn.Conv2d(16, 10, 1, bias=False),  # out: 5
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 5, bias=False),  # out: 1
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.trans_block1(x)
        x = self.block2(x)
        x = self.out_block1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.dropout_value = 0.05

        # ConvBlock 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False),  # out: 26
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(8, 9, 3, bias=False),  # out: 24
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(9, 10, 3, bias=False),  # out: 22
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value),
        )

        # TransBlock 1
        self.trans_block1 = nn.Sequential(
            nn.MaxPool2d(2, 2), # out: 11
        )

        # ConvBlock 2
        self.block2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, bias=False),  # out: 9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value),
        )

        # OutBlock 1
        self.out_block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5), # out: 1
            nn.Conv2d(16, 10, 1, bias=False),  # out: 1
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.trans_block1(x)
        x = self.block2(x)
        x = self.out_block1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.dropout_value = 0.05

        # ConvBlock 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False),  # out: 26
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(8, 9, 3, bias=False),  # out: 24
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Dropout(self.dropout_value),
            
        )

        # TransBlock 1
        self.trans_block1 = nn.Sequential(
            nn.MaxPool2d(2, 2), # out: 12
        )

        # ConvBlock 2
        self.block2 = nn.Sequential(
            nn.Conv2d(9, 10, 3, bias=False),  # out: 10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(10, 16, 3, bias=False),  # out: 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(16, 16, 3, bias=False),  # out: 4
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value),
        )

        # OutBlock 1
        self.out_block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4), # out: 1
            nn.Conv2d(16, 10, 1, bias=False),  # out: 1
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.trans_block1(x)
        x = self.block2(x)
        x = self.out_block1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)