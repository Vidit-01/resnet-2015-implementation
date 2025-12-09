import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, stride=stride)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.projection:
            identity = self.projection(identity)
            identity = self.bn3(identity)
        out = F.relu(out + identity)
        return out

class StackBlock(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks,stride=2):
        super().__init__()
        layers = []
        layers.append(ResidualBlock(in_channels,out_channels,stride=stride))
        for i in range(num_blocks-1):
            layers.append(ResidualBlock(out_channels,out_channels))
        self.stack = nn.Sequential(*layers)
    def forward(self,x):
        return self.stack(x)

class ResNet(nn.Module):
    def __init__(self,num_classes=10,blocks=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.stack_1 = StackBlock(16,16,blocks,stride=1)
        self.stack_2 = StackBlock(16,32,blocks)
        self.stack_3 = StackBlock(32,64,blocks)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stack_1(x)
        x = self.stack_2(x)
        x = self.stack_3(x)
        # print(x.shape)
        x = self.gap(x)
        # print(x.shape)
        x = torch.flatten(x,1)
        # print(x.shape)
        x = self.fc(x)
        return x