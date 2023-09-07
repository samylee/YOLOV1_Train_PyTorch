import torch
import torch.nn as nn
from .ReferenceNet import ReferenceNet


class YOLOV1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOV1, self).__init__()
        self.S, self.B, self.C = S, B, C
        in_channels = 3
        out_channels = 1024
        add_channels = 256
        self.features = ReferenceNet(in_channels=in_channels, out_channels=out_channels)
        self.additional = nn.Sequential(
            nn.Conv2d(out_channels, add_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(add_channels),
            nn.LeakyReLU(0.1)
        )
        self.fc = nn.Linear(S * S * add_channels, S * S * (5 * B + C))

    def forward(self, x):
        x = self.features(x)
        x = self.additional(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x