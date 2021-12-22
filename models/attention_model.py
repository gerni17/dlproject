import torch
from torch import nn


class AttentionNet(nn.Module):
    """
    Implemenation according to paper:
        `Unsupervised Attention-guided Image-to-Image Translation`
    """
    def __init__(
        self,
    ):
        super(AttentionNet, self).__init__()
        relu = nn.ReLU(inplace=True)
        sigmoid = nn.Sigmoid()

        #c7s1-32-R
        conv7_1 = nn.Conv2d(3, 32, 7, stride=1, padding=3)
        norm32_1 = nn.InstanceNorm2d(32)
        self.block_1 = nn.Sequential(conv7_1, norm32_1, relu)

        #c3s2-64-R
        conv3_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        norm64_2 = nn.InstanceNorm2d(64)
        self.block_2 = nn.Sequential(conv3_2, norm64_2, relu)

        #r64
        #“rk” denotes a residual block formed by two 3×3 convolutions with k filters, stride 1 and a ReLU activation
        conv3_3 = nn.Conv2d(64, 64, 3, stride=1)
        self.res_block = nn.Sequential(nn.ReflectionPad2d(1), conv3_3, norm64_2, relu, 
                        nn.ReflectionPad2d(1), conv3_3, norm64_2, relu)
        
        # use nearest-neighbor upsampling layers “up2” that doubles the height and width of its input
        self.up2 = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None) 

        #c3s1-64-R
        conv3_4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.block_4 = nn.Sequential(conv3_4, norm64_2, relu)

        #up2 (doesnt make sense to do it)

        #c3s1-32-R
        conv3_5 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.block_5 = nn.Sequential(conv3_5, norm32_1, relu)

        #c7s1-1-S
        conv7_6 = nn.Conv2d(32, 1, 7, stride=1, padding=3)
        self.block_6 = nn.Sequential(conv7_6, sigmoid)

    def forward(self, x, shortcut=None):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.res_block(x) + x
        x= self.up2(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        return x