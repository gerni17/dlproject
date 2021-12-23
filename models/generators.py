from torch import nn
from models.layers import Downsample, Upsample, ResBlock


class CycleGANGenerator(nn.Module):
    def __init__(self, filter=64, in_channels=3, out_channels=3):
        super(CycleGANGenerator, self).__init__()
        self.downsamples = nn.ModuleList(
            [
                Downsample(in_channels, filter, kernel_size=4, apply_instancenorm=False),
                Downsample(filter, filter * 2),
                Downsample(filter * 2, filter * 4),
                Downsample(filter * 4, filter * 8),
                Downsample(filter * 8, filter * 8),
                Downsample(filter * 8, filter * 8),
                Downsample(filter * 8, filter * 8),
            ]
        )

        self.upsamples = nn.ModuleList(
            [
                Upsample(filter * 8, filter * 8),
                Upsample(filter * 16, filter * 8),
                Upsample(filter * 16, filter * 8),
                Upsample(filter * 16, filter * 4, dropout=False),
                Upsample(filter * 8, filter * 2, dropout=False),
                Upsample(filter * 4, filter, dropout=False),
            ]
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        return out

class AttentionGenerator(nn.Module):
    
    def __init__(self, input_nc=3, output_nc=3, n_resblocks=9, norm=False):
        super(AttentionGenerator, self).__init__()
        
        model = [   nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 32, 7),
            nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 32
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_resblocks):
            model += [ResBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(32, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    
    def forward(self, x):
        return self.model(x)
