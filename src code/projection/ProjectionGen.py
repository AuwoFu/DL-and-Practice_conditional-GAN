import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init,utils

from projection.conditional_batchnorm import CategoricalConditionalBatchNorm2d
#from conditional_batchnorm import CategoricalConditionalBatchNorm2d


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(Block, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))




class Generator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self,args, bottom_width=4,distribution='normal'):
        super(Generator, self).__init__()
        self.device = args.device
        self.num_features = args.ngf
        self.dim_z = args.nz
        self.activation = nn.LeakyReLU()
        self.num_class = args.num_class

        self.bottom_width = bottom_width
        self.distribution = distribution

        self.l1 = nn.Linear(self.dim_z, 16 * self.num_features * bottom_width ** 2)

        self.block2 = Block(self.num_features * 16, self.num_features * 8,
                            activation=self.activation, upsample=True,
                            num_classes=self.num_class)
        self.block3 = Block(self.num_features * 8, self.num_features * 4,
                            activation=self.activation, upsample=True,
                            num_classes=self.num_class)
        self.block4 = Block(self.num_features * 4, self.num_features * 2,
                            activation=self.activation, upsample=True,
                            num_classes=self.num_class)
        self.block5 = Block(self.num_features * 2, self.num_features,
                            activation=self.activation, upsample=True,
                            num_classes=self.num_class)
        self.b6 = nn.BatchNorm2d(self.num_features)
        self.conv6 = nn.Conv2d(self.num_features, 3, 1, 1)



    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b6(h))
        return torch.tanh(self.conv6(h))


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # input setting
    parser.add_argument('--nc', default=3, type=int, help='number of image color channel')
    parser.add_argument('--nz', default=100, type=int, help='Size of z latent vector')
    parser.add_argument('--ngf', default=64, type=int, help='Size of feature maps in generator')
    parser.add_argument('--ndf', default=64, type=int, help='Size of feature maps in discriminator')
    parser.add_argument('--label_dim', default=24, type=int, help='number of data label')
    parser.add_argument('--image_size', default=(64,64), type=tuple, help='size of training image after transforms')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=parse_args()
    
    netG=Generator(args)
    print(netG)