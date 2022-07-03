import math
from xml.sax import handler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init,utils


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


class Discriminator_Projection(nn.Module):
    def __init__(self, args):
        super(Discriminator_Projection, self).__init__()
        self.num_features = args.ndf
        self.num_classes = args.num_class # 24
        self.label_dim=args.label_dim #3
        
        self.activation = nn.LeakyReLU()

        self.block1 = OptimizedBlock(3, self.num_features)
        self.block2 = Block(self.num_features, self.num_features * 2,
                            activation=self.activation, downsample=True)
        self.block3 = Block(self.num_features * 2, self.num_features * 4,
                            activation=self.activation, downsample=True)
        self.block4 = Block(self.num_features * 4, self.num_features * 8,
                            activation=self.activation, downsample=True)
        self.block5 = Block(self.num_features * 8, self.num_features * 16,
                            activation=self.activation, downsample=True)
        
        self.l6 = utils.spectral_norm(nn.Linear(self.num_features * 16, 1))
        
        
        if self.num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(self.num_classes+1, self.num_features * 16,padding_idx=self.num_classes))
            self.linear_y=nn.Linear(self.num_features * 16*self.label_dim,self.num_features * 16) 

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)
    
    def forward(self, x, y=None):
        h = x
        for i in range(1, 6):
            h = getattr(self, f'block{i}')(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        
        if y is not None:
            y=self.l_y(y) # [bs,3,ndf*16]
            ly=self.linear_y(y.view(-1,self.num_features * 16*self.label_dim)) #[bs,ndf*16*3] to [bs,ndf*16]
            output += torch.sum(ly*h, dim=1,keepdim=True) # inner product and sum
        return output


