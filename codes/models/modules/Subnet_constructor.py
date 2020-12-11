import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class PA(nn.Module):
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out

class PADB(nn.Module):
    def __init__(self, in_channels,out_channels,c_weight=1):
        super(PADB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dc = int(in_channels*c_weight)
        self.c_weight = c_weight
        expand_dim = in_channels
        if c_weight > 1:
           expand_dim = self.dc
           self.expandconv = nn.Conv2d(in_channels, expand_dim, 3,padding=1)
        self.c1_d = nn.Conv2d(expand_dim, self.dc, 1)
        self.c1_r = nn.Conv2d(expand_dim, expand_dim, 3, padding=1)
        self.c2_d = nn.Conv2d(expand_dim, self.dc, 1)
        self.c2_r = nn.Conv2d(expand_dim, expand_dim, 3, padding=1)
        self.c3_d = nn.Conv2d(expand_dim, self.dc, 1)
        self.c3_r = nn.Conv2d(expand_dim, expand_dim, 3, padding=1)
        self.c4 = nn.Conv2d(expand_dim, self.dc, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.c5 = nn.Conv2d(self.dc*4, out_channels, 1)
        self.PA = PA(out_channels)

    def forward(self, input):
        residual = input
        if self.c_weight > 1:
            input = self.act(self.expandconv(input))
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out_cat = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.PA(self.c5(out_cat))
        return out

def subnet(net_structure):
    def constructor(channel_in, channel_out,c_weight=1):
        if net_structure == 'DBNet':
            return PADB(channel_in, channel_out,c_weight=c_weight)
        else:
            return None
    return constructor