import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, x, target):
        return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))

class L1(nn.Module):
    def __init__(self,eps=1e-6):
        super(L1, self).__init__()
        self.eps = eps
    def forward(self, x, target):
        diff = x - target
        return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))

class Charbonnier_loss(nn.Module):
    def __init__(self,eps=1e-6):
        super(Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss 
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a,b, c * d)
    G = torch.bmm(features, torch.transpose(features,1,2))
    return G.div(b * c * d)

class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()
    def forward(self, x,x1):
        x = gram_matrix(x)
        x1 = gram_matrix(x1)
        loss = F.mse_loss(x, x1)
        return loss

class CCX_loss(nn.Module):
    def __init__(self,eps=1e-6,h=0.5):
        super(CCX_loss, self).__init__()
        self.eps = eps
        self.h = h
    def forward(self, x, y):
        N, C, H, W = x.size()

        y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        x_normalized = x_normalized.reshape(N, C, -1)                               
        y_normalized = y_normalized.reshape(N, C, -1)                               
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)          

        d = 1 - cosine_sim                                  
        d_min, _ = torch.min(d, dim=2, keepdim=True)        

        d_tilde = d / (d_min + self.eps)

        w = torch.exp((1 - d_tilde) / self.h)

        ccx_ij = w / torch.sum(w, dim=2, keepdim=True)       
        ccx = torch.mean(torch.max(ccx_ij, dim=1)[0], dim=1) 
        ccx_loss = torch.mean(-torch.log(ccx + self.eps))
        return ccx_loss

class ReconstructionLoss(nn.Module):
    def __init__(self, device,losstype='l2', eps=1e-6,h=0.5):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        if self.losstype == 'l2':
            self.loss = L2().to(device)
        elif self.losstype == 'texture':
            self.loss = TextureLoss().to(device)    
        elif self.losstype == 'l1':
            self.loss = L1(eps).to(device)
        elif self.losstype == 'charbonnier':
            self.loss = Charbonnier_loss(eps).to(device)
        elif self.losstype == 'ccx':
            self.loss = CCX_loss(eps,h).to(device)
        else:
            raise Exception("reconstruction loss type error!")
    def forward(self, x, target):
        return self.loss(x, target)        

    

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
