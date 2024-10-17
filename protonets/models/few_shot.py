import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
# from sklearn.metrics import confusion_matrix
import numpy as np
from protonets.models import register_model

from .utils import euclidean_dist, cosine_similarity, kl_divergence, JSD, jensen_shannon_divergence
from .convnext import convnextT, convnextS, convnextB, convnextL, convnextXL, convnextT_pretrained
from .resnet import resnet_18, resnet34, resnet50, resnet101, resnet152, resnet18_pretrained, ResNet12
from .teacher import load_task

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder
        
    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query
        # print(sample['class'])
        # print(xs.shape)

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        
        # for sam in loader:
        #     data = sam[0]

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        
        if xq.is_cuda:
            target_inds = target_inds.cuda()
        
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)
        
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]
        
        # Euclidean dist
        dists = euclidean_dist(zq, z_proto)
        
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        if self.encoder.training:
            loader = load_task(sample['class'])
            for sam in loader:
                data = sam[0]
                n = torch.sum(sam[1] == 0).item()
                # print(sam[1])
            with torch.no_grad():
                z_teacher = self.encoder.forward(data.cuda())
            z_teacher_dim = z_teacher.size(-1)
            
            teacher_proto = z_teacher.view(n_class, n, z_teacher_dim).mean(1)
            # torch.save(z_teacher, 'z_teacher.pt')
            # torch.save(z[:n_class*n_support], 'z')
            # torch.save(teacher_proto, 'teacher_proto.pt')
            # torch.save(z_proto, 'z_proto.pt')
            teacher_results = euclidean_dist(zq, teacher_proto)
            teacher_log = F.log_softmax(-teacher_results, dim=1).view(n_class, n_query, -1)
            

        #     # Based on prototypes
            # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            # teacher_proto_log = F.log_softmax(-teacher_proto, dim=1)
            # z_proto_log = F.log_softmax(-z_proto, dim=1)
            # lsdk = kl_loss(teacher_proto_log, z_proto_log)

        #     # Based on logits
            # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            # lsdk = kl_loss(teacher_log, log_p_y)

        #     # Based on predictions
            lsdk = -teacher_log.gather(2, target_inds).squeeze().view(-1).mean()

        #     # Total loss
            loss_val = lsdk - 0.5 * log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        else:
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        # # cf = confusion_matrix(target_inds.squeeze(), y_hat)
        # # print(cf)
        
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )
    
    return Protonet(encoder)


@register_model('resnet')
def load_resnet(**kwargs):

    x_dim = kwargs['x_dim'] 

    encoder = nn.Sequential(
        ResNet12(),
        Flatten()
    )

    return Protonet(encoder)


@register_model('convnext')
def load_convnext(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']

    encoder = nn.Sequential(
        convnextT_pretrained(),
        Flatten()
    )
        

    return Protonet(encoder)