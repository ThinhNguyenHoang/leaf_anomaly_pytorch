import math
import torch
from torch import nn
from custom_models import *
import torch.nn.functional as F
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import os
from torchvision import transforms
from custom_models.u2net import u2net_test

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P
# ================== SALIENCY DETECTOR =========================
def load_saliency_detector_arch(c):
    # u2net_dir = os.path.join(os.getcwd(),'custom_models' ,'u2net')
    u2net_weight_path = c.u2net_weight_path
    if 'saliency' in c.sub_arch and not u2net_weight_path:
        raise RuntimeError('Cannot use saliency without path for saved weights of resnet model')
    return u2net_test.load_u2net_eval(u2net_weight_path)

def binarize_map(saliency_map, threshold=0.5):
    return (saliency_map>threshold)*1.0
def get_saliency_map(detector, input_img):
    pred = u2net_test.eval_with_u2net(detector, input_img)
    return binarize_map(pred)
    # return pred
# ================== DECODER ======================
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=False)
    return coder


def load_decoder_arch(c, dim_in):
    if   c.dec_arch == 'freia-flow':
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == 'freia-cflow':
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.dec_arch))
    #print(decoder)
    return decoder

# ==================== ENCODER ========================
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def load_encoder_arch(c, L):
    # encoder pretrained on natural images:
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer'+str(i) for i in range(L)]
    if 'resnet' in c.enc_arch:
        if   c.enc_arch == 'resnet18':
            encoder = resnet18(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet34':
            encoder = resnet34(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet50':
            encoder = resnet50(pretrained=True, progress=True)
        elif c.enc_arch == 'resnext50_32x4d':
            encoder = resnext50_32x4d(pretrained=True, progress=True)
        elif c.enc_arch == 'wide_resnet50_2':
            if c.gcp:
                # Load model state_dict directly
                resnet_w_50_2_dir = c.wide_resnet50_weight_path
                if not os.path.exists(c.wide_resnet50_weight_path):
                    raise RuntimeError('Weight file for wide resnet 50 is not defined')
                encoder = wide_resnet50_2(pretrained=True, progress=True, model_prepared_path=resnet_w_50_2_dir)
            else:
                encoder = wide_resnet50_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        # L = 4 => first branch, L == 2
        if L >= 3:
            # Record the layers output into global dictionary activation => To feed the multiscale feature map into NF decoders
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
    #
    return encoder, pool_layers, pool_dims


class Wide(nn.Module):
    def __init__(self, input_dims, num_class=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class),
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ClassificationHead(nn.Module):
    # input_dims: 2D anomaly score_map
    def __init__(self, input_dims, num_class=2):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(input_dims[0] * input_dims[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torch.utils.data import Dataset
class ScorerDataset(Dataset):
    def __init__(self, super_mask_list, label_list):
        self.x = super_mask_list
        self.y = label_list

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

import torch.optim as optim
def train_class_head(c, class_head, super_mask_list, label_list, start_lr=0.001):
    train_loader = torch.utils.data.DataLoader(ScorerDataset(super_mask_list, label_list), batch_size=c.batch_size, shuffle=True, drop_last=True)
    class_weights = weights = torch.FloatTensor([1.2, 1.2 * c.anomaly_weight]) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print('========= ClassHead Training =========')
    optimizer = optim.SGD(class_head.parameters(), lr=start_lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(c.class_head_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = class_head(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if c.verbose and (i % 10 == 0):
                print(f'[Epoch: {epoch}] loss: {running_loss / 10 :.3f}')
                running_loss = 0.0

    print('========= Done! ClassHead Training =========')