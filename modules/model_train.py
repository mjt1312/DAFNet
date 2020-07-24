import os
import scipy.io
import numpy as np
from collections import OrderedDict
from scipy import io
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time
import pdb

import sys
sys.path.insert(0,'./roi_align')
from roi_align.modules.roi_align import RoIAlignAvg,RoIAlignMax

def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.iteritems():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None,K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)
                                    )),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2, dilation=1),
                                    nn.ReLU(),
                                    LRN(),
                                    )),

            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, dilation=3),
                                    nn.ReLU(),
                                    ))]))
        self.layers2 = nn.Sequential(OrderedDict([
            ('fc4', nn.Sequential(
                nn.Linear(512 * 3 * 3, 512),
                nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=1),
                                    nn.ReLU(),
                                    LRN()))]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1),
                                    nn.ReLU(),
                                    LRN()))]))
        self.L = 32
        self.r = 4
        #d = max(int(features / r), self.L)
        #sk1
        self.sk_fc1 = nn.Sequential(OrderedDict([
            ('sk_fc11', nn.Sequential(
                nn.Linear(96, self.L)))]))
        self.sk_fcs1 = nn.Sequential(OrderedDict([
            ('sk_fcs11', nn.Sequential(
                nn.Linear(self.L, 96))),
            ('sk_fcs12', nn.Sequential(
                nn.Linear(self.L, 96)))]))

        #sk2
        self.sk_fc2 = nn.Sequential(OrderedDict([
            ('sk_fc22', nn.Sequential(
                nn.Linear(256, self.L)))]))
        self.sk_fcs2 = nn.Sequential(OrderedDict([
            ('sk_fcs21', nn.Sequential(
                nn.Linear(self.L, 256))),
            ('sk_fcs22', nn.Sequential(
                nn.Linear(self.L, 256))),
            ('sk_fcs23', nn.Sequential(
                nn.Linear(self.L, 256)))]))

        #sk3
        self.sk_fc3 = nn.Sequential(OrderedDict([
            ('sk_fc33', nn.Sequential(
                nn.Linear(512, self.L)))]))
        self.sk_fcs3 = nn.Sequential(OrderedDict([
            ('sk_fcs31', nn.Sequential(
                nn.Linear(self.L, 512))),
            ('sk_fcs32', nn.Sequential(
                nn.Linear(self.L, 512))),
            ('sk_fcs33', nn.Sequential(
                nn.Linear(self.L, 512)))]))



        self.softmax = nn.Softmax(dim=1)

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        self.roi_align_model = RoIAlignMax(3, 3, 1. / 8)

        self.receptive_field = 75.  # it is receptive fieald that a element of feat_map covers. feat_map is bottom layer of ROI_align_layer

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for name, module in self.layers2.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv4.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv5.named_children():
            append_params(self.params, module, name)
        #sk
        for name, module in self.sk_fc1.named_children():
            append_params(self.params, module, name)
        for name, module in self.sk_fcs1.named_children():
            append_params(self.params, module, name)
        for name, module in self.sk_fc2.named_children():
            append_params(self.params, module, name)
        for name, module in self.sk_fcs2.named_children():
            append_params(self.params, module, name)
        for name, module in self.sk_fc3.named_children():
            append_params(self.params, module, name)
        for name, module in self.sk_fcs3.named_children():
            append_params(self.params, module, name)


        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                print "-------------------------------------------------------------"
                print "Warnning!! This train space,but it has unlearnable parammeters"
                print "-------------------------------------------------------------"
                pdb.set_trace()
                p.requires_grad = False


    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x1,x2, k=0, in_layer='conv1', out_layer='fc6'):
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x1 = module(x1)
                x2 = module(x2)
                if name == 'conv1':
                    x1_down = F.max_pool2d(x1, kernel_size=(5, 5), stride=2)
                    x2_down = F.max_pool2d(x2, kernel_size=(5, 5), stride=2)
                    gaoyuan11 = x1_down.unsqueeze(dim=1)
                    gaoyuan12 = x2_down.unsqueeze(dim=1)
                    feas_1 = torch.cat([gaoyuan11,gaoyuan12],dim=1)
                    fea_U1 = torch.sum(feas_1,dim=1)
                    fea_s1 = fea_U1.mean(-1).mean(-1)
                    fea_z1 = self.sk_fc1(fea_s1)
                    for i, fc in enumerate(self.sk_fcs1):
                        vector1 = fc(fea_z1).unsqueeze_(dim=1)
                        if i == 0:
                            attention_vectors1 = vector1
                        else:
                            attention_vectors1 = torch.cat([attention_vectors1, vector1], dim=1)
                    attention_vectors1 = self.softmax(attention_vectors1)
                    attention_vectors1 = attention_vectors1.unsqueeze(-1).unsqueeze(-1)
                    fea_v1 = (feas_1 * attention_vectors1).sum(dim=1)
                    fea_v11 = self.conv4(fea_v1)
                    #fea_v11 = F.max_pool2d(fea_v1, kernel_size=(5, 5), stride=2)
                if name == 'conv2':
                    gaoyuan21 = x1.unsqueeze(dim=1)
                    gaoyuan22 = x2.unsqueeze(dim=1)
                    gaoyuan23 = fea_v11.unsqueeze(dim=1)
                    feas_2 = torch.cat([gaoyuan21,gaoyuan22,gaoyuan23],dim=1)
                    fea_U2 = torch.sum(feas_2,dim=1)
                    fea_s2 = fea_U2.mean(-1).mean(-1)
                    fea_z2 = self.sk_fc2(fea_s2)
                    for i, fc in enumerate(self.sk_fcs2):
                        vector2 = fc(fea_z2).unsqueeze_(dim=1)
                        if i == 0:
                            attention_vectors2 = vector2
                        else:
                            attention_vectors2 = torch.cat([attention_vectors2, vector2], dim=1)
                    attention_vectors2 = self.softmax(attention_vectors2)
                    attention_vectors2 = attention_vectors2.unsqueeze(-1).unsqueeze(-1)
                    fea_v2 = (feas_2 * attention_vectors2).sum(dim=1)
                    #fea_v22 = F.max_pool2d(fea_v2, kernel_size=(5, 5), stride=2)
                    #x11 = F.max_pool2d(x1, kernel_size=(5, 5), stride=2)
                    #x21 = F.max_pool2d(x2, kernel_size=(5, 5), stride=2)
                    #x12 = torch.cat([x11, x21, x1, x2], 1)
                    #pdb.set_trace()
                    #x12 = torch.cat([fea_v11, fea_v2], 1)
                    x12 = F.max_pool2d(fea_v2, kernel_size=(3, 3), stride=1, padding=0, dilation=3)
                    x12 = self.conv5(x12)
                if name == 'conv3':
                    gaoyuan31 = x1.unsqueeze(dim=1)
                    gaoyuan32 = x2.unsqueeze(dim=1)
                    gaoyuan33 = x12.unsqueeze(dim=1)
                    feas_3 = torch.cat([gaoyuan31,gaoyuan32,gaoyuan33],dim=1)
                    fea_U3 = torch.sum(feas_3,dim=1)
                    fea_s3 = fea_U3.mean(-1).mean(-1)
                    fea_z3 = self.sk_fc3(fea_s3)
                    for i, fc in enumerate(self.sk_fcs3):
                        vector3 = fc(fea_z3).unsqueeze_(dim=1)
                        if i == 0:
                            attention_vectors3 = vector3
                        else:
                            attention_vectors3 = torch.cat([attention_vectors3, vector3], dim=1)
                    attention_vectors3 = self.softmax(attention_vectors3)
                    attention_vectors3 = attention_vectors3.unsqueeze(-1).unsqueeze(-1)
                    fea_v3 = (feas_3 * attention_vectors3).sum(dim=1)
                    #x13 = torch.cat([x12, fea_v3], 1)
                    x = fea_v3
                    #x = self.conv5(fea_v3)
                if name == out_layer:
                    return x

        for name, module in self.layers2.named_children():
            if name == in_layer:
                x=x1
                run2 = True
            if run2:
                x = module(x)
                if name == out_layer:
                    return x
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):		
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

    def trainSpatialTransform(self, image, bb):

        return


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]

        loss = (pos_loss.sum() + neg_loss.sum())/(pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):

        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):

        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)

        return prec.data[0]



