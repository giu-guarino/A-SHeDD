import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
from functions import revgrad
from torch.autograd import Function

############## FROM GITHUB ####################

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

#alpha = torch.tensor([1.])


class FC_Classifier_NoLazy_GRL(torch.nn.Module):
    def __init__(self, input_dim, n_classes, alpha=1.):
        super(FC_Classifier_NoLazy_GRL, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

        self.grl = GradientReversal(alpha=alpha)

    def forward(self, X):
        X_grl = self.grl(X)
        return self.block(X_grl)


class FC_Classifier_NoLazy(torch.nn.Module):
    def __init__(self, input_dim, n_classes):
        super(FC_Classifier_NoLazy, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim,n_classes)
        )
    
    def forward(self, X):
        return self.block(X)


class SHeDD(torch.nn.Module):
    def __init__(self, input_channel_source=4, input_channel_target=2, emb_dim = 256, num_classes=10):
        super(SHeDD, self).__init__()

        source_model = resnet18(weights=None)
        source_model.conv1 = nn.Conv2d(input_channel_source, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.source = nn.Sequential(*list(source_model.children())[:-1])

        target_model = resnet18(weights=None)
        target_model.conv1 = nn.Conv2d(input_channel_target, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.target = nn.Sequential(*list(target_model.children())[:-1])

        self.domain_cl = FC_Classifier_NoLazy(emb_dim, 2)        
        self.task_cl = FC_Classifier_NoLazy(emb_dim, num_classes)        

    
    def forward_source(self, x, source):
        emb = None
        if source == 0:
            emb = self.source(x).squeeze()
        else:
            emb = self.target(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:,0:nfeat//2]
        emb_spec = emb[:,nfeat//2::]
        return emb_inv, emb_spec, self.domain_cl(emb_spec), self.task_cl(emb_inv)
        
    
    def forward(self, x):
        self.source.train()
        self.target.train()
        x_source, x_target = x
        emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl = self.forward_source(x_source, 0)
        emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl =  self.forward_source(x_target, 1)
        return emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl, emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl

    def forward_test_target(self, x):
        self.target.eval()
        emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl =  self.forward_source(x, 1)
        return emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl

    def forward_test_source(self, x):
        self.source.eval()
        emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl = self.forward_source(x, 0)
        return emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl

    def get_target_and_task_weights(self):
        # Return only the parameters, not the names
        target_params = list(self.target.parameters())
        task_cl_params = list(self.task_cl.parameters())
        return target_params, task_cl_params


class TeacherModel(torch.nn.Module):
    def __init__(self, input_channel_source=4, input_channel_target=2, emb_dim=256, num_classes=10):
        super(TeacherModel, self).__init__()

        target_model = resnet18(weights=None)
        target_model.conv1 = nn.Conv2d(input_channel_target, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.target = nn.Sequential(*list(target_model.children())[:-1])
        self.task_cl = FC_Classifier_NoLazy(emb_dim, num_classes)

    def forward(self, x):
        self.target.train()
        emb = self.target(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:, 0:nfeat // 2]
        return self.task_cl(emb_inv)

    def get_weights(self):
        # Return only the parameters, not the names
        target_params = list(self.target.parameters())
        task_cl_params = list(self.task_cl.parameters())
        return target_params, task_cl_params
