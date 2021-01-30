'''
Shared functions/constructs are here.
'''
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

def pretrain_optimizer(parameters, momentum, weight_decay, lr, lars=True):
    '''
        If using the LARS optimizer, cosine decay schedule is used. 
        This is a simpler scheduler, drops to 10% of learning rate at epochs
        120 and 160.
        Called by configure_optimizers from all pretraining methods (coco pretraining, cifar10 pretraining and jeopardy)
    '''
    
    # look at exclude bn/bias terms from weight decay stuff
    optimizer = SGD(parameters, momentum=momentum, weight_decay=weight_decay, lr=lr)
    if lars:
        # check these numbers
        optimizer = LARSWrapper(optimizer, eta=0.001, clip=False)
        return optimizer
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[120,160], gamma=0.1)
        return [optimizer], [scheduler]
    
def pretrain_scheduler(learning_rate, train_iters_per_epoch, num_epochs, scheduler_config):
    # define LR schedule
    # returns evenly spaced numbers in the interval
    # scheduler_config would contain learning rate, batch_size, start_lr, final_lr, warmup_epochs
    start_lr = scheduler_config.start_lr
    final_lr = scheduler_config.final_lr
    warmup_epochs = scheduler_config.warmup_epochs

    warmup_lr_schedule = np.linspace(
        start_lr, learning_rate, train_iters_per_epoch * warmup_epochs
    )
    iters = np.arange(train_iters_per_epoch * (num_epochs - warmup_epochs))
    cosine_lr_schedule = np.array([final_lr + 0.5 * (learning_rate - final_lr) * (
        1 + math.cos(math.pi * t / (train_iters_per_epoch * (num_epochs - warmup_epochs)))
    ) for t in iters])

    return np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

class Projection(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # two layer MLP projection head
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

"""
References:
    - https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
"""
import torch
from torch.optim import Optimizer


class LARSWrapper(object):
    """
    Wrapper that adds LARS scheduling to any optimizer. This helps stability with huge batch sizes.
    """

    def __init__(self, optimizer, eta=0.02, clip=True, eps=1e-8):
        """
        Args:
            optimizer: torch optimizer
            eta: LARS coefficient (trust)
            clip: True to clip LR
            eps: adaptive_lr stability coefficient
        """
        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group
        self.__setstate__ = self.optim.__setstate__
        self.__getstate__ = self.optim.__getstate__
        self.__repr__ = self.optim.__repr__

    @property
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get('weight_decay', 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group['weight_decay'] = 0

            # update the parameters
            [self.update_p(p, group, weight_decay) for p in group['params'] if p.grad is not None]

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group['lr'], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr

import pickle
def get_pretrained_emb_layer():
    # load from pickle file and return a FloatTensor
    # how many words in my lookup table/ how large is the vocabulary?
    file_loc = os.environ.get('GLOVE_LOC')
    with open(f'{file_loc}', 'rb') as f:
        # arr = pickle.load(f)
        arr = torch.load(f) # remove this in ec2 - , map_location={'cuda:1':'cuda:4'}
        return arr