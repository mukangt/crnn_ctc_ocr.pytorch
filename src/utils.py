'''
@Author: Tao Hang
@Date: 2019-10-17 14:43:52
@LastEditors  : Tao Hang
@LastEditTime : 2019-12-19 02:16:10
@Description: 
'''
import torch.nn as nn
import torch
from torch.autograd import Variable


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


def load_data(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def get_alphabet(alpha_path):
    with open(alpha_path, 'r', encoding='utf8') as fp:
        data = fp.readlines()
        alphabet = ''.join([c.strip('\n') for c in data[1:]] + ['卍'])
    return alphabet


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def val(self):
        res = 0.0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)

        return res

    def reset(self):
        self.n_count = 0
        self.sum = 0