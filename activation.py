import torch.nn as nn

from typing import Any



class ActivationFactory:
    @staticmethod
    def create(name):
        if name.lower() == 'relu':
            fn = nn.ReLU()
        elif name.lower() == 'tanh':
            fn = nn.Tanh()
        elif name.lower() == 'celu':
            fn = nn.CELU()
        elif name.lower() == 'gelu':
            fn = nn.GELU()
        elif name.lower() == 'elu':
            fn = nn.ELU()
        elif name.lower() == 'silu':
            fn = nn.SiLU()
        elif name.lower() == 'softplus':
            fn = nn.Softplus()
        elif name.lower() == 'prelu':
            fn = nn.PReLU()
        elif name.lower() == 'sigmoid':
            fn = nn.Sigmoid()
        else:
            raise Exception(f'Unknown activation name: {name}')
        return fn
