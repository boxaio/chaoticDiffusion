import torch
import torch.nn as nn
import numpy as np
from torch.optim import LBFGS, Adam
import random
import re

from .adam import AdamOptim
from .adam_lbfgs import Adam_LBFGS



def get_optimizer(opt_name:str, opt_params,  model_params):
    opt_name = opt_name.lower()
    if opt_name == 'adam':
        return Adam(model_params, **opt_params)
    elif opt_name == 'lbfgs':
        if 'history_size' in opt_params:
            opt_params['history_size'] = int(opt_params['history_size'])
        return LBFGS(model_params, **opt_params, line_search_fn='strong_wolfe')
    elif opt_name == 'adam_lbfgs':
        if "switch_epochs" not in opt_params:
            raise KeyError("switch_epochs not specified for Adam_LBFGS optimizer.")
        switch_epochs = opt_params['switch_epochs']

        # assure switch_epochs is a list of integers
        if not isinstance(switch_epochs, list):
            switch_epochs = [switch_epochs]
        switch_epochs = [int(epoch) for epoch in switch_epochs]

        # Get parameters for Adam and LBFGS, remove the prefix "adam_" and "lbfgs_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS(model_params, switch_epochs, adam_params, lbfgs_params)
    
    else:
        raise ValueError(f'Optimizer {opt_name} not supported')
