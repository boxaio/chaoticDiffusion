from dataclasses import dataclass, field
from typing import Any
import torch
import torch.optim.lr_scheduler as lr_scheduler

from optimizer.adam import *
from mlp import MLPxt
from hypara import Hypara

from chaoticDDE import *


@dataclass
class ParamClsSetting:
    cls: Any
    param_dict: dict[str, Any] = field(default_factory=lambda: {})


seed = 6811926
in_dim = 3    # x and t
out_dim = 1   # s(x,t)
num_layer = 4
hidden_dim = [512, 512, 512]
# activations = ['sigmoid', 'sigmoid', 'sigmoid']
activations = ['silu', 'silu', 'silu']
opt = 'adam'
# opt = 'adam_lbfgs'
adam_lr = 2e-4
switch_epochs = 2000
lbfgs_lr = 1
lbfgs_history_size = 100
batch_size = 256
num_train_epochs = 2000
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

is_val = False
is_val = True

# log_freq = None   # set None to log every epoch
log_freq = 2


dict_chaoticSYS_setting = {
    'chaoticDDE_n2_mix_gaussian_slow': ParamClsSetting(
        cls=chaoticDDE,
        param_dict={
            'name': 'chaoticDDE_n2_mix_gaussian_slow',
            'n': 2, 's': 5.4, 'a': 5, 'b': 6.6, 'c': 6.0,   
            'delay': 1.0, 'T': 80.0,
            'Lyaps_dict': {'num_lyaps': 150, 'start_time': 20, 'end_time': 2000,
                           'time_step': 2,
                           'maximum_lyapunov_estimated': 0.995735,
                           },
            'init_dist_name': 'mix_gaussian_2d',
            'init_dist_params': {'seed': 1907698919, 'radius': 2.2, 'num': 5,
                                 'phi': 0, 'scale': 0.1, 
                                 },
            'hist_labels': [5, 39, 59, 99, 499],
        }
    ),
    'chaoticDDE_n2_mix_gaussian_fast': ParamClsSetting(
    cls=chaoticDDE,
    param_dict={
        'name': 'chaoticDDE_n2_mix_gaussian_fast',
        'n': 2, 's': 10.5, 'a': 7.5, 'b': 9.6, 'c': 6.0,   
        'delay': 1.0, 'T': 30.0,
        'Lyaps_dict': {'num_lyaps': 150, 'start_time': 20, 'end_time': 2000,
                        'time_step': 2,
                        'maximum_lyapunov_estimated': 0.995735,
                        },
        'init_dist_name': 'mix_gaussian_2d',
        'init_dist_params': {'seed': 1007698919, 'radius': 2.2, 'num': 5,
                             'phi': 0, 'scale': 0.1, 
                             },
        'hist_labels': [5, 39, 59, 99, 499],
    }
    ),
    'chaoticDDE_n2_olympic_fast': ParamClsSetting(
    cls=chaoticDDE,
    param_dict={
        'name': 'chaoticDDE_n2_olympic_fast',
        'n': 2, 's': 10.5, 'a': 7.5, 'b': 9.6, 'c': 6.0,   
        'delay': 1.0, 'T': 30.0,
        'Lyaps_dict': {'num_lyaps': 150, 'start_time': 20, 'end_time': 2000,
                        'time_step': 2,
                        'maximum_lyapunov_estimated': 0.995735,
                        },
        'init_dist_name': 'olympic_mix',
        'init_dist_params': {'seed': 97609851, 'noise': 0.26},
        'hist_labels': [5, 39, 59, 99, 499],
    }
    ),
    'chaoticDDE_n1_multimodal': ParamClsSetting(
        cls=chaoticDDE,
        param_dict={
            'name': 'chaoticDDE_n1_multimodal', 
            'n': 1, 's': 5.5, 'a': 5, 'b': 6.6, 'c': 6.0,   
            'delay': 1.0, 'T': 25.0,
            'Lyaps_dict': {'num_lyaps': 60, 'start_time': 20, 'end_time': 2000,
                           'time_step': 2,
                           'maximum_lyapunov_estimated': 0.995735,
                           },
            'init_dist_name': 'multimodal',
            'init_dist_params': {'seed': 1897698919, 'locs': [-1.2, 1.1], 'scales': [0.48, 0.32]},
            # 'init_dist': Multimodal(seed=1919, locs=jnp.array([-1.2, 1.1]), scales=jnp.array([0.48, 0.32])),
            'hist_labels': [5, 39, 59, 99, 499],
        }
    ),
    'chaoticDDE_n1_uniform': ParamClsSetting(
        cls=chaoticDDE,
        param_dict={
            'name': 'chaoticDDE_n1_uniform',
            'n': 1, 's': 10.5, 'a': 7.5, 'b': 9.6, 'c': 6.0,   
            'delay': 1.0, 'T': 25.0,
            'Lyaps_dict': {'num_lyaps': 60, 'start_time': 20, 'end_time': 2000,
                           'time_step': 2,
                           'maximum_lyapunov_estimated': 0.995735,
                           },
            'init_dist_name': 'uniform',
            'init_dist_params': {'seed': 1099289072309, 'low': -2.5, 'high': 2.5},
            'hist_labels': [6, 39, 59, 99, 499],
        }
    ),
    'chaoticDDE_varyingdelay_multimodal': ParamClsSetting(
        cls=chaoticDDE,
        param_dict={
            'name': 'chaoticDDE_varyingdelay_multimodal',
            'n': 1, 's': 10.5, 'a': 7.5, 'b': 9.6, 'c': 6.0,   
            'delay': 1.0, 'A': 0.9, 'm': 20.0, 'T': 25.0,
            'Lyaps_dict': {'num_lyaps': 60, 'start_time': 20, 'end_time': 2000,
                           'time_step': 2,
                           'maximum_lyapunov_estimated': 0.995735,
                           },
            'init_dist_name': 'multimodal',
            'init_dist_params': {'seed': 193890809, 'locs': [-1.2, 1.1], 'scales': [0.48, 0.32]},
            # 'init_dist': Multimodal(seed=1939, locs=jnp.array([-1.2, 1.1]), scales=jnp.array([0.48, 0.32])),
            'hist_labels': [6, 39, 59, 99, 499],
        },
    ),
    'chaoticDDE_varyingdelay_uniform': ParamClsSetting(
        cls=chaoticDDE,
        param_dict={
            'name': 'chaoticDDE_varyingdelay_uniform',
            'n': 1, 's': 10.5, 'a': 7.5, 'b': 9.6, 'c': 6.0,   
            'delay': 1.0, 'A': 0.9, 'm': 20.0, 'T': 25.0,
            'Lyaps_dict': {'num_lyaps': 60, 'start_time': 20, 'end_time': 2000,
                           'time_step': 2,
                           'maximum_lyapunov_estimated': 0.995735,
                           },
            'init_dist_name': 'uniform',
            'init_dist_params': {'seed': 1237868090, 'low': -3.0, 'high': 3.0},
            # 'init_dist': Uniform(low=-3.0, high=3.0),
            'hist_labels': [6, 39, 59, 99, 499],
        },
    ),
}


dict_hypara_setting = {
    'ema_rate': None,
    'device': device,
    'train_batch_size': batch_size,
    'num_train_epochs': num_train_epochs,
    'is_val': is_val,
    'log_freq': log_freq,
    'snapshot_freq': 200,
}

dict_model_setting = {
    'mlp_action': ParamClsSetting(
        cls=MLPxt,
        param_dict={
            'in_dim': in_dim,
            'out_dim': out_dim,
            'num_layer': num_layer,
            'hidden_dim': hidden_dim,  # len = num_layer-1
            'activations': activations,
        }
    ),
}

dict_scheduler_setting = {
    # scheduler only for adam/sgd optimizer
    'ConstantLR': ParamClsSetting(
        cls=lr_scheduler.ConstantLR,
        param_dict={
            'factor': adam_lr,
        }
    ),
    'StepLR': ParamClsSetting(
        cls=lr_scheduler.StepLR,
        param_dict={
            'step_size': 500,
            'gamma': 0.5,
        }
    ),
    'MultiStepLR': ParamClsSetting(
        cls=lr_scheduler.MultiStepLR,
        param_dict={
            'milestones': [300, 800],
            'gamma': 0.5,
        }
    ),
    'CosineAnnealingLR': ParamClsSetting(
        cls=lr_scheduler.CosineAnnealingLR,
        param_dict={
            'T_max': 10,      # maximum number of iterations
            'min_lr': 1e-7,   # minimum learning rate
        }
    ),
}

dict_optimizer_setting = {
    'adam': ParamClsSetting(
        cls=None,
        param_dict={
            'lr': adam_lr,
            'weight_decay': 0.0,
            'eps': 1e-8,
        }
    ),
    'adam_lbfgs': ParamClsSetting(
        cls=None,
        param_dict={
            'switch_epochs': switch_epochs,
            'adam_lr': adam_lr,
            'lbfgs_lr': lbfgs_lr,
            'lbfgs_history_size': lbfgs_history_size,
        }
    ),
}

class getConfig:
    @staticmethod
    def get_chaoticSYS_setting(name):
        return dict_chaoticSYS_setting[name]
    
    def get_hypara_param_dict():
        return dict_hypara_setting
    
    @staticmethod
    def get_model_setting(name):
        return dict_model_setting[name]
    
    @staticmethod
    def get_optimizer_setting(name):
        return dict_optimizer_setting[name]
    
    @staticmethod
    def get_scheduler_setting(name):
        return dict_scheduler_setting[name]

