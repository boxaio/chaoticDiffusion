'''
This program is mainly adapted from 
https://github.com/lingxiaoli94/SCVM/blob/main/scvm/auto/ec.py

'''

import argparse
import yaml
import copy
from pathlib import Path
import shutil
from collections import namedtuple
import sys
import wandb
import torch


from exp_config import ParamClsSetting, getConfig
from hypara import Hypara
from datasets import *
from optimizer.utils import *


def create_parser():
    return argparse.ArgumentParser(allow_abbrev=False)


class Config:
    def __init__(self, param_dict):
        self.param_dict = copy.copy(param_dict)

    def has_same_params(self, other):
        return self.param_dict == other.param_dict

    def __getitem__(self, k):
        return self.param_dict[k]

    def __contains__(self, k):
        return k in self.param_dict

    def __setitem__(self, k, v):
        self.param_dict[k] = v

    def get(self, k, default_v):
        return self.param_dict.get(k, default_v)

    def __repr__(self):
        return str(self.param_dict)

    @staticmethod
    def from_yaml(yaml_path):
        return Config(yaml.safe_load(open(yaml_path, 'r')))

    def save_yaml(self, yaml_path):
        open(yaml_path, 'w').write(yaml.dump(self.param_dict))


class ConfigBlueprint:
    def __init__(self, default_param_dict):
        '''
        Args:
          default_param_dict:
            A dict, where the values have to be one of [string, int, float].
        '''
        self.default_param_dict = default_param_dict

    def prepare_parser(self, parser):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError(f'Boolean value expected but received {v}.')

        def str2list(s):
            from ast import literal_eval
            return [int(x) for x in s.split(',')]

        for k, v in self.default_param_dict.items():
            if type(v) == bool:
                parser.add_argument('--{}'.format(k), type=str2bool, default=v)
            elif type(v) == list:
                if type(v[0]) == int:
                    parser.add_argument('--{}'.format(k), type=str2list, default=v)
                else:
                    parser.add_argument('--{}'.format(k), type=str2list, default=v)
                    # raise argparse.ArgumentTypeError(f'Only support integer list but {k} has default {v}')
            else:
                parser.add_argument('--{}'.format(k), type=type(v), default=v)


ERParseResult = namedtuple('ERParseResult', ['tmp_dict', 'config', 'exp_dir'])


class ExperimentRegistry:
    def __init__(self, root_dir):
        '''
        We assume the following hierarchy of directories:
          root_dir/exps/exp_name:
            conf.yml:
              The configuration corresponding to an instance of Config class.
          Then arbitrary files and subfolders can be placed here depending
          on the solver.

        This class maintains multiple blueprints that will be combined and
        form an argparser. Hence duplicated keys need to be avoided.

        Args:
            root_dir:
              Root directory of the experiments.
        '''
        self.root_path = Path(root_dir)

        # Temporary blueprints are non-persistent.
        self.temporary_blueprints = []

        # Common blueprints contain common parameters and the ones related
        # to problem.
        self.common_blueprints = [ConfigBlueprint({
            'project': 'my_project',
            'wandb': True,
            'seed': 67029135845,
        })]

    def add_temporary_arguments(self, param_dict):
        self.temporary_blueprints.append(ConfigBlueprint(param_dict))

    def add_common_arguments(self, param_dict):
        self.common_blueprints.append(ConfigBlueprint(param_dict))

    def _parse_single_str(self, key, args, required=False):
        parser = create_parser()
        parser.add_argument('--{}'.format(key), type=str, required=required)
        parsed, rest_args = parser.parse_known_args(args)
        return vars(parsed)[key], rest_args


    def parse_args(self):
        '''
        There are two types of arguments: temporary and persistent.
        Temporary arguments won't be saved (e.g. --num_train_step), while
        persistent arguments (e.g. model architecture) will be saved.

        "exp_name" is the name of the experiment which is the same as the
        folder name containing this experiment's related files. If not
        provided, a random unique name will be generated (which can later
        be changed).
        '''
        parser = create_parser()
        rest_args = sys.argv
        exp_name, rest_args = self._parse_single_str('exp_name', rest_args)

        # Load config if it exists.
        exist_config = False
        if exp_name:
            config_path = self._get_conf_yaml(self._get_exps_path() / exp_name)
            if config_path.exists():
                exist_config = True
                config_dict = Config.from_yaml(config_path).param_dict
                print(f'Found existing config at {config_path}')

        if not exist_config:
            config_dict = {}

        # If override, then skip [Y/N] prompts.
        parser.add_argument('--override', action='store_true', default=True)
        for b in self.temporary_blueprints:
            b.prepare_parser(parser)
        tmp_args, rest_args = parser.parse_known_args(rest_args)
        tmp_dict = vars(tmp_args)
        tmp_dict['exp_name'] = exp_name

        parser = create_parser()
        for b in self.common_blueprints:
            b.prepare_parser(parser)
        common_args, rest_args = parser.parse_known_args(rest_args)
        config_dict.update(vars(common_args))

        # config_dict['device'] = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        #---------------- Handle the chaotic DDE arguments  ----------------
        chaotic_sys, rest_args = self._parse_single_str('chaotic_sys', rest_args)
        if chaotic_sys is None:
            if 'chaotic_sys' not in config_dict:
                raise Exception('If omitting --chaotic_sys, then it must be set in the config.')
            chaotic_sys = config_dict['chaotic_sys']
        config_dict['chaotic_sys'] = chaotic_sys
        chaotic_sys_setting = getConfig.get_chaoticSYS_setting(chaotic_sys)
        chaotic_sys_blueprint = ConfigBlueprint(chaotic_sys_setting.param_dict)

        parser = create_parser()
        chaotic_sys_blueprint.prepare_parser(parser)
        chaotic_sys_args, rest_args = parser.parse_known_args(rest_args)
        config_dict['chaotic_sys_config'] = vars(chaotic_sys_args)

        #---------------- Handle the model arguments ----------------
        model, rest_args = self._parse_single_str('model', rest_args)
        if model is None:
            if 'model' not in config_dict:
                raise Exception('If omitting --model, then it must be set in the config.')
            model = config_dict['model']
        config_dict['model'] = model
        model_setting = getConfig.get_model_setting(model)
        model_blueprint = ConfigBlueprint(model_setting.param_dict)

        parser = create_parser()
        model_blueprint.prepare_parser(parser)
        model_args, rest_args = parser.parse_known_args(rest_args)
        config_dict['model_config'] = vars(model_args)

        #---------------- Handle Hypara ----------------
        parser = create_parser()
        hypara_blueprint = ConfigBlueprint(getConfig.get_hypara_param_dict())
        hypara_blueprint.prepare_parser(parser)
        hypara_args, rest_args = parser.parse_known_args(rest_args)
        tmp_dict['hypara'] = vars(hypara_args)


        # Handle optimizer arguments.
        optimizer, rest_args = self._parse_single_str('optimizer', rest_args)
        if optimizer is None:
            if 'optimizer' not in config_dict:
                raise Exception('If omitting --optimizer, then it must be set in the config.')
            optimizer = config_dict['optimizer']
        config_dict['optimizer'] = optimizer
        optimizer_blueprint = ConfigBlueprint(getConfig.get_optimizer_setting(optimizer).param_dict)

        parser = create_parser()
        optimizer_blueprint.prepare_parser(parser)
        optimizer_args, rest_args = parser.parse_known_args(rest_args)
        config_dict['optimizer_config'] = vars(optimizer_args)

        # Optionally handle scheduler arguments.
        scheduler, rest_args = self._parse_single_str('scheduler', rest_args)
        if scheduler is not None:
            config_dict['scheduler'] = scheduler
            if isinstance(getConfig.get_scheduler_setting(scheduler), ParamClsSetting):
                scheduler_blueprint = ConfigBlueprint(
                    getConfig.get_scheduler_setting(scheduler).param_dict
                )
                parser = create_parser()
                scheduler_blueprint.prepare_parser(parser)
                scheduler_args, rest_args = parser.parse_known_args(rest_args)
                config_dict['scheduler_config'] = vars(scheduler_args)
        
        # reset the experiment name
        exp_name = config_dict['chaotic_sys'] + '_{:}'.format(config_dict['optimizer'])
        tmp_dict['exp_name'] = exp_name

        config = Config(config_dict)
        exp_dir = self._make_persistent(config, exp_name, override=tmp_dict['override'])


        rest_args = rest_args[1:] # first one is always run.py
        if len(rest_args) > 0:
           print('WARNING: Args unprocessed: ', rest_args)

        self.parse_result = ERParseResult(tmp_dict=tmp_dict, config=config, exp_dir=exp_dir)
        return self.parse_result

    def _get_exps_path(self):
        path = self.root_path / 'exps/'
        path.mkdir(exist_ok=True)
        return path

    def _get_conf_yaml(self, exp_dir):
        return Path(exp_dir) / 'conf.yml'
    

    def _make_persistent(self, config, exp_name, override):
        exist = False

        if exp_name is not None:
            exp_dir = self._get_exps_path() / exp_name
            config_path = self._get_conf_yaml(exp_dir)
            if config_path.exists():
                old_config = Config.from_yaml(config_path)
                print(f'Found existing experiment {exp_name}!')
                diff = False
                for k, v in config.param_dict.items():
                    if k not in old_config.param_dict:
                        print(f'Existing config missing {k}!')
                        diff = True
                    elif old_config[k] != v:
                        print(f'Existing config has {k}={old_config[k]}, whereas new config has {k}={v}!')
                        diff = True
                for k in old_config.param_dict:
                    if k not in config.param_dict:
                        print(f'New config missing {k}!')
                        diff = True

                if diff and not override:
                    override = input("Override? [Y/N]")
                if override == True or override == 'Y':
                    print('Removing {}...'.format(exp_dir))
                    shutil.rmtree(exp_dir)
                elif diff:
                    raise Exception('Found config with same name but different parameters! Abort.')
                else:
                    print('Resuming experiment {} with '.format(exp_name) + 'identical config...')
                    exist = True
                    config = old_config

        if not exist:
            # Save config
            if config['wandb']:
                config['wandb_id'] = wandb.util.generate_id()
            if exp_name is None:
                exp_name = config['wandb_id']
            exp_dir = self._get_exps_path() / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            config.save_yaml(self._get_conf_yaml(exp_dir))
            print('Saved a new config to {}.'.format(exp_dir))

        return exp_dir


    def create_flow(self):
        config = self.parse_result.config
        exp_dir = self.parse_result.exp_dir
        tmp_dict = self.parse_result.tmp_dict

        config['exp_dir'] = str(exp_dir)
        config['exp_name'] = tmp_dict['exp_name']

        # wandb.init(
        #     project=config['project'],
        #     mode='online' if config['wandb'] else 'offline',
        #     config={
        #         'exp_dir': exp_dir,
        #         'tmp_dict': tmp_dict,
        #         **config.param_dict,
        #     },
        #     name=('' if tmp_dict['exp_name'] is None
        #           else f'{tmp_dict["exp_name"]}'),
        #     id=config['wandb_id'],
        #     resume='allow'
        # )

        # get the training-data sampler
        # chaossys_setting = getConfig.get_chaoticSYS_setting(config['chaotic_sys'])
        # chaossys = chaossys_setting.cls(chaossys_setting.param_dict)

        # get the model network
        model_setting = getConfig.get_model_setting(config['model'])
        model = model_setting.cls(**config['model_config'])

        model_params = model.parameters()

        # get the optimizer parameters
        # optimizer_setting = getConfig.get_optimizer_setting(config['optimizer'])
        opt_params = config['optimizer_config']

        hypara = Hypara(**tmp_dict['hypara'])

        optim = get_optimizer(opt_name=config['optimizer'], 
                              opt_params=opt_params, 
                              model_params=model_params)
        
        if 'scheduler' in config:
            scheduler_setting = getConfig.get_scheduler_setting(config['scheduler'])
            
            return config, model, optim, scheduler_setting, hypara
        else:
            return config, model, optim, hypara




