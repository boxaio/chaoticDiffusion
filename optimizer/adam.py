import torch
from torch.optim import Adam, Optimizer
from torch.optim import lr_scheduler


class AdamOptim(Optimizer):
    def __init__(self, params, opt_params):
        '''
        opt_params: dictionary containing the following keys:
            lr: learning rate (default: 0.001)
            betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay: weight decay (L2 penalty) (default: 0)
            grad_clip: maximum norm of the gradients (default: None)
            warmup: number of steps to warm up the learning rate (default: 0)
        '''
        self.params = list(params)
        self.opt_params = opt_params
        self.optim = Adam(self.params, **opt_params)

        super(AdamOptim, self).__init__(self.params, defaults={})

        self.state['step'] = 0

    def step(self, closure=None):
        lr = self.opt_params['lr']
        warmup = self.opt_params['warmup'] if 'warmup' in self.opt_params else 0
        grad_clip = self.opt_params['grad_clip'] if 'grad_clip' in self.opt_params else -1
        if warmup > 0:
            for g in self.optim.param_groups:
                g['lr'] = lr * min(1, self.state['step'] / self.opt_params['warmup'])
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=grad_clip)
        
        self.optim.step(closure)
        self.state['step'] += 1


class LinearLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, initial_lr, end_lr):
        self.total_steps = total_steps
        self.end_lr = end_lr
        super(LinearLR, self).__init__(optimizer)
        self.initial_lr = initial_lr

    def get_lr(self):
        current_step = self.last_epoch
        decay_rate = (self.initial_lr - self.end_lr) / self.total_steps
        lr = self.initial_lr - decay_rate * current_step
        return [max(lr, self.end_lr) for base_lr in self.base_lrs]

# # usage
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# scheduler = LinearLR(optimizer, total_steps=100, initial_lr=0.1, end_lr=0.01)



