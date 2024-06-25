'''
Hyperparameter class.
This class contains hyperparameters that should not be saved to config files.
'''

from dataclasses import dataclass

@dataclass
class Hypara:
    device: str
    ema_rate: float
    train_batch_size: int
    num_train_epochs: int  
    is_val: bool
    log_freq: int
    snapshot_freq: int

    def should_validate(self, step):
        # Whether to validate after training step "step".
        return step % self.log_freq == 0

    # def should_save(self, step):
    #     # Whether to save after training step "step".
    #     return step % self.save_freq == 0
