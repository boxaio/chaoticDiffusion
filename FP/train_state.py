import flax
import optax
from typing import Any


class TrainState(flax.struct.PyTreeNode):
    step: int
    rng: Any
    optimizer: optax.GradientTransformation=flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    opt_state: optax.OptState

    def apply_grad(self, *, grads, **kwargs):
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step+1, params=new_params, opt_state=new_opt_state, **kwargs)
    
    @classmethod
    def create(cls, *, rng, params, optimizer, **kwargs):
        opt_state = optimizer.init(params)
        return cls(step=0, rng=rng, optimizer=optimizer, params=params, opt_state=opt_state, **kwargs)