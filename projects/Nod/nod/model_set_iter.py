from typing import Optional

from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class SetIterInfoHook(Hook):
    def before_train_iter(self, 
                          runner: Runner,
                          batch_idx: int,
                          data_batch: Optional[dict] = None):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.set_iter(curr_step)
