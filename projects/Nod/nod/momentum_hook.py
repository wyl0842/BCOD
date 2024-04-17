import torch

from mmengine.hooks import Hook

from mmdet.registry import HOOKS
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from copy import deepcopy


@HOOKS.register_module()
class MomentumUpdateHook(Hook):  
    def __init__(self, momentum=0.999):  
        self.momentum = momentum
        self.ema_model = None

    def before_run(self, runner):  
        if self.ema_model is None:  
            self.ema_model = deepcopy(runner.model)  
            self.ema_model.eval()
      
    def after_train_iter(self, runner):
        with torch.no_grad():
            model = runner.model  
            momentum_model = runner.ema_model  
            for param, momentum_param in zip(model.parameters(), momentum_model.parameters()):  
                momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data

        # 将 EMA 模型存储在 runner 对象中，以便在模型损失计算中引用  
        runner.model.ema_model = self.ema_model