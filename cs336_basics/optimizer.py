import torch
import math
from torch.optim import Optimizer
from collections.abc import Iterable

class AdamW(Optimizer):
    def __init__(self, 
                 params,
                 lr: float = 1e-3,
                 weight_delay:float = 0.1, 
                 eps:float = 1e-8,
                 betas = [0.9, 0.99] 
                 ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_delay < 0.0:
            raise ValueError(f"Invalid weight delay: {weight_delay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        
        defaults = dict(lr=lr, weight_delay=weight_delay, eps=eps, betas=betas)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, colsure):
        loss = None

        if colsure is not None:
            torch.enable_grad()
            loss = colsure
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_delay = group['weight_delay']
            eps = group['eps']

            for p in group['params']:
                grad = p.grad

                if p.grad is None:
                    continue
                
                if weight_delay != 0:
                    p = p * (1 - lr * weight_delay)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                state['step'] += 1
                t = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                demon = math.sqrt(exp_avg_sq) + eps

                p.addcdiv_(exp_avg, demon, value=-step_size)
            
           return loss