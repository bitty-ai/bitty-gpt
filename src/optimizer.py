#building / testing optimizer from scratch in torch 


import torch 
import torch.nn as nn 
from collections.abc import Callable, Iterable
from torch.optim import Optimizer
from typing import Optional
import math
from functools import partial


class SGD(Optimizer):
    def __init__(self,params:torch.nn.parameter.Parameter, lr : float):
        if lr<0:
            raise ValueError(f'Invalid learning rate {lr}')
        
        defaults = {'lr':lr}
        super().__init__(params, defaults)
        # p.grad and store the updated value in the p.data field
         

    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure() # what is this closure ? This runs a full step , that includes zero grad , forward pass , backward pass and outputs loss value 

        for group in self.param_groups:
            lr = group['lr'] # get the learning rate 
            for p in group['params']:
                # print(dir(p))
                if p.grad is None:
                    continue

                state = self.state[p] # Get the state associated with the parameter p 
                t=state.get('t', 0) # get iteration number from the state or initial value (this if not there we will create one for our use case !)

                grad = p.grad.data # get grad of loss with respect to p 
                p.data -= lr / math.sqrt(t+1)*grad # update weight tensor in place 
                state['t'] = t+1

        return loss


class Adam(Optimizer):
    def __init__(self, params , lr = 1e-4, betas:tuple[float,float] = (0.9, 0.95) , eps = 1e-8):
        defaults = {
            'betas':betas, 
            'eps':eps,
            'lr':lr
        }
        
        super().__init__(params, defaults)
    
    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure() # this helps in finding the loss

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    pass

                state = self.state[p]
                t = state.get('t', 1)

                grad = p.grad.data # detached tensor wont affect the computation graph 

                first_moment = state.get('first_moment', torch.zeros_like(p.grad.data))
                second_moment = state.get('second_moment', torch.zeros_like(p.grad.data))

                b1,b2 = betas

                first_moment = b1 * first_moment + (1-b1)*grad
                second_moment = b2 * second_moment + (1-b2)*grad*grad
                
                m_cap = first_moment / (1-(b1**t))
                v_cap = second_moment / (1-(b2**t))

                p.data -= lr * m_cap / ((v_cap**0.5) + eps)

                state['first_moment'] = first_moment
                state['second_moment'] = second_moment

        return loss


class AdamW(Optimizer):
    '''
    Param Groups : List[dict], these are grouped in a way that these share same values that are in the param-group and each dict has the values that you define in the init method 

    State : dict, role is persistent memory history , each param has this dictionary with him keeps 

    p.grad : same value as p.data and keeps grad of that values 

    p.grad.data : raw storage for the grad  
    '''
    def __init__(self, params:torch.nn.parameter.Parameter, lr:float, betas:tuple[float,float]= (0.9, 0.95), eps:float = 1e-8, weight_decay = 0.01, lr_scheduler=None , min_lr:float=None):
        # In Optimizer class we have state, that maps , dict(Parameter, dict[str, int])
        defaults = {'lr':lr , 'betas':betas, 'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params, defaults)
        self.lr_scheduler = partial(lr_scheduler , max_lr=lr , min_lr =min_lr, warm_up_steps = 20 ,cosine_annealing_steps = 2000) if lr_scheduler else None

    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            decay_rate = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] #state dict is present there

                t = state.get('t', 1)
                if self.lr_scheduler:
                    lr = self.lr_scheduler(t)

                grad = p.grad.data # p is a tensor that has initialised values for iteration1 

                # print('grad value is : ', grad)
                # p.data -= lr * decay_rate * p.data
                # p.data.mul_(1 - lr * decay_rate)

                moment1 = state.get('moment1', torch.zeros_like(p.grad.data))
                moment2 = state.get('moment2', torch.zeros_like(p.grad.data)) # initialise with zeros

                b1,b2 = betas
                
                moment1 = b1 * moment1 + (1-b1)*grad
                moment2 = b2 * moment2 + (1-b2)*grad*grad

                alpha_t = lr * math.sqrt(1-(b2)**t) / (1-(b1**t))

                p.data -= alpha_t * (moment1 / (moment2.sqrt() + eps))

                p.data -= lr * decay_rate * p.data

                state['t'] = t+1
                state['moment1'] = moment1
                state['moment2'] = moment2

        return loss # this can be none


def cosine_learning_rate_scheduler(t:int, max_lr:float, min_lr:float, warm_up_steps:int, cosine_annealing_steps:int):
    
    # warm up 
    if t < warm_up_steps: # so this will peak at t = warm-up-steps
        return t * max_lr / warm_up_steps   

    # cosine annealing steps 
    elif t >= warm_up_steps and t<= cosine_annealing_steps:
        x = 1 + math.cos((t-warm_up_steps)/(cosine_annealing_steps - warm_up_steps) * math.pi)

        return min_lr + 0.5 * x * (max_lr - min_lr)

    # post annealing
    else:
        return min_lr


def gradient_clipping(params, max_norm: float):
    """
    Clips gradient norm of an iterable of parameters.
    
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    
    Args:
        params: (Iterable[torch.Tensor] or torch.Tensor): an iterable of Tensors or a 
                single Tensor that will have gradients normalized.
        max_norm (float or int): max norm of the gradients.
    """
    # 1. Handle single tensor input vs iterable
    if isinstance(params, torch.Tensor):
        params = [params]
    
    # Filter parameters that have gradients
    params = [p for p in params if p.grad is not None]
    
    if len(params) == 0:
        return 0.0

    # 2. Calculate the global L2 norm of the gradients
    # We sum the squares of the norms of each parameter's gradient
    # total_norm = torch.norm(
    #     torch.stack([torch.norm(p.grad.detach(), 2.0) for p in params]), 2.0
    # )

    # print(total_norm, type(total_norm))
    
    # Alternatively, manual calculation without creating stack (more memory efficient):
    total_norm = torch.tensor(math.sqrt(sum(p.grad.detach().norm(2)**2 for p in params)))

    # 3. Compute the clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # 4. Clip gradients if necessary (if total_norm > max_norm)
    # The 'clamp' ensures we only scale down, never up (if clip_coef > 1, we use 1.0)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    for p in params:
        p.grad.detach().mul_(clip_coef_clamped)

    # Usually helpful to return the total norm calculated
    return total_norm


if __name__ == '__main__':

    import torch 
    import torch.nn as nn 
    import torch.nn.functional as F

    class _TestNet(nn.Module):
        def __init__(self, d_input: int = 100, d_output: int = 10):
            super().__init__()
            self.fc1 = nn.Linear(d_input, 200)
            self.fc2 = nn.Linear(200, 100)
            self.fc3 = nn.Linear(100, d_output)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 100
    model = _TestNet(d_input=d_input, d_output=d_output)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    it = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        print('loss is : ', loss.detach())
        loss.backward()
        optimizer.step()
        it += 1



