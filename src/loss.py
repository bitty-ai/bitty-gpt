# Adding all loss functions 

import torch 
import torch.nn as torch 
from typing import Optional,Union
from activations import softmax


# TODO: fix this 
def cross_entropy(logits:torch.Tensor , targets:torch.Tensor, ignore_idx:Optional[Union[list[int], int]]):
    '''
    Ignore value that are equal to these index values ..   
    '''
    targets = targets.to(device = logits.device , dtype =torch.long)
    
    *batch_dim,T,E = logits.shape

    if len(batch_dim) == 0:
        aggregate_idx = 0
    else:
        aggregate_idx = 1

    logits = softmax(logits, dim =-1) # B,T,E
    logits_gathered = logits.gather(dim = -1, index = targets.unsqueeze(1)).squeeze(-1)
    logits_gathered = torch.clamp(logits_gathered, min=1e-9) # clamp for log explode
    log_softmax = -torch.log(logits_gathered).mean(dim = 0) # this does natural log (ln) here NOT the base-2

    return log_softmax
    