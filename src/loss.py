# Adding all loss functions 

import torch 
import torch.nn as nn 
from typing import Optional,Union
from activations import softmax, log_softmax


# # TODO: fix this 
# def cross_entropy(logits:torch.Tensor , targets:torch.Tensor, ignore_idx:Optional[Union[list[int], int]]=None):
#     '''
#     logits: expected shape to be in (N x T x vocab-size)
#     targets: expected shape to be in (N x T x vocab-size) (one hot encoded)
#     ignore_idx : indexes to ignore in loss calculations 

#     so its -PlogQ()

#     Input : 1,2,3,0,0,0
#     Output: 2,3,0,0,0,0

#     '''
#     log_probs = log_softmax(logits, dim=-1)
#     assert logits.shape == log_probs.shape, f"Output shape and logits shape are mismatching, expected to be same found {logits.shape} and {log_probs.shape}"

#     loss_per_token = -(targets * log_probs).sum(dim=-1)
    
#     # 3. Create the Mask
#     # Since targets are one-hot, we find the index of the 'true' class 
#     # using argmax to compare against ignore_idx.
#     target_classes = targets.argmax(dim=-1)
    
#     # Create a mask: 1.0 if we keep it, 0.0 if we ignore it
#     mask = torch.tensor(data = float(target_classes != ignore_idx))
    
#     print('Mask is : ',mask)
#     # 4. Apply Mask
#     masked_loss = loss_per_token * mask
    
#     # 5. Reduction (Mean over VALID tokens only)
#     # We divide by the sum of the mask (number of valid tokens)
#     # Added a small epsilon (1e-8) to prevent division by zero if everything is ignored
#     return masked_loss.sum() / (mask.sum() + 1e-8)


import torch
import torch.nn.functional as F
from typing import Optional, Union

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_idx: Optional[int] = None):
    '''
    logits: (NT, vocab-size)
    targets: (NT,)
    ignore_idx : int index to ignore (e.g., pad_token_id)
    '''
    # 1. Calculate Log Softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    assert logits.shape == log_probs.shape, \
        f"Shape mismatch: logits {logits.shape}, log_probs {log_probs.shape}"

    # 2. Calculate raw NLL per token (N x T)
    loss_per_token = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)

    # 3. Handle Masking
    if ignore_idx is not None:
        # Convert one-hot targets back to indices to check against ignore_idx
        # target_classes = targets.argmax(dim=-1)
        
        # Create mask: (target != ignore_idx)
        # .float() converts True->1.0 and False->0.0
        mask = (targets != ignore_idx).float()
        
        # Apply mask
        masked_loss = loss_per_token * mask
        
        # Divide by number of VALID tokens (sum of mask)
        return -masked_loss.sum() / (mask.sum() + 1e-8)
        
    else:
        # If no ignore_idx, just take the mean of everything
        return -loss_per_token.mean()

if __name__ == "__main__":
    inputs = torch.randn(size=(2,4,1024), dtype = torch.float16)
    labels = softmax(torch.randn_like(inputs), dim = -1)
    # ce = torch.nn.functional.cross_entropy(inputs.permute(0,2,1), labels.permute(0,2,1), ignore_index=0)

    cel = cross_entropy(inputs, labels, ignore_idx=0)
    # print(ce)
    print(cel)

    # assert ce.item() == cel.item()
