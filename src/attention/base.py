# base structure for all classes


import torch
import torch.nn as nn 
from abc import ABC , abstractmethod
import random

torch.manual_seed(42)
random.seed(42)


class BaseAttention(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x:torch.Tensor,):
        pass

    
    # test out attention from shapes 
    def _test(self,input:torch.Tensor):
        B,T,C = input.shape
        attn_output = self.forward(input)
        assert input.shape == attn_output.shape , "Attention output shapes are mismatch"
        
        return True