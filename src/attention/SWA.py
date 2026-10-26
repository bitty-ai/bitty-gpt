# Sliding window attention, we slide attention , this restricts each token to a fixed token lookup 
# local : global ratio , this tells how many local layers we have for each global layer 
# Naively : this uses the same attention format as MHA version we are saving nothing here, 
# Real Save: the memory save comes from using flash attention variants that is using tiled GPU compute over this
# Occasional global memory add as well that done as decoder layer initialization  

import torch 
import torch.nn as nn 
from base import BaseAttention
from dataclasses import dataclass , asdict
import random 

torch.manual_seed(42)
random.seed(42)

class SWA(BaseAttention):
    def __init__(self, *, num_heads:int, hidden_size:int, embedding_dim:int, ctx_lookup:int, dtype= torch.float16, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.ctx_lookup = ctx_lookup 
        assert self.hidden_size % self.num_heads == 0, f"Embedding dim is not divisible by heads, {self.embedding_dim} % {self.num_heads}"
        
        self.head_dim = int(self.hidden_size / self.num_heads)
        
        self.q_proj = nn.Linear(embedding_dim, hidden_size , bias = False ,dtype =dtype)
        self.k_proj = nn.Linear(embedding_dim , hidden_size, bias = False,dtype =dtype)
        self.v_proj = nn.Linear(embedding_dim , hidden_size, bias = False,dtype =dtype)
        self.out_proj = nn.Linear(hidden_size , embedding_dim, bias = False,dtype =dtype)
        
    def forward(self, input:torch.Tensor):
        B,T,C = input.shape
 
        q = self.q_proj(input).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(input).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(input).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        attn_score = (q @ k.transpose(2,3)) * (self.head_dim**(-0.5)) # B,N,T,T 
        
        attn_mask = torch.tril(torch.ones_like(attn_score), diagonal=0) > 0.0 # dont leak future 
        sliding_mask = torch.tril(torch.ones_like(attn_score) , diagonal=-self.ctx_lookup) > 0.0 # masked area is 1 here
        attn_mask = attn_mask & (~sliding_mask) # sliding window
        masked_score = attn_score.masked_fill(attn_mask==0,  -float('inf'))
        
        attn_softmaxed = torch.softmax(masked_score ,dim=-1)
        row_sums = attn_softmaxed.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-3,
            rtol=1e-3,
        )
        
        print("Softmax test passed") 
        
        output = (attn_softmaxed @ v).transpose(1,2).contiguous().view(B,T,self.hidden_size) # b,n,t,h
        return self.out_proj(output).view(B,T,C)

if __name__ == "__main__":
    @dataclass
    class Config:
        batch_size:int = 1
        ctx_length:int = 100
        ctx_lookup:int = 50 # sliding window
        num_heads:int = 8
        hidden_size:int = 1024
        embedding_dim:int = 512 # embedding table output
            
    cfg = Config()
    cfg_dict = asdict(cfg)
    dtype = torch.float16
    mha = SWA(**cfg_dict, dtype = dtype)

    B,T,C = cfg_dict.get("batch_size"), cfg_dict.get("ctx_length"), cfg_dict.get("embedding_dim") 
    input_vec = torch.randn((B,T,C), dtype = dtype)
    
    out = mha(input_vec)
    passed = mha._test(input_vec)
    print(out.shape)
    print("Test Passed ? ", passed)
    
    # Test-1
    # B, T, C = 1, 3, 4
    # num_heads = 2
    # hidden_size = 4
    # embedding_dim = 4

    # mha = SWA(num_heads=num_heads, hidden_size=hidden_size, embedding_dim=embedding_dim)
    # input_matrix = torch.tensor(data = [[[10, 20, 30, 40] , [20, 30, 40, 50] , [30, 40, 50, 70]]], dtype = dtype) # 1x3x4 , BTC , (123)
    # out = mha(input_matrix)
    # print(out.shape)
    # print("Results are : ", out)
 
 
    # # Test-2 
    # x = input_matrix
    # flat = mha.q_proj(x)
    # print(mha.head_dim)

    # heads = flat.view(
    #     B, T,mha.num_heads, mha.head_dim
    # ).transpose(1,2)

    # recovered = (
    #     heads.transpose(1, 2) # b,nh,t,d
    #     .contiguous()
    #     .view(B, T, mha.hidden_size)
    # )

    # assert torch.equal(flat, recovered)
    # print("Head layout test passed")

