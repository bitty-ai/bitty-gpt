# Deepseek Sparse Attention (DSA)
# Same idea as SWA we attend only a subset of previous tokens. we attend those in sparsity 
# Lightning Indexer : compute similarity score between each new query token and prev tokens
# Top-K : keep K largest score
# sparse attention score : mask out additional tokens

import torch 
import torch.nn as nn 
from base import BaseAttention
from dataclasses import dataclass , asdict
import random 

torch.manual_seed(42)
random.seed(42)

class DSA(BaseAttention):
    def __init__(self, *, num_heads:int, hidden_size:int, embedding_dim:int, idx_heads:int, idx_dim:int, top_k:int,  dtype= torch.float16,  **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0, f"Embedding dim is not divisible by heads, {self.embedding_dim} % {self.num_heads}"
        
        self.idx_heads = idx_heads
        self.idx_dim = idx_dim
        self.top_k = top_k
        
        
        # attention model
        self.q_proj = nn.Linear(embedding_dim, hidden_size , bias = False ,dtype = dtype)
        self.k_proj = nn.Linear(embedding_dim , hidden_size, bias = False,dtype = dtype)
        self.v_proj = nn.Linear(embedding_dim , hidden_size, bias = False,dtype = dtype)
        self.out_proj = nn.Linear(hidden_size , embedding_dim, bias = False,dtype = dtype)
        
        # indexer model
        self.idx_q = nn.Linear(embedding_dim, idx_heads * idx_dim ,bias=False, dtype=dtype)
        self.idx_k = nn.Linear(embedding_dim, idx_dim, bias=False, dtype=dtype)
        self.idx_w = nn.Linear(embedding_dim, idx_heads, bias=False, dtype=dtype)
    
    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def index_scores(self, x):
        B, T, _ = x.shape
        q = self.idx_q(x).view(B, T, self.idx_heads, self.idx_dim).transpose(1, 2)  # B,H,T,D
        k = self.idx_k(x).unsqueeze(1)                                               # B,1,T,D
        w = self.idx_w(x).transpose(1, 2).unsqueeze(-1)                              # B,H,T,1
        s = torch.relu(q @ k.transpose(2, 3))                                        # B,H,T,T
        return (w * s).sum(dim=1)                                                    # B,T,T

    def forward(self, input:torch.Tensor):
        B, T, C = input.shape

        idx = self.index_scores(input.detach())  # B,T,T
        causal_mask = torch.tril(torch.ones_like(idx), diagonal=0) > 0.0  # dont leak future
        idx = idx.masked_fill(causal_mask==0, -float('inf'))

        top = idx.topk(min(self.top_k, T), dim=-1).indices # top-k largest values are choosen 
        keep_mask = torch.zeros_like(causal_mask).scatter_(-1, top, True) & causal_mask

        q = self.split_heads(self.q_proj(input))
        k = self.split_heads(self.k_proj(input))
        v = self.split_heads(self.v_proj(input))

        attn_score = (q @ k.transpose(2, 3)) * (self.head_dim**(-0.5))  # B,N,T,T
        masked_score = attn_score.masked_fill(keep_mask.unsqueeze(1)==0, -float('inf')) # take up those tokens 
        attn_softmaxed = torch.softmax(masked_score, dim=-1)

        output = (attn_softmaxed @ v).transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        with torch.no_grad():
            target = attn_softmaxed.float().sum(dim=1)
            target = target / target.sum(dim=-1, keepdim=True)
        
        log_p = torch.log_softmax(idx.float().masked_fill(keep_mask==0, -float('inf')), dim=-1)
        log_p = log_p.masked_fill(keep_mask==0, 0.0)
        idx_loss = -(target * log_p).sum(-1).mean()
        
        return self.out_proj(output), idx_loss

if __name__ == "__main__":
    @dataclass
    class Config:
        batch_size:int = 1
        ctx_length:int = 100
        num_heads:int = 8
        idx_heads:int = 4
        idx_dim:int = 64
        top_k:int = ctx_length // 8 # guessing out 
        hidden_size:int = 1024
        embedding_dim:int = 512 # embedding table output
            
    cfg = Config()
    cfg_dict = asdict(cfg)
    dtype = torch.float16
    mha = DSA(**cfg_dict, dtype = dtype)

    B,T,C = cfg_dict.get("batch_size"), cfg_dict.get("ctx_length"), cfg_dict.get("embedding_dim") 
    input_vec = torch.randn((B,T,C), dtype = dtype)
    
    out = mha(input_vec)
    print(out[0].shape)
    # passed = mha._test(input_vec)
    # print("Test Passed ? ", passed)
    
    # Test-1
    # B, T, C = 1, 3, 4
    # num_heads = 2
    # hidden_size = 4
    # embedding_dim = 4

    # mha = DSA(num_heads=num_heads, hidden_size=hidden_size, embedding_dim=embedding_dim)
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
    

'''
Impt : 
.view() reads the same memory again in same order so with new shape number the prefix matches works out for this
.transpose() does not move numbers it swaps jump rules so you read the same memory in different order 
'''
 

 