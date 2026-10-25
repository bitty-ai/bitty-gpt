# MLA : multihead latent attention 
# idea is same as GQA to reduce KV cache here, but we dont reduce KV heads rather we use lora (low-rank approximation) for weights matrices for all 3 Q,K,V  !
# TODO: deep dive why and where for this RoPE fails
 
import torch 
import torch.nn as nn 
from base import BaseAttention
from dataclasses import dataclass , asdict
import random 

torch.manual_seed(42)
random.seed(42)

class MLA(BaseAttention):
    def __init__(self, *, num_heads:int, hidden_size:int, embedding_dim:int, kv_latent_dim:int, q_latent_dim:int, dtype=torch.float16, **kwargs):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_down = nn.Linear(embedding_dim, q_latent_dim, bias=False, dtype=dtype)
        self.q_up = nn.Linear(q_latent_dim, hidden_size, bias=False, dtype=dtype) # q is reduced here to save computation / training memory not for cache 

        self.kv_down = nn.Linear(embedding_dim, kv_latent_dim, bias=False, dtype=dtype)
        self.k_up = nn.Linear(kv_latent_dim, hidden_size, bias=False, dtype=dtype)
        self.v_up = nn.Linear(kv_latent_dim, hidden_size, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(hidden_size, embedding_dim, bias=False, dtype=dtype)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, input:torch.Tensor, kv_cache=None):
        B, T, C = input.shape

        q = self.split_heads(self.q_up(self.q_down(input)))

        c_kv = self.kv_down(input)
        if kv_cache is not None:
            c_kv = torch.cat([kv_cache, c_kv], dim=1)  # cache holds only the small latent
        S = c_kv.shape[1]

        k = self.split_heads(self.k_up(c_kv))
        v = self.split_heads(self.v_up(c_kv))

        attn_score = (q @ k.transpose(2, 3)) * self.head_dim ** -0.5  #B,N,T,S --- #Trick : q * k_up * c_kv , rather than multiplying each tokne in cache to c_kv multiple just q to k_up and then do multiplication to c_kv .. 
        mask = torch.ones(T, S, dtype=torch.bool, device=input.device).tril(diagonal=S - T)  # shift for cached tokens
        attn_score = attn_score.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn_score, dim=-1)

        output = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.out_proj(output), c_kv


if __name__ == "__main__":
    @dataclass
    class Config:
        batch_size:int = 1
        ctx_length:int = 100
        num_heads:int = 8
        hidden_size:int = 1024
        embedding_dim:int = 512 # embedding table output
        kv_latent_dim:int = 64 
        q_latent_dim:int = 64
            
    cfg = Config()
    cfg_dict = asdict(cfg)
    dtype = torch.float16
    mha = MLA(**cfg_dict, dtype = dtype)

    B,T,C = cfg_dict.get("batch_size"), cfg_dict.get("ctx_length"), cfg_dict.get("embedding_dim") 
    input_vec = torch.randn((B,T,C), dtype = dtype)
    
    out = mha(input_vec)
    # passed = mha._test(input_vec)
    print(out[0].shape)
    # print("Test Passed ? ", passed)
    
    # Test-1
    # B, T, C = 1, 3, 4
    # num_heads = 2
    # hidden_size = 4
    # embedding_dim = 4

    # mha = MLA(num_heads=num_heads, hidden_size=hidden_size, embedding_dim=embedding_dim)
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
 

