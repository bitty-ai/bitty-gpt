import torch 
import torch.nn as nn 
from base import BaseAttention
from dataclasses import dataclass , asdict
import random 

torch.manual_seed(42)
random.seed(42)

class GQA(BaseAttention):
    '''
    Grouped query attention , limiting key and values heads and reusing those to save KV cache
    Query heads are not reduced 
    '''
    def __init__(self, *, num_heads:int, key_value_heads:int, hidden_size:int, embedding_dim:int,dtype= torch.float16, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_value_heads = key_value_heads
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        assert self.hidden_size % self.num_heads == 0, f"Embedding dim is not divisible by heads, {self.embedding_dim} % {self.num_heads}"
        assert self.num_heads % self.key_value_heads == 0
        self.keyvalue_ratio = self.num_heads // self.key_value_heads
        
        self.head_dim = self.hidden_size // self.num_heads
        assert self.keyvalue_ratio > 1

        kv_hidden_size = hidden_size // self.keyvalue_ratio
        self.q_proj = nn.Linear(embedding_dim , hidden_size , bias = False ,dtype =dtype)
        self.k_proj = nn.Linear(embedding_dim , kv_hidden_size , bias = False,dtype =dtype)
        self.v_proj = nn.Linear(embedding_dim , kv_hidden_size , bias = False,dtype =dtype)
        self.out_proj = nn.Linear(hidden_size , embedding_dim , bias = False,dtype =dtype)
        
    def forward(self, input:torch.Tensor):
        B,T,C = input.shape
 
        q = self.q_proj(input).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(input).view(B,T,self.key_value_heads,self.head_dim).transpose(1,2) # B,KVH,T,HD
        v = self.v_proj(input).view(B,T,self.key_value_heads,self.head_dim).transpose(1,2)
        
        print("K shape is : ", k.shape)

        k = k.repeat_interleave(self.keyvalue_ratio, dim=1) # B,NH,T,HD
        print("k shape : ", k.shape)
        print("q shape : ", q.shape)
        
        attn_score = (q @ k.transpose(2,3)) * (self.head_dim**(-0.5)) # B,NH,T,T 
        
        attn_mask = torch.tril(torch.ones_like(attn_score), diagonal=0) > 0.0 # dont leak future 
        masked_score = attn_score.masked_fill(attn_mask==0,  -float('inf'))
        
        attn_softmaxed = torch.softmax(masked_score ,dim=-1)
        
        # add a torch hook here !!
        row_sums = attn_softmaxed.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        )
        
        print("Softmax test passed") 
        
        v = v.repeat_interleave(self.keyvalue_ratio, dim=1) # B,NH,T,HD
        output = (attn_softmaxed @ v).transpose(1,2).contiguous().view(B,T,self.hidden_size) # b,n,t,h
        return self.out_proj(output).view(B,T,C)

if __name__ == "__main__":
    @dataclass
    class Config:
        batch_size:int = 1
        ctx_length:int = 100
        num_heads:int = 8
        hidden_size:int = 1024
        key_value_heads:int = 3 
        embedding_dim:int = 512 # embedding table output
            
    cfg = Config()
    cfg_dict = asdict(cfg)
    dtype = torch.float16
    mha = GQA(**cfg_dict, dtype = dtype)

    B,T,C = cfg_dict.get("batch_size"), cfg_dict.get("ctx_length"), cfg_dict.get("embedding_dim") 
    input_vec = torch.randn((B,T,C), dtype = dtype)
    
    out = mha(input_vec)
    passed = mha._test(input_vec)
    print(out.shape)
    print("Test Passed ? ", passed)
    
    # # Test-1
    # B, T, C = 1, 3, 4
    # num_heads = 2
    # hidden_size = 4
    # embedding_dim = 4

    # mha = GQA(num_heads=num_heads, hidden_size=hidden_size, embedding_dim=embedding_dim)
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
 