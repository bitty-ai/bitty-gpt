import torch 
import torch.nn as nn 
import math
from activations import softmax, SiLU
from typing import Optional


# linear layer 
class Linear(nn.Module):
    def __init__(self, in_features:int , out_features:int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.device = device
        self.dtype= dtype
        factory_kwargs = {"dtype": dtype, 'device':device}

        std = math.sqrt(2.0 / (in_features + out_features))  # correct formula

        weight_init_for_linear_layer =  torch.nn.init.trunc_normal_(torch.empty(out_features, in_features, **factory_kwargs), mean = 0.0 , std = std , a = -3*std , b = 3*std)

        self.weight = torch.nn.Parameter(data = weight_init_for_linear_layer)

    def forward(self,x:torch.Tensor)-> torch.Tensor:

        assert x.shape[-1] == self.in_features, 'The shapes of tensors are mismatching'

        output = torch.matmul(x , self.weight.transpose(-1,-2)) # (T x d_in) x ( d_in x d_out) 

        if self.device:
            output = output.to(self.device)        
        return output.contiguous() # This is an important flag to keep 

# Embedding layers
class Embedding(nn.Module):
    '''
    These are defined in torch longtensor format 

    num_embeddings are the size of vocab
    embedding_dim are the dims of embedding vectors 
    '''
    
    def __init__(self, num_embeddings:int, embedding_dim:int, device =None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype':dtype}

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.init.trunc_normal_(torch.empty(size = (num_embeddings, embedding_dim) ,**factory_kwargs), a = -3, b=3)
        self.embedding_matrix = torch.nn.Parameter(data = self.weight)
    
    def forward(self, token_ids : torch.Tensor)->torch.Tensor:
        out = self.embedding_matrix[token_ids]
        return out

# Normalization layer 
class RMSNorm(nn.Module):
    '''
    This is applied parameter wise to the input parameters that are received in the forward pass !

    '''
    def __init__(self, d_model:int, eps:float =1e-5, device = None , dtype=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device':device, 'dtype':dtype} 
        self.weight = torch.nn.Parameter(data=torch.ones(size = (d_model,), **factory_kwargs)) # this way all the input tokens across all batches will have the same layer gain values ( see feature dim5 got upscaled maybe you should also upscale it , that is rule transfer, different from batchnorm) 
        # this should / will be updated in the backprop of the model !! 

    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        input shape of x is :  N x T x E , and the gain is (1,1, E) 
        '''
        
        assert x.shape[-1] == self.d_model, f'X shape is : {x.shape} and d_model is {self.d_model}'

        in_dtype = x.dtype

        x = x.to(torch.float32) # need to set this to float 32 for numeric stability 

        square_root = torch.sqrt((torch.mean(x*x, dim =-1) + self.eps)) # reverse square root value
        rms  = 1/square_root 
        result =  x*self.weight # this gains is defined as a parameter so this should be taken in via model.parameters() and this will be updated in the backprop function ! 

        result = result*rms.unsqueeze(-1)
        
        return result.to(in_dtype)

# Feed forward model 
class FFN(nn.Module):
    def __init__(self, d_model:int, d_ff:int=None, **kwargs):
        super().__init__()
        if not d_ff:
            NEAREST_MULTIPLE = 64
            d_ff = NEAREST_MULTIPLE * math.floor(8*d_model/(3*NEAREST_MULTIPLE)) # nearest floor (as the value should be closer to the value of the nearest multiple) and should be close to dim of 64 

        self.w1 = Linear(in_features = d_model , out_features= d_ff, **kwargs) # weight are created as : out_features x in_features and in linear operation these are tranposed and used as in x out
        self.w2 = Linear(in_features = d_ff,  out_features = d_model, **kwargs)
        self.w3 = Linear(in_features = d_model , out_features= d_ff, **kwargs)


    def forward(self, x:torch.Tensor):
        return self.w2(SiLU(self.w1(x)) * self.w3(x))


# RoPE implementation 
class RotaryPositionalEmbedding(nn.Module):
    '''
    Rotational positional embedding this what was propeosed in the attention paper also and we just use a modified version of that same here !
    This is quite complex to understand so a mental understanding of having relative positional encoding compared to the absolute one also works well

    '''
    def __init__(self, d_k:int, max_seq_length:int, **kwargs) -> None:
        super().__init__()

        assert d_k%2 ==0 , f'Found the dimension as : {d_k} should be a multiple of 2'
        self.theta = 10_000 # Fixed frequency  
        self.d_k = d_k
        self.max_seq_length= max_seq_length 

    
    def rope_freqs(self):
        half = self.d_k//2
        power_raised = (-2*torch.arange(start =0, end = half) /self.d_k)
        freq = self.theta ** power_raised

        t = torch.arange(self.max_seq_length) # this works only for the t dimension (else will require) 
        angles= t[:,None] * freq[None,:]
        return angles.cos(), angles.sin() # precomputed values  


    def efficient_rope(self,x, cos, sin):
        
        #split 
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = cos.to(x.device)
        sin = sin.to(x.device)

        # apply rot
        x_rot_even = x_even * cos - x_odd * sin # 
        x_rot_odd = x_even * sin + x_odd * cos

        # interleave back to original shape
        x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        x_out = x_out.flatten(-2)    # fuse last 2 dims
        return x_out.to(device = x.device, dtype = x.dtype) 


    def forward(self, x:torch.Tensor , token_position:torch.Tensor=None):
        assert x.shape[-1]== self.d_k, f'Expected {self.d_k} ,recieved {x.shape} '

        # this could be one way : iterate throught hte batches and get the values out from the input matrix , total iterations are : O(batch x max_seq_len x d//2) 

        cos,sin = self.rope_freqs() # this buffer is made considering the max sequence length in action 
        # slice cos, sin to match this token lengths 
        if token_position is not None:
            *buf, seq_length = token_position.shape        
        else:
            *buf, seq_length, d_k = x.shape

        cos, sin = cos[:seq_length, ...] ,sin[:seq_length, ...]
        return self.efficient_rope(x,cos,sin)



# Attention operation as in paper   
def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask :Optional[torch.Tensor] = None)->torch.Tensor:
    '''
    Attention its a way of communicating with the batches
    
    Attention values : (Q @ K.T)
    Dimension is : d_k
    Apply a mask then softmax to convert those to probs
    Multiply with values to get the outputs out from the attention operation 

    Inputs are : q,k,v 
    Q could be : B,T,D
    or it could also be : B,T,H,D//H
    '''

    assert q.shape() == 3, f'The shape of query tensor should be of length 3 but found {q.shape}'
    D = q.shape[-1] 
    
    out = torch.matmul(q, k.transpose(-1,-2))
    scaled_out = out * (D ** (-0.5))
    
    # Causal masking 
    if mask is not None:
        masked_out = torch.where(mask == 0 , -torch.inf , scaled_out) # keep that value is mask == 1 else ignore that value 
        assert masked_out.shape == scaled_out.shape
    else:
        masked_out = scaled_out

    scaled_softmax_out = softmax(masked_out, -1)
    attention_score = torch.matmul(scaled_softmax_out ,  v)
    return attention_score

 
class Multihead_self_attention(nn.Module):
    '''
    Causal attention means that the model cannot look at the future tokens 
    Input params are: 
    
    * d_model : dimensions of the model 
    * num_heads are the total no. of heads to divide to and its belived that each head learn nuanced features that make up a sentence like punctuation ,structure, grammar etc  
    * max-seq-length : this defines the maximum input sequence length, (but what should be its capped value ?) # TODO    


    '''

    def __init__(self, d_model:int, num_heads:int, max_seq_length:int=None, device:str=None, dtype:str=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        
        factory_args = {'device':device ,'dtype':dtype}
        assert d_model %  num_heads == 0, f'The heads should be a factor of d-model .Found d_model as {d_model} and heads as {num_heads}' 
        
        self.positional_embedding = RotaryPositionalEmbedding(max_seq_length=max_seq_length,d_k=d_model//num_heads, device=device , dtype= dtype) # d_k = d_model / multi-si nice to be  

        self.q_proj = Linear(d_model, d_model, **factory_args)
        self.k_proj = Linear(d_model, d_model, **factory_args)
        self.v_proj = Linear(d_model, d_model, **factory_args)
        self.o_proj = Linear(d_model, d_model, **factory_args)
        

    def forward(self,x:torch.Tensor, token_position=None):
        B,T,D = x.shape
        q = self.q_proj(x).view(B, T,self.num_heads, self.d_model//self.num_heads).transpose(1,2) # TODO: revisit : in view the dimensions cant be split up randomly , that can cause errors later ! 
        k = self.k_proj(x).view(B, T,self.num_heads, self.d_model//self.num_heads).transpose(1,2)
        v = self.v_proj(x).view(B, T,self.num_heads,self.d_model//self.num_heads).transpose(1,2)

        mask = torch.where(torch.triu(input = torch.ones(size = (T,T),device = self.device), diagonal=1) == 1 , 0 , 1)
    
        q = self.positional_embedding(q,token_position)
        k = self.positional_embedding(k,token_position)
        
        out= scaled_dot_product_attention(q,k,v, mask = mask)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int,num_heads: int,d_ff: int,max_seq_len: int, weights=None, *args , **kwargs):
        '''
        Single Transformer Block
        d_model : dimension of the model 
        num_heads : no. of heads to divide into
        d_ff : dimension of feed forward network 
        max-seq-length : Max sequence length to maintain for this 
        

        '''
        
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        # device = kwargs.get('device', 'cpu')
        # dtype = kwargs.get('dtype', None)

        self.attn = Multihead_self_attention(d_model = d_model ,num_heads=num_heads, max_seq_length= max_seq_len, **kwargs)

        self.ffn = FFN(d_model, d_ff, **kwargs)

        self.ln1 = RMSNorm(d_model, **kwargs)
        self.ln2 = RMSNorm(d_model, **kwargs)

    def forward(self, in_features:torch.Tensor):
        # In_features are Batch , context length , D-model 
        B,T,D = in_features.shape
        
        x = in_features # input features
        x = self.ln1(x)
        x = self.attn(x)
        x =  in_features + x # dot wise addition  
        x_partial = x
        x = self.ln2(x)
        x = self.ffn(x)

        return x_partial + x


class TransformerLM(nn.Module):
    '''
    Finally the training pipeline begins !! 
    '''

    def __init__(self, vocab_size :int , context_length:int, num_layers:int, d_model:int, num_heads:int , d_ff:int , rope_theta:int , weights = None, dtype = None, device = 'cpu'):
        '''
        This is the transformer lm layer that we build over this !

        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        '''
        super().__init__()

        factory_args = {'device':device, 'dtype':dtype}
        self.token_embeddings = Embedding(num_embeddings = vocab_size, embedding_dim = d_model, **factory_args)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model = d_model, d_ff = d_ff , max_seq_len=context_length, theta = rope_theta , num_heads=num_heads, **factory_args) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model = d_model, **factory_args)
        self.lm_head = Linear(in_features = d_model, out_features = vocab_size, **factory_args)
        
    def forward(self, x:torch.Tensor):
        print('position embedding values are : ',self.token_embeddings.weight.mean(),self.token_embeddings.weight.max())
        x = self.token_embeddings(x)
        
        for module in self.layers:
            x = module(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits  # this excludes the softmax calculation

