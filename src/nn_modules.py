import torch 
import torch.nn as nn 
import math

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
        return output


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

