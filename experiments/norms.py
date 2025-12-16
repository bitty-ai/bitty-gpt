# how layer norm differ from rms norm in practice not just in theory 
import random

from torch.serialization import normalize_storage_type 
random.seed(42)

import torch
torch.manual_seed(42)


def batchnorm(x:torch.Tensor):
    eps = torch.tensor(data = 1e-10 , dtype= x.dtype)
    initial_term = (x - x.mean(dim=0, keepdim=True)) / ((x.std(dim=0)**2 + eps) ** 0.5)
    gain = torch.ones_like(initial_term) # fixed for explaining
    bias = torch.ones_like(initial_term)
    running_mean = torch.nn.parameter.Buffer(data=torch.tensor(data = 0)) # initialised (gets updated during training)
    running_std = torch.nn.parameter.Buffer(data =torch.tensor(data = 1))
    
    return (initial_term * gain) + bias
    
    
def rmsnorm(x:torch.Tensor):
    # root mean square 
    eps = torch.tensor(data = 1e-10, dtype = x.dtype)
    if x.dim() == 3:
        B, T, D = x.shape
        dim = -1
    else:
        # Assume T is the only dimension (1D input)
        x = x.unsqueeze(0).unsqueeze(-1)
        B, T, D = x.shape
        dim = 1
    rms = torch.sqrt(eps + torch.mean(x*x, dim = dim)) # BxT 
    gain = torch.ones(size = (B,T,D)) # in a NN this is a learnable parameters and has its impact 

    return torch.div(input = (gain * x) , other = rms.unsqueeze(-1))

def layernorm(x:torch.Tensor):
    eps = torch.tensor(data = 1e-10 , dtype= x.dtype)
    initial_term = (x - x.mean(dim =-1).unsqueeze(-1)) / ((x.std(dim=-1)**2 + eps) ** 0.5)
    gain = torch.ones_like(initial_term) # fixed for explaining
    bias = torch.ones_like(initial_term)
    
    return (initial_term * gain) + bias
    

if __name__ == "__main__":
    data = [1000, 0.001, 10, 1, 100, 0.1, 10000, 0.0001, 1 , 100 , 0.001 , 1e5, 1e-4] # prev layer input
    x = torch.tensor(data = data)
    # using a non linearity like sigmoid
    out = torch.nn.functional.sigmoid(x)

    grad_out = out * ( 1.0 - out ) 
    normalised_grad_out = batchnorm(grad_out)

    print('Input is : ', x)
    print('Output after activation is : ', out)
    print('Output after gradient is : ', grad_out)
    print('Normalised Grad Output after gradient is : ', normalised_grad_out)
    
    
    import sys; sys.exit(0)

    x= torch.randn(size = (4,10,32)) # 4 batches of 10 tokens each and there embedding dims are 32   
    x= torch.normal(mean = -1 , std = 3, size = (4,10,32)) # 4 batches of 10 tokens each and there embedding dims are 32   

    print('Input shape is ; ', x.shape)
    print('input mean is : ', x.mean().item())
    print('Input std is : ', x.std().item())

    out=rmsnorm(x)
    print('\nOutput shape is ; ', out.shape)
    print('Output mean is : ', out.mean().item())
    print('Output std is : ', out.std().item())


    pass


'''
GRADIENTS ARE CALCULATED INDIVIDUALLY FOR EACH DIMENSION THAT IS A VECTOR OF 100 DIMS WILL HAVE A DERIVATE FOR 100 DIMS (OBVIOUS)

RESULTS: 

rms norm : The inputs were from distribution : (-1, 3) and outputs are from distribution (-0.32 , 0.94) 

Interpretations :

1. Means tensor values that were sparse earlier they are now close ! 
2. means I have restricted there range
3. The min and max of features are within this range
4. So values that were larger earlier now have limited dominance , so this is used to get to avoid very large values and avoid being dominated by single params  

Internal covariate shift ( that is in iteration-1 the mean,std was 0,1 in iteraton-2 the mean,std was 3,-3 etc .. so in all these subsequent iteration before it converges down .. so to avoid all these shifting and to make it stabalise we do norms 

Making input values of mean,std as 0,1 this is called whitening

Q) I have one doubts lets say my starting weights were 0 mean, 1 std and not layers are learning to find the real values based on on the dataset that we are entering in the model , so if I use batchnorm and restrict weight movement in to 0 mean , 1 std ( as output form one laye risthe input to the other layer ) so then how will model reach to its correct distribution ?

A) The second part that is adding the learning Y and B is the one that is doing all the shifts. You are not restricting the weights to stay at 0 mean, 1 std forever. You are merely centering them there to make the calculations stable, and then giving the network a set of "knobs" ($\gamma, \beta$) to move the distribution to wherever it needs to be to solve the problem.

Q) why to use rmsnorm over layer norm? do a visual comparison between them 
A) RMS only scales input without shifting it ? but like batchnorm authors gave a method to come back to original distro why not that same in this  

'''