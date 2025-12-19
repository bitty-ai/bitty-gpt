import torch 
import torch.nn as nn

# region  ACTIVATION FUNCTIONS 
def SiLU(x):
    return x * torch.sigmoid(x) # element wise ops


def GLU(x, weight1, weight2):
    assert x.shape[-1] == weight1.shape[0], f'Found some mismatching dimensions : x is {x.shape} and weight is : {weight1.shape}'

    first_term = torch.sigmoid(x @ weight1) # (t x d_in) (d_in x d_out) # so we need to output that in the transposed

    assert weight2.shape[-1] == x.shape[0], f'Found some mismatched dimensions : x is {x.shape} and weight2 is : {weight2.shape}'

    second_term = weight2 @ x
    return first_term * second_term



def softmax(x:torch.Tensor, dim:int =-1):
    max_values = torch.argmax(x, dim = dim) # N size 
    
    maximum_in_each_row = x.gather(dim,index = max_values.unsqueeze(dim))
    
    # assert maximum_in_each_row.shape == max_values.shape, f'shape mismatch : {maximum_in_each_row.shape} and {max_values.shape} '

    normalised_x = x-maximum_in_each_row # broadcast operation ( this operation is done to avoid getting numerical stability issues )

    exp_norm_x = torch.exp(normalised_x)

    each_row_summed = torch.sum(exp_norm_x, dim =dim).unsqueeze(dim) # N x 1 

    return exp_norm_x / each_row_summed



def log_softmax(x:torch.Tensor, dim = -1):
    max_values = torch.argmax(x, dim = dim) # N size 
    
    maximum_in_each_row = x.gather(dim,index = max_values.unsqueeze(dim))
    
    # assert maximum_in_each_row.shape == max_values.shape, f'shape mismatch : {maximum_in_each_row.shape} and {max_values.shape} '

    normalised_x = x-maximum_in_each_row # broadcast operation ( this operation is done to avoid getting numerical stability issues )

    exp_norm_x = torch.exp(normalised_x)

    each_row_summed = torch.sum(exp_norm_x, dim =dim).unsqueeze(dim) # N x 1 

    return normalised_x - torch.log(each_row_summed)

