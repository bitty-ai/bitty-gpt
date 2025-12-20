# Training a Language model from scratch 

So in the last blog we learned how to build a tokenizer from scratch the sole thing that powers how LLM are learning and we also saw the caveats to not fall into while building one from scratch and ended with our own small tokenizer library from scratch called it `bitty-tokenizer`. So today as more complete version we will be building `bitty-gpt`, a custom gpt implmentation build on top of core pytorch and understanding what all caveats to look before. In this article you will learn about how to build a Optimizer function, activation function, loss function, neural nets, layers all from scratch. Read this twice and fork and play with the github link only then you will really understand this

ANDREJ QUOTE : LEARNING SHOULD BE PAINFUL, IF ITS NOT PAINFUL THEN YOU ARE NOT LEARNING YOU ARE PASSIVELY READING IT  


So we have have to create dataset, 
Here we will use `bitty-tokenizer` library and we will import Tokenizer class from there and will train our tokenizer on this vocabulary and then convert our dataset to this format , for an indepth understanding on tokenizer refer to previous article :[LINK IT HERE]  


### LANGUAGE MODELING 
So once we have created the dataset now we need to build a language model using the attention architecture 

-- add image of attention architecture--


## Embedding layers  
So now our dataset is a unsigned integer values like `[1000, 329, 2412 .... 312, 0]` and we need to project those an trained embedding space so that these abstract tokens get there meaning from it, we need to create an trainable embedding table ( trainable here means having `requires_grad` as True) for these tokens max value of a token can be upto length of vocab size ( 10,000 in our case) and we will project that to an embedding space of 1024 , depends from model architecture.  

Initialization in neural networks can help you break or make a system so a better way is to intialize with some lower values and aftet a lot of ablation studies researches have agreed to these intialization parameters

[Add Picture from the handbook Initialization parameters]

[Implementation of the same ]

Initializing as a parameter auto sets `requires_grad = True`


## Rotational Positional Embedding ( RoPE )
So the idea of RoPE is to add relative positional embedding by taking an absolute difference of values between the relative positions and using some basic maths we can reduce memory computation by a significant factor.

I highly recommend watching 'How Rotary Positional Embedding' youtube video by Jia-Bin Huang


## Building Attention block 

So Attention blocks are composed of attention mechanism (finds how much attention to give to each token and then we use MLP)

### Attention Operation 
Attention Operation is used to find relationship /importance between words and how to create a meaningful sentence from it. 
Here we project input to a latent dimension do an attention operation

Attention is a soft operation, where we find all relevant tokens and based on there probability pick those up in weighted manner

### MLP 
MLP first projects that to a latent dimension adds non linearity over it and then brings back dimension to its original shape 
Projecting to a higher space acts as an expansion layer it acts like expansion layer useful to detect patterns or concepts in input
Then adding non-linearity is for understand complex / non-linear concepts 
Then projecting back to lower dimension acts as a bottleneck to pass most important information learned   

MLP also acts as a key-value lookup and this is linearity acts as a way to remove the values that give a low score and then we take there value vector so that we multiply this with the correct value and that too in a weighted fashion !! 

So its like expansion in MLP acts as a  

This is a reason when we do finetuning, we update weights of the MLP layers as those are ones that act as `key-value` and those are the ones that learn about what to output compared to attention that has more of which token is matters 
So we can assume non-linearity in MLP as one more layer of filtering before getting actual relevant values out       

See papers  : https://arxiv.org/pdf/2012.14913

### Linear Layers 
Linear layers are useful for information expansion and used to convert from from one latent space to another so this is used in attention operation to convert from hidden spaces to functional space(token space for our case) so that communication can happen. 

The initialisation of linear layers are done in the manner as shown in above image. 

## RMS Norm
Please refer to article on normalization layer for indepth explanation for it [LINK ARTICLE]  

## Attention block 
Attention blocks are quite simple to understand we do attention operation that include :
1. Matrix Multiplying of Query with Keys to get relevant tokens
2. Get Attention matrix and normalize it with dimension 
3. Get probability scores as output 
4. Multiply with values for those 

And the intuition for all these are already explained above and code looks like this : 
<code image add>  


## MLP 
The activation function we use here is called SwiGLU and its quite similar to ReLU check the image below 

[Image from tldraw on activation]

Rest of code is self explanatory and intuition has been explained above

[code photo]

## Transformer Block
The concatenation of the Attention block and MLP makes up a transformer block  


## Transformer LM 
Adding multiple layers and adding these all together we get transformer LM


## Non Linearity / Activation functions 

There are used to give non-linearity to a model else its just a linear transformation only so the Magic of DL comes from these activation layer, there are various types of activation layers and we need to carefully choose these cause of gradient explosion or collapse 

[need to create indepth article for this]


## Loss functions
Here we are using Cross Entropy loss and this stems back from information theory and in DL we now use variants of this only like getting entropy , KL divergence etc , these are easy to understand to dial up     


## Optimizer function 
We used AdamW and 

## Training  


## Inference 


## Saving and loading 



## Caveats to consider before 

* Custom Loss function : Its easy to build a custom cross entropy function and not add the logic for `ignore_idx` this can easily break your system as model will learn to predict `<|endoftext|> / <pad>` token as a next token  

* Using multiprocessing map over a string , this takes individual bytes for a list 

* Handling special tokens in encoder (bitty tokenizer covers this for you :)  

* Using a custom optimizer function :

## End Note
Thank you for reading til the end if you have made so far that definitely means you want to a real Machine learning Engineer and not just make API calls to remote servers and I am pretty sure you will love bitty-ai for same 