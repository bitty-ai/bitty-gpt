import torch 
import sys 
import os
import wandb
import torch.nn as nn 
from dotenv import load_dotenv
load_dotenv()
import traceback 
from functools import partial
import numpy as np 

import argparse
from dataclasses import dataclass , asdict
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root-folder
data_path = os.path.join(dir_path , 'data/') 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nn_modules import TransformerLM 
from src.loss import cross_entropy
from src.optimizer import AdamW , cosine_learning_rate_scheduler
from data.processing import Processing
from src.inference import Inference
# from cs336_basics.lm_model.data import dataloader ,process_file_parallel

from bitty import Tokenizer

dtype = torch.bfloat16
# dtype = torch.float32
device = 'cpu'
if torch.cuda.is_available():
    device ='cuda'
elif torch.mps.is_available():
    device = 'mps'

print(f"Using device : {device} and Using datatype as : {dtype}\n\n")

# needs to be type annotated to be defined as instance attributes  
@dataclass
class GPT_2_XL:
    vocab_size:int = 300
    context_length:int = 1024
    num_layers:int = 12
    d_model:int = 1024//2
    num_heads:int = 16//2
    d_ff:int = 4096//2
    
epochs = 100
batch_size = 4
lr = 3e-4
config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size,
    "architecture": "Bitty_LLM_validation"
}

dtype_for_bin = None
if GPT_2_XL.vocab_size < 64000:
    dtype_for_bin = np.uint16
else:
    dtype_for_bin = np.uint32


# TODO: remove 
merges_input_path = dir_path + 'vocab/tiny_story_merges.json' # load merges
special_tokens = [b'<|endoftext|>']


def _get_dataset(dataset_type):
    if dataset_type == 'overfit':
        ## Overfitting
        dataset_path = data_path + 'Overall.txt'
        tokenizer_input_path = dir_path + '/vocab-tinystories-training-set.json' #Load-Vocab
        save_path = dir_path + '/overfitted_tokenized_data.bin'


    elif dataset_type == 'testing':
        ## Testing
        dataset_path = data_path + 'TinyStoriesV2-GPT4-valid.txt'
        tokenizer_input_path = dir_path + '/vocab-tinystories-training-set.json' #Load-Vocab
        save_path = dir_path + '/valid_tokenized_data.bin'


    else:
        ## Training
        dataset_path = data_path + 'TinyStoriesV2-GPT4-train.txt'
        tokenizer_input_path = dir_path + '/vocab-tinystories-training-set.json' #Load-Vocab
        save_path = dir_path + '/train_tokenized_data.bin'

    return dataset_path, tokenizer_input_path , save_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=int, required=False, default = 0)
    parser.add_argument("--train-dataset", type=str , required = False, default = None)
    parser.add_argument("--train-vocab", type=int , required = False, help = 'If this is set to false then the we will load the vocab from the relevant folder', default = 0)
    parser.add_argument("--dataset-name", type=str, required=False, choices=["training", "testing", "overfit"], default="training", help="Possible categories are: training, testing, overfit")

    args = parser.parse_args()

    use_wandb = args.wandb
    train_dataset = args.train_dataset
    train_vocab = args.train_vocab
    dataset_name =args.dataset_name

    if dataset_name:
        dataset_path , tokenizer_input_path , save_path = _get_dataset(dataset_name)
    else:
        raise ValueError('Need to add Dataset name to this !')

    if use_wandb:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project="bitty-ai-overfit", config=config)

    if train_vocab:
        if len(dataset_path) == 0 : 
            raise ValueError(f'Please provide a valid dataset path for training the vocab , currently found {dataset_path}')
        vocab = Tokenizer(dataset_path = dataset_path, num_merges = GPT_2_XL.vocab_size, special_tokens = special_tokens, save_path = tokenizer_input_path).train()
    else:
        if len(tokenizer_input_path) == 0:
            raise ValueError(f'Either provide path for loading tokenizer or provide a pass --train-vocab 1 , Found Current path as {tokenizer_input_path}')
        vocab = Tokenizer.load_vocab(file_path = tokenizer_input_path)

    # tokenizer 
    tokenizer = Tokenizer(vocab = vocab, special_tokens=special_tokens) # create a tokenizer class 

    processing = Processing(tokenizer , input_path = dataset_path, output_path= save_path, special_tokens=special_tokens)
    
    if train_dataset: # converts input dataset to .bin to train a model 
        print('STARTING TO CONVERT DATA TO BINARY FORMATED BASED ON THE TRAINED VOCABULARY')
        processing.data_to_bin(dtype = dtype_for_bin)

    print('Data converted to tokens')

    # model architecture defined 
    print('model parameters are : ', asdict(GPT_2_XL()))    
    model = TransformerLM(**asdict(GPT_2_XL()) , rope_theta = 10_000, device = device, dtype = dtype)
    print('model is defined as : ', model)

    total_params = sum(param.numel() for param in model.parameters())
    print("Total number of parameters in the model:", total_params//1e+6 , " M")

    lrs = (3e-4, 4e-4)
    optimizer = AdamW(params = model.parameters(), lr = lrs[0], lr_scheduler=cosine_learning_rate_scheduler, min_lr = lrs[1])
    
    pad_id = tokenizer.encoder('<|endoftext|>')[0]
    loss_fn = partial(cross_entropy, ignore_idx =pad_id) #TODO : removed this to isolate the bug 
    print('the pad id that i found is : ', pad_id)
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction= 'mean') #this is working completely fine and this is using the pad-id also !   

    # import sys; sys.exit(0)

    if device=='cuda':
        print(f"Initial Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        print("Starting memory recording...")
        torch.cuda.memory._record_memory_history(max_entries=100000)

    if use_wandb:
        wandb.watch(model, optimizer, log="all", log_freq=10)

    try:
        # training loop
        step = 0
        for epoch in range(1,epochs):
            train_dataloader = processing.dataloader(batch_size = batch_size, context_length = GPT_2_XL.context_length, padding_token = '<|endoftext|>', device = device)

            print(f'Epoch iteration is : {epoch}')
            for idx, data in enumerate(train_dataloader):
                # labels = torch.tensor(labels, dtype = torch.long).contiguous()
                # targets = torch.tensor(targets, dtype = torch.long).contiguous()
                print(data)
                labels, targets = data
                # print("Output labels are: " , tokenizer.decoder(labels[0].detach().tolist()))
                # print('Targets labels are : ' , tokenizer.decoder(targets[0].detach().tolist()))

                optimizer.zero_grad(set_to_none = True)
                predictions = model(labels)
                B,T,C = predictions.shape
                loss = loss_fn(predictions.view(-1, C), targets.view(-1).to(device))

                print('loss is : ', loss)
                loss.backward()
                optimizer.step()
                step+=1

                print(f'Epoch : {epoch} | Step : {step} | Loss : {loss.detach().item():.3f}')
                if use_wandb:
                    wandb.log({
                        "train_loss": loss.item(), 
                        "epoch": epoch,
                        "custom_metric": loss.item() * 1.5 # You can log any math you want
                    })

                if idx % 10 == 0:
                    print(f'Inferencing and Saving for step: {idx} in Epoch : {epoch}')
                    inference_obj = Inference(model = model , tokenizer = tokenizer)
                    inference_obj.save()
                    output_generated = inference_obj.generate_sample(stop_token=0, device = device)
                    

            # output_generated = Inference(model = model , tokenizer = tokenizer).save().generate_sample(stop_token=0)
            # print(output_generated)
    
    except torch.cuda.OutOfMemoryError as e:
        if device=='cuda':
            print("Got an OOM error and saving screenshot for same...")
            try:
                # Take the snapshot of the recorded history
                snapshot = torch.cuda.memory._snapshot()
                
                # Save to file
                import pickle
                with open("memory_profile.pickle", "wb") as f:
                    pickle.dump(snapshot, f)
                    
                print("\nSUCCESS! Snapshot saved as 'memory_profile.pickle'")
        
            except Exception as e:
                print(f"Failed to save snapshot: {e}")

            # --- STOP RECORDING ---
            torch.cuda.memory._record_memory_history(enabled=None)
    

    except ValueError as e:
        traceback_string = traceback.format_exc()
        print('got a value error : ', e)
        print("--- Start of Captured Traceback String ---")
        print(traceback_string)
        print("--- End of Captured Traceback String ---")
        pass

    if use_wandb:
       wandb.finish()

