
import os 
import torch 
import torch.nn as nn 
from typing import Union, Generator
import multiprocessing as mp
import numpy as np

class Processing:
    def __init__(self, tokenizer, input_path , output_path, special_tokens:list[bytes]):
        self.tokenizer = tokenizer
        self.input_path = input_path
        self.output_path = output_path
        self.special_tokens = special_tokens

    def read_data_to_bytes(self):
        # read the data in chunks of chunk-size from self.input_path andand make sure for encoding use self.tokenizer.encode(text_chunk)  

        CHUNK_SIZE = 50 
        filesize_in_mb = os.path.getsize(self.input_path) / (1024 * 1024)
        no_of_processes = filesize_in_mb // CHUNK_SIZE 
        processes = min(os.cpu_count()-3, no_of_processes)
        
        with open(self.input_path , 'rb') as f:
            # Read the file in chunks and make sure not to break multibyte utf-8 sequences.
            # We keep a buffer of leftover bytes, append new bytes, and detect full utf-8 code-points.
            buffer = b''
            while True:
                chunk = f.read(CHUNK_SIZE * 1024 * 1024)
                if not chunk: # EOF 
                    if buffer:
                        try:
                            text_chunk = buffer.decode('utf-8')
                            yield text_chunk
                        except UnicodeDecodeError:
                            raise ValueError('Something is wrong in the core logic then this case should never come  ')
                    break

                buffer += chunk
                try:
                    text_chunk = buffer.decode('utf-8')
                    yield text_chunk
                    buffer = b''
                except UnicodeDecodeError as e:
                    for i in range(1,5):
                        try:
                            text_chunk = buffer[:-i].decode('utf-8')
                            yield text_chunk
                            buffer = buffer[-i:]
                            break
                        except UnicodeDecodeError as e:
                            print('Got decoding error as e', e)
                            continue
                        

    def append_data_to_binary(self, text_chunk:list[int]):
        with open(self.output_path, 'ab') as bin:
            bin.write(text_chunk)

    def data_to_bin(self, dtype):
        # processing file parallel , use the tokenizer
        open(self.output_path, 'wb').close()


        # ... inside your class method ...

        with mp.Pool(processes=min(5,mp.cpu_count())) as pool:
            # pool.imap takes the function and the generator.
            # It passes each item yielded by the generator to the encoder function.
            # 'chunksize' can be set to 1 if your data chunks are large.
            for encoded_tokens in pool.imap(self.tokenizer.encoder, self.read_data_to_bytes()): # iterator map, this iteartes over the generator itself  
                
                # 'encoded_tokens' is now the result for one FULL chunk of data
                encoded_tokens_numpy = np.array(encoded_tokens, dtype=dtype)
                
                # Serialize to raw bytes
                binary_data = encoded_tokens_numpy.tobytes()
                
                # Write to your .bin file
                self.append_data_to_binary(binary_data)
                    # str_data = data 
            
            # encoded_tokens = self.tokenizer.encoder(str_data) # encoder should always contains special tokens  

            # encoded_tokens_numpy = np.array(encoded_tokens).astype(dtype).tobytes() # this serialises using which encoding format ? 
            # # add this to a .bin file  
            # self.append_data_to_binary(encoded_tokens_numpy)

        # self._check_bin_file(dtype= dtype) # check back this 

    def _check_bin_file(self, dtype):
        '''
        This method should be used check whether the saved values are the same as dataset   
        '''
        # Read raw bytes from .bin file
        with open(self.output_path, 'rb') as f:
            byte_data = f.read()

        # Convert back to numpy array of tokens
        tokens = np.frombuffer(byte_data, dtype=dtype)
        print(f"Loaded {len(tokens)} tokens from binary file.")

        # Try to decode those tokens using tokenizer (should match with the original text chunks)
        decoded_text = self.tokenizer.decoder(tokens.tolist())

        # Assert decoded text matches the original file stored in self.input_path
        with open(self.input_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        decoded_prefix = decoded_text[:len(original_text)]
        assert decoded_prefix == original_text, f"Found this : {decoded_prefix} should be : {original_text}"


    def dataloader(self, batch_size: int, context_length:int, device:str, dtype:Union[torch.dtype, np.dtype]=np.uint16, padding_token='<|endoftext|>'):
        # read from the saved output file
        file_bytes = os.path.getsize(self.output_path)
        total_tokens = file_bytes // np.dtype(dtype).itemsize
        
        arr = np.memmap(filename = self.output_path, dtype=dtype, mode = 'r' , shape=(total_tokens,)) # reading a .bin using memmap so that we dont bog the memory 
        tokens_per_loop_iteration = (context_length + 1) * batch_size
        num_batches = (total_tokens - 1) // tokens_per_loop_iteration
        pad_id = self.tokenizer.encoder(padding_token) 

        final_end_idx = 0

        for i in range(num_batches):
            start_idx = i * tokens_per_loop_iteration
            end_idx = start_idx + tokens_per_loop_iteration
            final_end_idx = end_idx # Keep track of where we stopped

            batch = arr[start_idx:end_idx]
            
            # Safety check (though logic guarantees this runs for full batches)
            if len(batch) != tokens_per_loop_iteration:
                break
                
            batch = batch.reshape(batch_size, (context_length + 1)) 
            
            x = torch.from_numpy(batch[:, :-1].astype(np.int64))
            y = torch.from_numpy(batch[:, 1:].astype(np.int64))
            
            yield x, y

        remaining_tokens = total_tokens - final_end_idx
        
        # We need at least 2 tokens to form one x->y pair. 
        if remaining_tokens > 1:
            # Fetch whatever is left
            leftover_data = arr[final_end_idx:]
            
            # Calculate how much padding is needed to fill the batch
            needed_padding = tokens_per_loop_iteration - remaining_tokens
            
            # Create the padding array
            padding = np.full((needed_padding,), pad_id, dtype=dtype)
            
            # Concatenate leftover data with padding
            full_batch = np.concatenate((leftover_data, padding))
            
            # Reshape and Yield
            batch = full_batch.reshape(batch_size, (context_length + 1))
            
            x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
            y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
            
            yield x, y
            
        del arr

            