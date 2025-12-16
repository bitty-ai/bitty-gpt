import os 
import sys
from uu import encode

from multiprocessing import Process, Pool
from bitty import Tokenizer
from bitty.chunking import read_data_by_delimiter


class Dataset:
    def __init__(self, vocab_path):

        pass

    def data_to_tokens(self, ):
        # encoder path  

        pass

def read_data_in_chunks(file_path:str, CHUNK_SIZE:int | None = None):
    if not CHUNK_SIZE:
        CHUNK_SIZE = 50 # 100 mb of chunk 
    
    CHUNK_SIZE_BYTES = CHUNK_SIZE * 1024 * 1024

    buffer = b''
    with open(file_path , 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE_BYTES, err)

            # end chunk on not a byte

            if not chunk:
                if buffer: 
                    yield buffer
                break 
            
             
    

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    train_dir = os.path.join(data_dir , 'TinyStoriesV2-GPT4-valid.txt')

    encoded_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'encoded')
    save_path = os.path.join(encoded_dir , 'vocab.json')

    special_tokens = [b'<endoftext>']

    # make dir if not already present 
    # dataset to download from 
    Tokenizer(dataset_path=train_dir, special_tokens=special_tokens, num_merges = 10_000, save_path=save_path)


    # load the vocab from here and then call this method defined in class
    tokenizer = Tokenizer(file_path=save_path)

    with Pool(processes=max(6, os.cpu_count()) )
    for chunk_data in read_data_by_delimiter(data_file = train_dir, delimiter = special_tokens):
        tokenizer.encoder(chunk_data)
        # the vocab.json is done now need to encode our dataset based on this !
        train_tokenizer.encoder() # can I multiprocess a function getting from a seperate class ? 


