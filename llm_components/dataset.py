from torch.utils.data import Dataset
import torch
from typing import Optional
from tokenizer.tokenizer import Tokenizer
import pickle
import os
import random
import torch

class UnsupervisedDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: Tokenizer, ctx_lim: int, data_limit: Optional[int]=None):
        self.tokens = None
        self.ctx_lim = ctx_lim
        self.load_data(data_file, tokenizer)
        
    def load_data(self, file_path, tokenizer):
        if os.path.exists('tokens.cache'):
            with open("tokens.cache", 'rb') as tokens_cache:
                self.tokens = pickle.load(tokens_cache)
                print("loaded cached tokens...")
            return
        content = open(file_path, 'r', encoding='utf-8').read()
        self.tokens = tokenizer.encode(content)
        with open("tokens.cache", 'wb') as token_cache:
            pickle.dump(self.tokens, token_cache)
            print("cached!!")
        print("loaded all data!")
    
    def __len__(self):
        return len(self.tokens)
    
    def torchify(self, x, **kwargs):
        return torch.tensor(x, **kwargs)

    def __getitem__(self, item):
        str_idx = random.randint(0, len(self.tokens)-self.ctx_lim-2)
        x = self.tokens[str_idx:str_idx+self.ctx_lim]
        y = self.tokens[str_idx+1:str_idx+self.ctx_lim+1]
        return self.torchify(x, dtype=torch.long), self.torchify(y, dtype=torch.long)

# tok = Tokenizer("tokenizer/vocab.bpe", "tokenizer/merges.bpe") 
# ds = UnsupervisedDataset('data.txt', tok, 10)
# print(ds[0])
