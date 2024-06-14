import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(42)

class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.ctoi = {c:i for i, c in enumerate(self.chars)}
        self.itoc = {i:c for i, c in enumerate(self.chars)}
    
    def encode(self, x):
        return [self.ctoi[c] for c in x]
    
    def decode(self, x):
        return "".join([self.itoc[c] for c in x])
    
class Dataset:
    def __init__(self, text, tokenizer, split=0.9):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        train_split = int(split* len(self.data))
        self.train_data = self.data[:train_split]
        self.val_data = self.data[train_split:]

    def get_batch(self, split, context_length=256, batch_size = 64, device='cpu'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i: i+context_length] for i in ix])
        y = torch.stack([data[i+1: i+context_length+1] for i in ix])
        x, y  = x.to(device), y.to(device)
        return x, y



    

