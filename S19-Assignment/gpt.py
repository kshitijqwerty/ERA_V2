import torch
import torch.nn as nn
from torch.nn import functional as F

n_embeds = 384
context_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_blocks = 10
num_heads = 6

# Simple Feed Forward after Attention heads
class FeedForward(nn.Module):
    def __init__(self, n_embeds):
        super().__init__()
        self.net = nn.Sequential(
            # Paper (Att. is all you need) has a quadruple internal layers increase for ff net
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            # add a projection layer, need to investigate
            nn.Linear(4 * n_embeds, n_embeds)
        )
    
    def forward(self, x):
        return self.net(x)
            
# Self Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embeds, head_size, bias=False)
        self.query = nn.Linear(n_embeds, head_size, bias=False)
        self.value = nn.Linear(n_embeds, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        # Calculate key, query and value pair for attention
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # The scaling factor for making the distribution having variance 1 instead of function of head count
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        
        out = wei @ v 
        # return attention
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Added a projection layer, need to investigate
        self.proj = nn.Linear(n_embeds, n_embeds)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# A Combined block which has self attention and feed forward
class Block(nn.Module):
    def __init__(self, n_embeds, n_heads):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_heads, n_embeds//n_heads)
        self.ff = FeedForward(n_embeds)
        # Defining layer norms 
        self.lnorm1 = nn.LayerNorm(n_embeds)
        self.lnorm2 = nn.LayerNorm(n_embeds)
    
    def forward(self, x):
        # Slight deviation from the paper, Layer norms will be applied before rather than after the attention and ff
        # Also has skip connections 
        x = x + self.sa_head(self.lnorm1(x))
        x = x + self.ff(self.lnorm2(x))
        return x

# Bigram Model -> Decoder 
# class BiGramModel(nn.Module):
class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # first get the token embedding from the vocab then create the new logits from the token embeds
        self.embedding_table = nn.Embedding(vocab_size, n_embeds) # (batch, time, embed_size)
        # Positional embedding of the size chunk X embed_size
        self.positional_embed_table = nn.Embedding(context_length, n_embeds)

        # add a attendtion head
        # self.sa_head = Head(n_embeds)
        # Modified for multihead
        # self.sa_head = MultiHeadAttention(4, n_embeds//4)

        # adding multiple blocks
        # self.blocks = nn.Sequential(
        #     Block(n_embeds, n_heads=4),
        #     Block(n_embeds, n_heads=4),
        #     Block(n_embeds, n_heads=4),
        #     nn.LayerNorm(n_embeds)
        # )
        self.blocks = nn.Sequential(*[Block(n_embeds, n_heads=num_heads) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size) # (batch, time, vocab_size)

        self.ff = FeedForward(n_embeds)
    
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.embedding_table(idx)
        # Create a positional embedding based on location [0,1,2,3,..]
        pos_emb = self.positional_embed_table(torch.arange(T, device=device))
        x = token_emb + pos_emb

        # x = self.sa_head(x)
        x = self.blocks(x)
        x = self.ff(x)

        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            # Calculate the loss 
            B, T, C = logits.shape
            
            # Matching the shape of the logits and targets 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Dont allow more than chunk size to enter the model at a time
            idx_chopped = idx[:, -context_length:]

            logits, loss = self(idx_chopped)
            
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx