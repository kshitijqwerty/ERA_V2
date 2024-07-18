from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import tiktoken
import time



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        # Scale Flag 
        self.c_proj.GPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embed size should be divisible by number of heads
        assert config.n_embed % config.n_head == 0
        # K, Q, V but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # Scale Flag 
        self.c_proj.GPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer("bias", torch.tril(torch.ones(config.chunk_size, config.chunk_size))
                             .view(1, 1, config.chunk_size, config.chunk_size))

    
    def forward(self, x):
        batch, time, channels = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)\
        # (batch, num_head, time, hs)
        k = k.view(batch, time, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch, time, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch, time, self.n_head, channels // self.n_head).transpose(1, 2)

        # Calculate attention

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # add masking by making upper tril -inf plus softmax
        # att = att.masked_fill(self.bias[:,:,:time,:time] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Use Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        y = self.c_proj(y)

        return y
    

class Block(nn.Module):
    # Hidden block impplementation
    def __init__(self, config):
        super().__init__()
        # Layer Norm 1
        self.ln_1 = nn.LayerNorm(config.n_embed)
        # causal self attention layer
        self.attn = CausalSelfAttention(config)
        # Layer norm 2
        self.ln_2 = nn.LayerNorm(config.n_embed)
        # Simple MLP
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Skip connnections 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    chunk_size = 1024 # context length
    # vocab_size = 50257 # number of tokens: 50k BPE merges, 256 bytes token + 1 end token
    vocab_size = 50304 # Powers of 2, for optimization
    n_layer = 12
    n_head = 12
    n_embed = 768 

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Weight tranformer embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            # weight positional embeddings
            wpe = nn.Embedding(config.chunk_size, config.n_embed),
            # hidden layers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # linear feed-forward
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        # linear head
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.chunk_size, f"Cannot forward sequence of length {T}"
        # Create T length positional embed
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # get the embeddings
        pos_emb = self.transformer.wpe(pos) # shape(T, n_embed)
        tok_emb = self.transformer.wte(idx) # shape(B, T, n_embed)
        #  pass the embedding to each block
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class DataLoaderLite:
    def __init__(self, B, T, file):
        self.B = B
        self.T = T

        with open(file, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) #inputs
        y = (buf[1:]).view(B, T) # targets

        # advance the position of tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds then reset it
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y


class TrainGPT:
    def __init__(self, optimizer="AdamW", lr=3e-4, model_compile=False):
        # Init device
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print(f'using device {device}')
        self.device =device

        self.model = GPT(GPTConfig())
        self.model.to(self.device)
        
        # If you want to compile the model for ahead of time compilation (Huge Speedup)
        if model_compile:
            self.model = torch.compile(self.model)

        # Add more optimizers later if required
        if optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr, betas=(0.9, 0.95), eps=1e-8)
        
        self.writer = SummaryWriter(log_dir='logs')

        
        

    def train_gpt(self, train_loader, steps=50, print_after_steps=10):
        self.model.train()
        # train optimized 
        torch.set_float32_matmul_precision('high')

        for i in range(steps):
            t0 = time.time()
            x, y = train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # Autocast optimization use bfloat only for Ampere and above otherwise float16
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)

            loss.backward()

            # Clipping gradiant norm to 1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            torch.cuda.synchronize() # wait for GPU to finish
            t1 = time.time()
            dt = (t1 - t0) * 1000
            tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
            if i % print_after_steps == 0:
                print(f"Step {i}, loss: {loss.item()}, dt: {dt:.2f}ms tok/sec: {tokens_per_sec:.2f}")

             # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', loss.item(), i)
            self.writer.add_scalar('Time/dt', dt, i)
            self.writer.add_scalar('Performance/tokens_per_sec', tokens_per_sec, i)
        
        self.writer.close()


        
        







# num_return_sequences = 5
# max_length = 30


# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'am a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
# x = tokens.to(device)




# torch.manual_seed(42)
# torch.cuda.manual_seed(42)


# model = GPT(GPTConfig())
# model.eval()
# model.to(device)

# # optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for i in range(50):

# while x.size(1) < max_length:
#     with torch.no_grad():
#         # get logits from the model
#         logits, loss = model(x)
#         print(f"Loss: {loss}")
#         logits = logits[:,-1,:]
#         probs = F.softmax(logits, dim=-1)

#         topk_prob, topk_indices = torch.topk(probs, 50, dim=-1)

#         ix = torch.multinomial(topk_prob, 1)

#         xcol = torch.gather(topk_indices, -1, ix)

#         x = torch.cat((x,xcol), dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
