import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.autograd import Variable

vocab_size = 4


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.key = nn.Linear(self.hyper_params["n_embd"], self.hyper_params["head_size"], bias=False)
        self.query = nn.Linear(self.hyper_params["n_embd"], self.hyper_params["head_size"], bias=False)
        self.value = nn.Linear(self.hyper_params["n_embd"], self.hyper_params["head_size"], bias=False)

        self.dropout = nn.Dropout(self.hyper_params["dropout"])

    def forward(self, x, attn_mask):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if attn_mask is not None:
            wei = wei.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.heads = nn.ModuleList([Head(self.hyper_params) for _ in range(self.hyper_params["n_head"])])
        self.proj = nn.Linear(self.hyper_params["head_size"] * self.hyper_params["n_head"], self.hyper_params["n_embd"])
        self.dropout = nn.Dropout(self.hyper_params["dropout"])

    def forward(self, x, attn_mask):
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.net = nn.Sequential(
            nn.Linear(self.hyper_params["n_embd"], 4 * self.hyper_params["n_embd"]),
            nn.ReLU(),
            nn.Linear(4 * self.hyper_params["n_embd"], self.hyper_params["n_embd"]),
            nn.Dropout(self.hyper_params["dropout"]),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, hyper_params):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.hyper_params = hyper_params
        self.hyper_params["head_size"] = self.hyper_params["n_embd"] // self.hyper_params["n_head"] 
        self.sa = MultiHeadAttention(self.hyper_params)
        self.ffwd = FeedFoward(self.hyper_params)
        self.ln1 = nn.LayerNorm(self.hyper_params["n_embd"])
        self.ln2 = nn.LayerNorm(self.hyper_params["n_embd"])

    def forward(self, x, attn_mask):
        x = x + self.sa(self.ln1(x),attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder_pos_emb(nn.Module):

    def __init__(self, hyper_params):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.hyper_params = hyper_params
        self.token_embdding_table = nn.Embedding(self.hyper_params["vocab_size"], self.hyper_params["n_embd"])
        self.position_embdding_table = nn.Embedding(self.hyper_params["block_size"], self.hyper_params["n_embd"])
        self.blocks = nn.ModuleList([Block(self.hyper_params) for _ in range(self.hyper_params["n_layer"])])
        self.ln_f = nn.LayerNorm(self.hyper_params["n_embd"]) # final layer norm
        self.lm_head = nn.Linear(self.hyper_params["n_embd"], 2)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.001)

    def forward(self, idx, attn_mask=None, targets=None):
        # idx = idx_in[:,:(d+1)**2+4*d]
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embdding_table(idx) # (B,T,C)
        pos_emb = self.position_embdding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x) # (B,T,C)
        last_token = T +1 - torch.sum( idx == 3, dim = -1)
        x = x[torch.arange(B),last_token,:]
        logits = self.lm_head(x) # (B,vocab_size)
        y = F.sigmoid(logits)

        if targets is None:
            loss = None
        else:
            eps = 1e-9
            loss = -torch.sum((targets*torch.log(y+eps)+(1-targets)*torch.log(1-y+eps)).mean(dim=0))
        return y, loss


class Encoder(nn.Module):

    def __init__(self, hyper_params):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.hyper_params = hyper_params
        self.token_embdding_table = nn.Embedding(self.hyper_params["vocab_size"], self.hyper_params["n_embd"])
        # self.position_embdding_table = nn.Embedding(self.hyper_params["block_size"], self.hyper_params["n_embd"])
        self.blocks = nn.ModuleList([Block(self.hyper_params) for _ in range(self.hyper_params["n_layer"])])
        self.ln_f = nn.LayerNorm(self.hyper_params["n_embd"]) # final layer norm
        self.lm_head = nn.Linear(self.hyper_params["n_embd"], 2)
        self.pos = PositionalEncoding(self.hyper_params["n_embd"], self.hyper_params["block_size"])

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.001)

    def forward(self, idx, attn_mask=None, targets=None):
        # idx = idx_in[:,:(d+1)**2+4*d]
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embdding_table(idx) # (B,T,C)
        # pos_emb = self.position_embdding_table(torch.arange(T, device=idx.device)) # (T,C)

        x = self.pos(tok_emb) # (B,T,C)
        # x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x) # (B,T,C)
        last_token = T +1 - torch.sum( idx == 3, dim = -1)
        x = x[torch.arange(B),last_token,:]
        logits = self.lm_head(x) # (B,vocab_size)
        y = F.sigmoid(logits)

        if targets is None:
            loss = None
        else:
            eps = 1e-9
            loss = -torch.sum((targets*torch.log(y+eps)+(1-targets)*torch.log(1-y+eps)).mean(dim=0))
        return y, loss


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False).to(x.device)
        return x