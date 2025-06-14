import math
import torch
from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, max_seq_len, attn_dropout, ff_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.mha = MHA(num_heads, head_dim, embed_dim, max_seq_len, attn_dropout)
        self.ln = FeedForward(embed_dim, ff_dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x + self.mha(self.layer_norm1(x))
        x = x + self.ln(self.layer_norm2(x))
        return x

class MHA(nn.Module):
    def __init__(self, num_heads, head_dim, embed_dim, max_seq_len, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.qkv_mh = nn.Linear(embed_dim, head_dim*3*num_heads, bias=False)
        self.proj = nn.Linear(self.num_heads*self.head_dim, self.embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x: torch.Tensor, return_attn=False):
        bs, seq_len, d_model = x.size()
        x = self.qkv_mh(x) # (bs, seq_len, head_dim*3*num_heads)
        x = x.view(bs, seq_len, self.num_heads, 3, self.head_dim)
        q, k, v = x[:, :, :, 0, :], x[:, :, :, 1, :], x[:, :, :, 2, :]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) # (bs, num_heads, seq_len, seq_len)
        attn = attn.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1) # (bs, num_head, seq_len, head_dim)
        out = attn @ v
        out = out.view(bs, seq_len, self.num_heads*self.head_dim)

        out = self.dropout(self.proj(out))
        if return_attn:
            return out, attn
        return out

class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            max_seq_len: int,
            num_heads: int,
            head_dim: int,
            num_layers: int,
            attn_dropout: float,
            ff_dropout: float
    ):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([ TransformerBlock(embed_dim, num_heads, head_dim, max_seq_len, attn_dropout, ff_dropout) for i in range(num_layers) ])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.shape
        token_embeddings = self.wte(idx)
        position_embeddings = self.wpe(torch.arange(t, device=idx.device))
        x = token_embeddings + position_embeddings
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits

gpt = GPT(
    vocab_size=32_000,
    embed_dim=768,
    max_seq_len=2048,
    num_heads=8,
    head_dim=64,
    num_layers=2,
    attn_dropout=0.2,
    ff_dropout=0.2
)

print(gpt)

input = torch.randint(0, 10, (1, 10))

output = gpt(input)
print(output.shape)

