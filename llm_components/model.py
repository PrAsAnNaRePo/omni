import math
import torch
from torch import nn
import torch.nn.functional as F


# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim: int, head_dim: int, max_seq_len: int):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.head_dim = head_dim
#         self.max_seq_len = max_seq_len

#         self.qkv = nn.Linear(embed_dim, 3 * head_dim, bias=False)

#         self.out = nn.Linear(head_dim, embed_dim)
#         self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))

#     def forward(self, x: torch.Tensor):
#         b, t, c = x.shape

#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#         print(q.shape, k.transpose(-2, -1).shape)
#         wei = q @ k.transpose(-2, -1) * (c ** -0.5)
#         wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
#         wei = F.softmax(wei, dim=-1)
#         out = wei @ v
#         out = self.out(out)
#         return out

class MHA(nn.Module):
    def __init__(self, num_heads, head_dim, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.qkv_mh = nn.Linear(embed_dim, head_dim*3*num_heads, bias=False)
        self.ln = nn.Linear(self.num_heads*self.head_dim, self.embed_dim)

    def forward(self, x: torch.Tensor):
        bs, seq_len, d_model = x.size()
        x = self.qkv_mh(x) # (bs, seq_len, head_dim*3*num_heads)
        x = x.view(bs, seq_len, self.num_heads, 3, self.head_dim)
        q, k, v = x[:, :, :, 0, :], x[:, :, :, 1, :], x[:, :, :, 2, :]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1) @ v # (bs, num_head, seq_len, head_dim)
        attn = attn.view(bs, seq_len, self.num_heads*self.head_dim)

        out = self.ln(attn)
        return out

class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            max_seq_len: int,
            num_heads: int,
            head_dim: int
    ):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)
        # self.sa = SelfAttention(embed_dim, head_dim, max_seq_len)
        self.mha = MHA(num_heads, head_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.shape
        token_embeddings = self.wte(idx)
        position_embeddings = self.wpe(torch.arange(t, device=idx.device))
        x = token_embeddings + position_embeddings
        print("x: ", x.shape)
        x = self.mha(x)
        # x = self.sa(x)
        x = self.ln1(x)
        x = self.mlp(x)
        x = self.ln2(x)
        logits = self.lm_head(x)
        return logits

gpt = GPT(
    vocab_size=32_000,
    embed_dim=768,
    max_seq_len=2048,
    num_heads=8,
    head_dim=64
)
input = torch.randint(0, 10, (1, 10))

output = gpt(input)
print(output.shape)

