import torch
from model import GPT
from dataset import UnsupervisedDataset
from tokenizer.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

data_file = "data.txt"

def train():
    tokenizer = Tokenizer("tokenizer/vocab.bpe", "tokenizer/merges.bpe")
    ds = UnsupervisedDataset(data_file, tokenizer, ctx_lim=350)

    model = GPT(len(tokenizer), embed_dim=512, max_seq_len=350, num_heads=12, head_dim=68, num_layers=4, attn_dropout=0.1, ff_dropout=0.1)

    loader = DataLoader(ds, batch_size=4)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        for idx, (x, y) in enumerate(tqdm(loader)):
            optim.zero_grad()

            loss = model(x, y)[-1]
            loss.backward()
            optim.step()
        
        print(f"Epoch {epoch+1} | Loss {loss.item()}")

if __name__ == '__main__':
    train()