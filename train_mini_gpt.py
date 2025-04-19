
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle

text = "hello world. hello AI. hello future. welcome to the world of AI. Capital of China is Beijing"
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
block_size = 8
batch_size = 4

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.ff(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_encoder = PositionalEncoding(n_embed)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_head, dropout) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x):
        x = self.token_embed(x)
        x = self.pos_encoder(x)
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        return self.lm_head(x)

n_embed = 64
model = MiniGPT(vocab_size, n_embed, n_head=2, n_layer=2, dropout=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(300):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mini_gpt.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump((stoi, itos), f)
