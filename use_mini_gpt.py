import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle

with open("vocab.pkl", "rb") as f:
    stoi, itos = pickle.load(f)
vocab_size = len(stoi)

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

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

block_size = 8
n_embed = 64
n_head = 2
n_layer = 2
model = MiniGPT(vocab_size, n_embed, n_head, n_layer)
model.load_state_dict(torch.load("mini_gpt.pth"))
model.eval()

def generate(model, start_text="hello ", steps=30):
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
    for _ in range(steps):
        logits = model(context[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_id), dim=1)
    return decode(context[0].tolist())

prompt = input("Enter prompt: ")
print("\nGenerated:", generate(model, prompt))



