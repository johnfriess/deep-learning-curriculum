# decoder only transformer
import torch
import re
from jaxtyping import Int, Float

class Attention(torch.nn.Module):
    def __init__(self, d_model=64, d_k=16, d_v=16, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = torch.nn.Parameter(torch.randn(n_heads, d_model, d_k))
        self.w_k = torch.nn.Parameter(torch.randn(n_heads, d_model, d_k))
        self.w_v = torch.nn.Parameter(torch.randn(n_heads, d_model, d_v))
        self.w_o = torch.nn.Parameter(torch.randn(n_heads * d_v, d_model))

    def forward(self, q: Float[torch.Tensor, "token d_model"], k: Float[torch.Tensor, "token d_model"], v: Float[torch.Tensor, "token d_model"]) -> Float[torch.Tensor, "token d_model"]:
        n, d_model = q.shape
        q = q @ self.w_q
        k = k @ self.w_k
        v = v @ self.w_v
        logits = (q @ k.transpose(1, 2)) / (self.d_k ** 0.5)
        logits = self.mask(logits)
        o = torch.softmax(logits, dim=2) @ v
        o = o.permute(1, 0, 2)
        return o.reshape(n, self.n_heads * self.d_v) @ self.w_o

    def mask(self, x: Float[torch.Tensor, "n_head token token"]) -> Float[torch.Tensor, "n_head token token"]:
        n_heads, n, n = x.shape
        i, j = torch.meshgrid(
            torch.arange(n),
            torch.arange(n),
            indexing="ij"
        )
        return torch.where(j <= i, x, -1e9)

class FFN(torch.nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_model)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(d_model, d_model)

    def forward(self, x: Float[torch.Tensor, "d_model"]) -> Float[torch.Tensor, "d_model"]:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, n_layers=1, d_model=64, d_k=16, d_v=16, n_heads=1, vocab_size=5000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = torch.nn.Parameter(torch.randn(vocab_size, d_model))
        self.attention = Attention(d_model, d_k, d_v, n_heads)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x: Int[torch.Tensor, "token"]) -> Float[torch.Tensor, "token d_model"]:
        x = self.embedding[x]
        x = x + self.pos_embedding(x)

        x = self.ln1(x + self.attention(x, x, x))
        x = self.ln2(x + self.ffn(x))
        x = self.linear(x)
        return x

    def pos_embedding(self, x: Float[torch.Tensor, "token d_model"]) -> Float[torch.Tensor, "token d_model"]:
        n, d_model = x.shape
        i, j = torch.meshgrid(
            torch.arange(n),
            torch.arange(d_model),
            indexing="ij"
        )
        pos_embedding = torch.where(j % 2 == 0, torch.sin(i / torch.pow(10000, j/d_model)), torch.cos(i / torch.pow(10000, (j-1)/d_model)))
        return pos_embedding


PAD = "<pad>"
UNK = "<unk>"

def tokenize(text: str):
    tokens = re.split(r"\b", text)
    tokens = [t for t in tokens if t != "" and not t.isspace()]
    return tokens

def build_vocab(tokens, min_freq=1):
    cnt = dict()
    for token in tokens:
        cnt[token] = cnt.get(token, 0) + 1
    
    vocab = [PAD, UNK] + [t for t, c in cnt.items() if c >= min_freq and t not in (PAD, UNK)]
    stoi = {t:i for i,t in enumerate(vocab)}
    itos = {i:t for t,i in stoi.items()}
    return stoi, itos

def numericalize(tokens, stoi):
    unk_id = stoi[UNK]
    return [stoi.get(t, unk_id) for t in tokens]

with open('shakespeare.txt', 'r') as f:
    content = f.read()

tokens = tokenize(content)
stoi, itos = build_vocab(tokens)
vocab_size = len(stoi)

model = Transformer(d_model=32, d_k=12, d_v=12, n_heads=4, vocab_size=vocab_size)
optimizer = torch.optim.Adam(model.parameters())
seq_len = 32
for epoch in range(1):
    print(f"training epoch {epoch}")
    for i in range(0, len(tokens), seq_len):
        print(f"training seq start {i}")
        start, end = i, i+seq_len
        if end > len(tokens):
            continue
        seq = torch.as_tensor(numericalize(tokens[start:end], stoi))
        x = seq[:-1]
        targets = seq[1:]
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    # test auto-regressiveness
    for _ in range(1):
        start = ["In", "singleness", "the", "parts", "that", "thou"]
        for _ in range(1):
            x = torch.as_tensor(numericalize(start, stoi))
            logits = model(x)
            probs = torch.softmax(logits[-1], dim=-1)
            topk = torch.topk(probs, 20)
            print([(itos[i.item()], float(p)) for i, p in zip(topk.indices, topk.values)])
            next_word = itos[torch.multinomial(probs, 1).item()]
            start.append(next_word)
        print(start)

    # test predictions on sequences
    for i in range(1000, 1100, seq_len):
        start, end = i, i+seq_len
        if end > len(tokens):
            continue

        x = torch.as_tensor(numericalize(tokens[start:end], stoi))
        logits = model(x)
        probs = torch.softmax(logits[-1], dim=-1)
        next_word = itos[torch.multinomial(probs, 1).item()]
        print(f"sequence: {" ".join(tokens[start:end])}, predicted: {next_word}")