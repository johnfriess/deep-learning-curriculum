# decoder only transformer
import torch
import hydra
import json
import re
from jaxtyping import Int, Float
from torch.utils.data import DataLoader

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

    def forward(self, q: Float[torch.Tensor, "batch token d_model"], k: Float[torch.Tensor, "batch token d_model"], v: Float[torch.Tensor, "batch token d_model"]) -> Float[torch.Tensor, "batch token d_model"]:
        B, n, d_model = q.shape
        q = q[:, None, :, :] @ self.w_q[None, :, :, :]
        k = k[:, None, :, :] @ self.w_k[None, :, :, :]
        v = v[:, None, :, :] @ self.w_v[None, :, :, :]
        logits = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        logits = self.mask(logits)
        o = torch.softmax(logits, dim=-1) @ v # B,H,T,T  x B,H,T,d_v => B, H, T, d_v
        o = o.permute(0, 2, 1, 3)
        return o.reshape(B, n, self.n_heads * self.d_v) @ self.w_o # B,T,H*d_v  x H*d_v, d_model => B, T, d_model

    def mask(self, x: Float[torch.Tensor, "batch n_head token token"]) -> Float[torch.Tensor, "batch n_head token token"]:
        B, n_heads, n, n = x.shape
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=x.device))
        return x.masked_fill(~mask, -1e9)

class FFN(torch.nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_model)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(d_model, d_model)

    def forward(self, x: Float[torch.Tensor, "batch d_model"]) -> Float[torch.Tensor, "batch d_model"]:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, n_layers=1, d_model=64, d_k=16, d_v=16, n_heads=1, vocab_size=5000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.attention = Attention(d_model, d_k, d_v, n_heads)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.sentiment = torch.nn.Linear(d_model, 1)

    def forward(self, x: Int[torch.Tensor, "batch token"]) -> tuple[Float[torch.Tensor, "token d_model"], Float[torch.Tensor, "sentiment"]]:
        x = self.embedding(x)
        x = x + self.pos_embedding(x)

        x = self.ln1(x + self.attention(x, x, x))
        x = self.ln2(x + self.ffn(x))
        logits = self.linear(x)
        sentiment = self.sentiment(x[:, -1]).squeeze(-1)
        return logits, sentiment

    def pos_embedding(self, x: Float[torch.Tensor, "batch token d_model"]) -> Float[torch.Tensor, "token d_model"]:
        B, n, d_model = x.shape
        i, j = torch.meshgrid(
            torch.arange(n),
            torch.arange(d_model),
            indexing="ij"
        )
        pos_embedding = torch.where(j % 2 == 0, torch.sin(i / torch.pow(10000, j/d_model)), torch.cos(i / torch.pow(10000, (j-1)/d_model)))
        return pos_embedding
    
class PPO(torch.nn.Module):
    def __init__(self, device=None, obs_dim=4, hidden_dim=64, action_dim=2, epsilon=0.2, c1=0.5, c2=0.01):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.policy = torch.nn.Linear(hidden_dim, action_dim)
        self.value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch hidden_dim"]:
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=self.device)
        if state.dtype != torch.float:
            state = state.float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.dtype == torch.uint8:
            state = state / 255.0
        B = state.shape[0]
        state = state.view(B, -1)
        return self.encoder(state)

    def get_policy(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> Float[torch.Tensor, "batch action_dim"]:
        state = self.forward(state)
        return self.policy(state)

    def get_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."]) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        state = self.forward(state)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=1).clamp_min(1e-8)

        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp = torch.log(probs.gather(dim=1, index=action.unsqueeze(-1))).squeeze(-1)
        value = self.value(state).squeeze(-1)

        return action, logp, value
    
    def evaluate_action(self, state: Float[torch.Tensor, "batch obs_dim1 obs_dim2 ..."], action: Float[torch.Tensor, "batch"]) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        state = self.forward(state)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)

        logp = torch.log(probs.gather(dim=1, index=action.unsqueeze(-1))).squeeze(-1)
        value = self.value(state).squeeze(-1)

        return logp, entropy, value

    def compute_clip_loss(self, logp: Float[torch.Tensor, "t"], logp_old: Float[torch.Tensor, "t"], advantage: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "t"]:
        ratio = torch.exp(logp - logp_old)
        clamped = ((ratio < 1 - self.epsilon).sum() + (ratio > 1 + self.epsilon).sum()).item()
        print(f"percent clipped: {clamped * 100 / ratio.shape[0]}")
        return torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage)
    
    def compute_vf_loss(self, value: Float[torch.Tensor, "t"], advantage: Float[torch.Tensor, "t"], value_old: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "t"]:
        return (value - (advantage + value_old))**2

    def compute_loss(self, clip_loss: Float[torch.Tensor, "t"], vf_loss: Float[torch.Tensor, "t"], entropy: Float[torch.Tensor, "t"]) -> Float[torch.Tensor, "loss"]:
        return -torch.mean(clip_loss - self.c1 * vf_loss + self.c2 * entropy)

PAD = "<pad>"
UNK = "<unk>"

def tokenize(text: str):
    tokens = re.split(r"\b", text)
    tokens = [t for t in tokens if t != "" and not t.isspace()]
    return tokens

def batch_tokenize(texts: str):
    batch_tokens = [re.split(r"\b", text) for text in texts]
    tokens = [[t for t in batch if t != "" and not t.isspace()] for batch in batch_tokens]
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

def batch_numericalize(batch_tokens, stoi):
    unk_id = stoi[UNK]
    return [[stoi.get(t, unk_id) for t in batch] for batch in batch_tokens]

def add_padding(batch_tokens, stoi):
    max_len = max([len(batch) for batch in batch_tokens])
    pad_id = stoi[PAD]
    for batch in batch_tokens:
        while len(batch) < max_len:
            batch.append(pad_id)
    return batch_tokens

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train transformer
    with open(config.transformer_data_path, 'r') as f:
        content = f.read()

    tokens = tokenize(content)
    stoi, itos = build_vocab(tokens)
    vocab_size = len(stoi)

    model = Transformer(d_model=32, d_k=12, d_v=12, n_heads=4, vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    # for epoch in range(config.transformer_epochs):
    #     print(f"training epoch {epoch}")
    #     for i in range(0, len(tokens), config.seq_len):
    #         print(f"training seq start {i}")
    #         start, end = i, i+config.seq_len
    #         if end > len(tokens):
    #             continue
            
    #         seq = torch.as_tensor(numericalize(tokens[start:end], stoi))
    #         inputs = seq[:-1]
    #         targets = seq[1:]

    #         optimizer.zero_grad()
    #         logits = model(inputs)
    #         loss = torch.nn.functional.cross_entropy(logits, targets)
    #         loss.backward()
    #         optimizer.step()

    # train reward model for RLHF
    with open(config.rlhf_data_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    rlhf_data = [[item["sample"], item["sentiment"]] for item in data]
    rlhf_labels = {
        "happy": 1.0,
        "sad": 0.0
    }

    optimizer = torch.optim.Adam(model.sentiment.parameters())
    for epoch in range(config.rlhf_epochs):
        print(f"training reward model epoch {epoch}")
        rlhf_data_loader = DataLoader(rlhf_data, batch_size=config.batch_size, shuffle=True)
        for samples, sentiments in rlhf_data_loader:
            print("run")
            targets = torch.tensor([rlhf_labels[sentiment] for sentiment in sentiments], device=device)
            seq = torch.tensor(add_padding(batch_numericalize(batch_tokenize(samples), stoi), stoi), device=device)

            optimizer.zero_grad()
            _, sentiment_logit = model(seq)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(sentiment_logit, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        samples = [
            "\nThat in the natures of their lords rebel;\nBring oil to fire, snow to their colder moods;\nRenege, affirm, and turn their halcyon beaks\nWith every gale and vary of their masters,\nKnowing naught, like dogs, but following.",
            "\nVengeance! plague! death! confusion!",
            "\nRive your concealing continents, and cry\nThese dreadful summoners grace. I am a man\nMore sinn’d against than sinning."
        ]
        seq = torch.tensor(add_padding(batch_numericalize(batch_tokenize(samples), stoi), stoi), device=device)
        print(seq.shape)
        _, sentiment_logit = model(seq)
        prob = torch.sigmoid(sentiment_logit)
        print(prob)

    exit()
    model.eval()
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

if __name__ == "__main__":
    main()
