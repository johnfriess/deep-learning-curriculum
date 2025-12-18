# decoder only transformer
import torch
from jaxtyping import Int, Float

class Attention(torch.nn.Module):
    def __init__(self, d_model=64, d_k=16, d_v=16, n_heads=1):
        self.d_k = d_k
        self.w_q = torch.nn.Parameter(torch.randn(d_model, d_k))
        self.w_k = torch.nn.Parameter(torch.randn(d_model, d_k))
        self.w_v = torch.nn.Parameter(torch.randn(d_model, d_v))
        self.w_o = torch.nn.Parameter(torch.randn(d_v, d_model))

    def forward(self, q: Float[torch.Tensor, "token d_model"], k: Float[torch.Tensor, "token d_model"], v: Float[torch.Tensor, "token d_model"]) -> Float[torch.Tensor, "token d_model"]:
        q = q @ self.w_q
        k = k @ self.w_k
        v = v @ self.w_v
        logits = (q @ k.t()) / (self.d_k ** 0.5)
        logits = self.mask(logits)
        o = torch.softmax(logits, dim=1) @ v
        return o @ self.w_o

    def mask(self, x: Float[torch.Tensor, "token token"]) -> Float[torch.Tensor, "token token"]:
        n, n = x.shape
        i, j = torch.meshgrid(
            torch.arange(n),
            torch.arange(n),
            indexing="ij"
        )
        return torch.where(j <= i, x, -1e9)

class FFN(torch.nn.Module):
    def __init__(self, d_model=64):
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
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = torch.nn.Parameter(torch.randn(vocab_size, d_model))
        self.attention = Attention(d_model, d_k, d_v, n_heads)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: Int[torch.Tensor, "token"]) -> Float[torch.Tensor, "token d_model"]:
        x = self.embedding[x]
        x = x + self.pos_embedding(x)

        x = self.ln1(x + self.attention(x, x, x))
        x = self.ln2(x + self.ffn(x))
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


transformer = Transformer()
# print(transformer.pos_embedding(torch.as_tensor([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [5, 6, 7, 8]])))
q = torch.as_tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
# v = torch.as_tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
print(transformer.attention(q, q, q))