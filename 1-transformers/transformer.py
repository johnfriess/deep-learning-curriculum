# decoder only transformer
import torch
from jaxtyping import Int, Float

class Transformer(torch.nn.Module):
    def __init__(self, d_model=64, n_heads=1):
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, x: Int[torch.Tensor, "token"]) -> Float[torch.Tensor, "token d_model"]:
        x = x + self.pos_embedding(x)

    def pos_embedding(self, x: Float[torch.Tensor, "token d_model"]) -> Float[torch.Tensor, "token d_model"]:
        n, d_model = x.shape
        i, j = torch.meshgrid(
            torch.arange(n),
            torch.arange(d_model),
            indexing="ij"
        )
        pos_embedding = torch.where(j % 2 == 0, torch.sin(i / torch.pow(10000, j/d_model)), torch.cos(i / torch.pow(10000, (j-1)/d_model)))
        return pos_embedding

    def attention(self, q: Float[torch.Tensor, "token d_k"], k: Float[torch.Tensor, "token d_k"], v: Float[torch.Tensor, "token d_v"]) -> Float[torch.Tensor, "token d_model"]:
        pass

transformer = Transformer()
print(transformer.pos_embedding(torch.as_tensor([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [5, 6, 7, 8]])))