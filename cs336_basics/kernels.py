from torch import Tensor
from jaxtyping import Float, Int
import torch

# def linear(
#     d_in: int,
#     d_out: int,
#     weights: Float[Tensor, " d_out d_in"],
#     in_features: Float[Tensor, " ... d_in"],
# ) -> Float[Tensor, " ... d_out"]:
    
#     return in_features @ weights.T

class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, 
                 weights: Float[Tensor, " d_out d_in"]):
        super().__init__()
        self.data = weights

    def forward(self, in_features: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return torch.matmul(in_features, self.data.T)

# def embedding(
#     vocab_size: int,
#     d_model: int,
#     weights: Float[Tensor, " vocab_size d_model"],
#     token_ids: Int[Tensor, "..."]
# ) -> Float[Tensor, "... d_model"]:
#     return weights[token_ids]

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, 
                 weights: Float[Tensor, " vocab_size d_model"]):
        super().__init__()
        self.data = weights

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.data[token_ids]

# def swiglu(
#     d_model: int,
#     d_ff: int,
#     w1_weight: Float[Tensor, " d_ff d_model"],
#     w2_weight: Float[Tensor, " d_model d_ff"],
#     w3_weight: Float[Tensor, " d_ff d_model"],
#     in_features: Float[Tensor, "... d_model"],
# ) -> Float[Tensor, "... d_model"]:
    
class SwigLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, 
                 w1_weight: Float[Tensor, " d_ff d_model"], 
                 w2_weight: Float[Tensor, " d_model d_ff"], 
                 w3_weight: Float[Tensor, " d_ff d_model"]):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, w1_weight)
        self.w2 = Linear(d_model, d_ff, w2_weight)
        self.w3 = Linear(d_ff, d_model, w3_weight)

    def forward(self, in_features: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x1 = self.w1(in_features)
        x3 = self.w3(in_features)
        silu_x1 = x1 / (1 + torch.exp(-x1))
        return self.w2(silu_x1 * x3)
    
def ScaledDotProductAttention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Float[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = scores / (K.shape[-1] ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 q_proj_weight: Float[Tensor, " d_k d_in"],
                 k_proj_weight: Float[Tensor, " d_k d_in"],
                 v_proj_weight: Float[Tensor, " d_v d_in"],
                 o_proj_weight: Float[Tensor, " d_model d_v"]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = q_proj_weight.shape[-2] // num_heads
        self.d_v = v_proj_weight.shape[-2] // num_heads
        self.q_proj = Linear(self.d_k * num_heads, self.d_model, q_proj_weight)
        self.k_proj = Linear(self.d_k * num_heads, self.d_model, k_proj_weight)
        self.v_proj = Linear(self.d_v * num_heads, self.d_model, v_proj_weight)
        self.o_proj = Linear(self.d_model, self.d_v * num_heads, o_proj_weight)

    def forward(self, 
                in_features: Float[Tensor, "... sequence_length d_in"]) -> Float[Tensor, "... sequence_length d_out"]:

        Q = self.q_proj(in_features).view(*in_features.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        K = self.k_proj(in_features).view(*in_features.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        V = self.v_proj(in_features).view(*in_features.shape[:-1], self.num_heads, -1).transpose(-2, -3)

        mask = ~torch.triu(torch.ones(in_features.shape[-2], in_features.shape[-2]), diagonal=1).bool()

        out = ScaledDotProductAttention(Q, K, V, mask).transpose(-2, -3).reshape(*in_features.shape[:-1], -1)

        return self.o_proj(out)