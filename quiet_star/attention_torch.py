import torch
import torch.nn


class TorchCausalSelfAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_length: int,
        dropout: float = 0.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout, device=device, dtype=dtype
        )
        self.mask = torch.triu(
            torch.full(
                (max_length, max_length), float("-inf"), dtype=dtype, device=device
            ),
            diagonal=1,
        )

    def forward(
        self, x: torch.Tensor, padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, e = x.size()

        y = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding,
            need_weights=False,
            attn_mask=self.mask[:t, :t],
            is_causal=True,
        )[0]

        return y
