import math
import warnings

import torch
from transformers import AutoConfig

from quiet_star.config import Config
from quiet_star.torch.pretrained import PretrainedThoughtModel


class OpenELMThoughtModel(PretrainedThoughtModel):
    def __init__(self, config: Config):
        pretrained_config = AutoConfig.from_pretrained(
            config.model.model_name,
            trust_remote_code=True,
        )

        if config.model.max_length > pretrained_config.max_context_length:
            warnings.warn(
                f"max_length was set to {config.model.max_length} which is "
                f"greater than the context window supported by the Qwen model "
                f"({pretrained_config.max_context_length})"
            )
            config.model.max_length = pretrained_config.max_context_length

        super().__init__(config)

        modules = dict(self.model.named_modules())

        self.num_kv_heads: list[int] = pretrained_config.num_kv_heads
        self.num_query_heads: list[int] = pretrained_config.num_query_heads
        self.num_gqa_groups: int = pretrained_config.num_gqa_groups
        self.head_dim: int = pretrained_config.head_dim

        self.layers = torch.nn.ModuleList(modules["transformer.layers"])

        self.ln = modules["transformer.norm"]

        freq_constant = 10000
        inv_freq = 1.0 / (
            freq_constant
            ** (
                torch.arange(
                    0, self.head_dim, 2, dtype=torch.int64, device=self.device
                ).to(torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=self.max_length, device=self.inv_freq.device, dtype=self.dtype
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def rotary_emb(
        self, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    @classmethod
    def rotate_half(cls, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def apply_rotary_pos_emb(
        cls,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        x_embed = (x * cos) + (cls.rotate_half(x) * sin)
        return x_embed

    def forward_for_testing(
        self, x: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, l = x.shape

        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = causal_mask1.unsqueeze(0)

        row = torch.arange(0, l, dtype=torch.int64, device=self.device)
        position_ids = row.reshape(1, l).tile((b, 1))

        x = self.tok_emb(x)
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.attn_norm(x)

            qkv = (
                layer.attn.qkv_proj(x)
                .reshape(b, l, 2 * self.num_kv_heads[i] + self.num_query_heads[i], -1)
                .swapaxes(1, 2)
            )
            q, k, v = qkv.split(
                [self.num_query_heads[i], self.num_kv_heads[i], self.num_kv_heads[i]],
                dim=1,
            )

            q = layer.attn.q_norm(q)
            k = layer.attn.k_norm(k)

            # apply rotary embedding
            cos, sin = self.rotary_emb(v, seq_len=l)
            q = self.apply_rotary_pos_emb(q, cos, sin, position_ids)
            k = self.apply_rotary_pos_emb(k, cos, sin, position_ids)

            k = k.repeat_interleave(self.num_gqa_groups, dim=1)
            v = v.repeat_interleave(self.num_gqa_groups, dim=1)

            a = torch.nn.functional.softmax(
                (torch.matmul(q, k.transpose(-2, -1)) + causal_mask1)
                / math.sqrt(q.size(-1)),
                dim=-1,
            )

            # attn_out is (B, H, L, E)
            attn_out = torch.matmul(a, v)
            attn_out = layer.attn.out_proj(
                attn_out.permute([0, 2, 1, 3]).reshape(
                    b, l, self.num_query_heads[i] * self.head_dim
                )
            )
            x = residual + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))
        # (B, L, E)
        h = self.ln(x)

        logits = self.lm_head(h)
        if return_hidden_state:
            return logits, h
        return logits

    def forward(
        self, x: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pass

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def generate_next_thought_token(
        self,
        x: torch.Tensor,
        t: int,
        activation_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if activation_cache is None:
            activation_cache = [{} for _ in range(len(self.layers))]

        return x, activation_cache

    def configure_optimizers(self) -> torch.optim.Optimizer:
        decay = []
        no_decay = []
        for name, params in self.named_parameters():
            if hasattr(params, "requires_grad") and not params.requires_grad:
                continue
            elif ("ffn" in name or "attn" in name) and "weight" in name:
                decay.append(params)
            else:
                no_decay.append(params)
        params = [
            {
                "params": decay,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
