import warnings

import torch
import torch.nn
import torch.utils.data
from transformers import AutoConfig

from quiet_star.config import Config
from quiet_star.torch.pretrained import PretrainedThoughtModel
from quiet_star.torch.utils import expand_dims


class QwenThoughtModel(PretrainedThoughtModel):
    def __init__(self, config: Config):
        pretrained_config = AutoConfig.from_pretrained(config.model.model_name)

        if config.model.max_length > pretrained_config.max_position_embeddings:
            warnings.warn(
                f"max_length was set to {config.model.max_length} which is "
                f"greater than the context window supported by the Qwen model "
                f"({pretrained_config.max_position_embeddings})"
            )
            config.model.max_length = pretrained_config.max_position_embeddings

        super().__init__(config)

        modules = dict(self.model.named_modules())

        assert (
            pretrained_config.num_key_value_heads
            == pretrained_config.num_attention_heads
        )
        self.num_heads = pretrained_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.layers = torch.nn.ModuleList(modules["model.layers"])

        self.ln = modules["model.norm"]

    def forward(
        self, x: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This function could be made even simpler but it seems
        # like it would probably be less efficient because it saves
        # and passes back *all* hidden states, not just the final one.
        #
        # result = self.model(x, output_attentions=False, output_hidden_states=True)
        # if return_hidden_state:
        #     return result.logits, result.hidden_states[-1]
        # return result.logits

        b, l = x.shape

        causal_mask = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).tile((b, 1, 1, 1))

        x = self.tok_emb(x)
        for layer in self.layers:
            x = layer(x, attention_mask=causal_mask)[
                0
            ]  # the layers return a tuple with a single element
        h = self.ln(x)
        logits = self.lm_head(h)

        if return_hidden_state:
            return logits, h
        return logits

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
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        cos = cos[position_ids]
        sin = sin[position_ids]
        x_embed = (x * cos) + (cls.rotate_half(x) * sin)
        return x_embed

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, L, M = 1 + T + 2 + D)
        b, l, m = x.shape

        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = expand_dims(causal_mask1, (0, 1, 3))
        causal_mask2 = torch.triu(
            torch.full(
                (m, m - 1), float("-inf"), dtype=self._dtype, device=self.device
            ),
            diagonal=0,
        )
        causal_mask2 = expand_dims(causal_mask2, (0, 1, 2))

        causal_mask1 = causal_mask1.repeat(
            b, self.num_heads, 1, causal_mask2.size(3), 1
        )
        causal_mask2 = causal_mask2.repeat(
            b, self.num_heads, causal_mask1.size(2), 1, 1
        )
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .reshape(b, self.num_heads * l, causal_mask1.size(3), -1)
            .contiguous()
        )

        row = torch.arange(0, m, dtype=torch.int64, device=self.device).reshape(1, m)
        offset = torch.arange(0, l, dtype=torch.int64, device=self.device).reshape(l, 1)
        position_ids = (
            (row + offset)
            .reshape((1, l, m))
            .unsqueeze(1)
            .tile((b, self.num_heads, 1, 1))
            .reshape(b, self.num_heads * l, m)
            .contiguous()
        )

        x = self.tok_emb(x)

        for layer in self.layers:
            residual = x
            x = layer.input_layernorm(x)

            q = (
                self.bfloat_safe_apply(layer.self_attn.q_proj, x)
                .reshape(b, l, m, self.num_heads, -1)
                .permute([0, 3, 1, 2, 4])
                .reshape(b, self.num_heads * l, m, -1)
            )
            k1 = (
                self.bfloat_safe_apply(layer.self_attn.k_proj, x[:, :, :1, :])
                .reshape(b, l, 1, self.num_heads, self.head_dim)
                .permute([0, 3, 1, 2, 4])
                .reshape(b, self.num_heads * l, 1, self.head_dim)
                .repeat(1, 1, l, 1)
            )
            k2 = (
                self.bfloat_safe_apply(layer.self_attn.k_proj, x[:, :, 1:, :])
                .reshape(b, l, m - 1, self.num_heads, self.head_dim)
                .permute([0, 3, 1, 2, 4])
                .reshape(b, self.num_heads * l, m - 1, self.head_dim)
            )
            v1 = (
                self.bfloat_safe_apply(layer.self_attn.v_proj, x[:, :, :1, :])
                .reshape(b, l, 1, self.num_heads, self.head_dim)
                .permute([0, 3, 2, 1, 4])
                .repeat(1, 1, l, 1, 1)
                .reshape(b, self.num_heads * l, l, self.head_dim)
            )
            v2 = (
                self.bfloat_safe_apply(layer.self_attn.v_proj, x[:, :, 1:, :])
                .reshape(b, l, m - 1, self.num_heads, self.head_dim)
                .permute([0, 3, 1, 2, 4])
                .reshape(b, self.num_heads * l, m - 1, self.head_dim)
            )

            # apply rotary embedding
            cos, sin = layer.self_attn.rotary_emb(v1, seq_len=l + m)
            q = self.apply_rotary_pos_emb(q, cos, sin, position_ids)
            k1 = self.apply_rotary_pos_emb(k1, cos, sin, position_ids[:, :, :1])
            k2 = self.apply_rotary_pos_emb(k2, cos, sin, position_ids[:, :, 1:])

            k1 = (
                k1.reshape(b, self.num_heads, l, l, self.head_dim)
                .transpose(2, 3)
                .reshape(b, self.num_heads * l, l, self.head_dim)
            )

            k = torch.concatenate([k1, k2], dim=-2)
            v = torch.concatenate([v1, v2], dim=-2)

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask
            )

            attn_out = (
                attn_out.reshape(b, self.num_heads, l, m, self.head_dim)
                .permute([0, 2, 3, 1, 4])
                .reshape(b, l, m, self.embed_dim)
            )
            attn_out = self.bfloat_safe_apply(layer.self_attn.o_proj, attn_out)
            x = residual + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # (B, L, M, E)
        h = self.ln(x)

        # only keep the hidden states which we care about
        h = h[:, :, -(self.lookahead_tokens + 1) : -1]

        return h

    def bfloat_safe_apply(
        self, layer: torch.nn.Linear, x: torch.Tensor
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            return layer(x)
        shape = x.shape
        return layer(x.reshape(-1, shape[-1])).reshape(shape)

    def generate_next_thought_token(
        self,
        x: torch.Tensor,
        t: int,
        activation_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """
        Generate one new thought token at every position in the sequences
        of `x` given that there are already `t` thought tokens.

        This currently requires `x` to have no padding tokens.
        """
        b, l, d = x.shape

        if activation_cache is None:
            activation_cache = [{} for _ in range(len(self.layers))]

        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = expand_dims(causal_mask1, (0, 1, 3))
        causal_mask2 = torch.triu(
            torch.full(
                (t + 1, t + 1), float("-inf"), dtype=self._dtype, device=self.device
            ),
            diagonal=1,
        )
        causal_mask2 = expand_dims(causal_mask2[t - d + 1 :, 1:], (0, 1, 2))

        causal_mask1 = causal_mask1.repeat(
            b, self.num_heads, 1, causal_mask2.size(3), 1
        )
        causal_mask2 = causal_mask2.repeat(
            b, self.num_heads, causal_mask1.size(2), 1, 1
        )
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .reshape(b, self.num_heads * l, causal_mask1.size(3), -1)
            .contiguous()
        )

        rows = torch.arange(0, d, dtype=torch.int64, device=self.device).reshape(1, d)
        row_offsets = torch.arange(
            t - d + 1, l + t - d + 1, dtype=torch.int64, device=self.device
        ).reshape(l, 1)
        position_ids = (
            (rows + row_offsets)
            .reshape((1, l, d))
            .unsqueeze(1)
            .tile((b, self.num_heads, 1, 1))
            .reshape(b, self.num_heads * l, d)
            .contiguous()
        )

        x = self.tok_emb(x)

        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.input_layernorm(x)
            cos, sin = layer.self_attn.rotary_emb(x, seq_len=l + t + 1)
            if activation_cache[i]:
                q = (
                    self.bfloat_safe_apply(layer.self_attn.q_proj, x)
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, 1, -1)
                )
                k2 = (
                    self.bfloat_safe_apply(layer.self_attn.k_proj, x)
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, 1, -1)
                )
                v2 = (
                    self.bfloat_safe_apply(layer.self_attn.v_proj, x)
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, 1, -1)
                )
                q = self.apply_rotary_pos_emb(q, cos, sin, position_ids)
                k2 = self.apply_rotary_pos_emb(k2, cos, sin, position_ids)
                k = torch.concatenate([activation_cache[i]["k"], k2], dim=-2)
                v = torch.concatenate([activation_cache[i]["v"], v2], dim=-2)
            else:
                q = (
                    self.bfloat_safe_apply(layer.self_attn.q_proj, x)
                    .reshape(b, l, d, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, d, -1)
                )
                k1 = (
                    self.bfloat_safe_apply(layer.self_attn.k_proj, x[:, :, :1, :])
                    .reshape(b, l, 1, self.num_heads, self.head_dim)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, 1, self.head_dim)
                    .repeat(1, 1, l, 1)
                )
                k2 = (
                    self.bfloat_safe_apply(layer.self_attn.k_proj, x[:, :, 1:, :])
                    .reshape(b, l, d - 1, self.num_heads, self.head_dim)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, d - 1, self.head_dim)
                )
                v1 = (
                    self.bfloat_safe_apply(layer.self_attn.v_proj, x[:, :, :1, :])
                    .reshape(b, l, 1, self.num_heads, self.head_dim)
                    .permute([0, 3, 2, 1, 4])
                    .repeat(1, 1, l, 1, 1)
                    .reshape(b, self.num_heads * l, l, self.head_dim)
                )
                v2 = (
                    self.bfloat_safe_apply(layer.self_attn.v_proj, x[:, :, 1:, :])
                    .reshape(b, l, d - 1, self.num_heads, self.head_dim)
                    .permute([0, 3, 1, 2, 4])
                    .reshape(b, self.num_heads * l, d - 1, self.head_dim)
                )
                q = self.apply_rotary_pos_emb(q, cos, sin, position_ids)
                k1 = self.apply_rotary_pos_emb(k1, cos, sin, position_ids[:, :, :1])
                k2 = self.apply_rotary_pos_emb(k2, cos, sin, position_ids[:, :, 1:])

                k1 = (
                    k1.reshape(b, self.num_heads, l, l, self.head_dim)
                    .transpose(2, 3)
                    .reshape(b, self.num_heads * l, l, self.head_dim)
                )

                k = torch.concatenate([k1, k2], dim=-2)
                v = torch.concatenate([v1, v2], dim=-2)

            # cache activations for future forward passes
            activation_cache[i]["k"] = k
            activation_cache[i]["v"] = v

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask
            )

            attn_out = (
                attn_out.reshape(b, self.num_heads, l, d, self.head_dim)
                .permute([0, 2, 3, 1, 4])
                .reshape(b, l, d, self.embed_dim)
            )
            attn_out = self.bfloat_safe_apply(layer.self_attn.o_proj, attn_out)
            x = residual + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        h = self.ln(x)
        logits = self.lm_head(h)

        assert activation_cache is not None

        # (B, L, D, vocab_size)
        return logits, activation_cache

    def configure_optimizers(self) -> torch.optim.Optimizer:
        decay = []
        no_decay = []
        for name, params in self.named_parameters():
            if hasattr(params, "requires_grad") and not params.requires_grad:
                continue
            elif ("mlp" in name or "self_attn" in name) and "weight" in name:
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
