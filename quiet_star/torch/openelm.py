import math
import warnings

import torch
from transformers import AutoConfig

from quiet_star.config import Config
from quiet_star.torch.pretrained import PretrainedThoughtModel
from quiet_star.torch.utils import expand_dims


class OpenELMThoughtModel(PretrainedThoughtModel):
    def __init__(self, config: Config):
        pretrained_config = AutoConfig.from_pretrained(
            config.model.model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
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
        unsqueeze: bool = False,
    ) -> torch.Tensor:
        cos = cos[position_ids]
        sin = sin[position_ids]
        if unsqueeze:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
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
            q = self.apply_rotary_pos_emb(q, cos, sin, position_ids, unsqueeze=True)
            k = self.apply_rotary_pos_emb(k, cos, sin, position_ids, unsqueeze=True)

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
        result = self.model(x, output_attentions=False, output_hidden_states=True)
        if return_hidden_state:
            return result.logits, result.hidden_states[-1]
        return result.logits

    def hidden_states_if_attn_mask_were_preserved(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        This implementation shows how to implement the function with only
        calls to the standard model interface.

        OpenELM does not respect the attention mask we pass it, so this
        implementation will not work, but we keep it around in case we
        eventually move to a different model that does respect the mask.
        """
        # x is (B, L, M = 1 + T + 2 + D)
        b, l, m = x.shape

        x_init = x[:, :, :1].transpose(1, 2).reshape(b, l)

        init_causal_mask = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        init_causal_mask = expand_dims(init_causal_mask, (0, 1)).repeat(b, 1, 1, 1)
        init_position_ids = (
            torch.arange(0, l, dtype=torch.int64, device=self.device)
            .reshape(1, l)
            .repeat(b, 1)
        )

        result = self.model(
            x_init,
            attention_mask=init_causal_mask,
            position_ids=init_position_ids,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )

        past_key_values = tuple(
            (key.tile(l, 1, 1, 1), value.tile(l, 1, 1, 1))
            for key, value in result.past_key_values
        )

        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = expand_dims(causal_mask1, (0, 1, 3))
        causal_mask2 = torch.triu(
            torch.full(
                (m - 1, m - 1), float("-inf"), dtype=self._dtype, device=self.device
            ),
            diagonal=1,
        )
        causal_mask2 = expand_dims(causal_mask2, (0, 1, 2))

        causal_mask1 = causal_mask1.repeat(b, 1, 1, causal_mask2.size(3), 1)
        causal_mask2 = causal_mask2.repeat(b, 1, causal_mask1.size(2), 1, 1)
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .transpose(1, 2)
            .reshape(b * l, 1, causal_mask1.size(3), -1)
            .contiguous()
        )

        row0 = torch.arange(0, l, dtype=torch.int64, device=self.device).reshape(1, l)
        position_ids0 = row0.tile((b * l, 1)).contiguous()
        row = torch.arange(1, m, dtype=torch.int64, device=self.device).reshape(
            1, m - 1
        )
        offset = torch.arange(0, l, dtype=torch.int64, device=self.device).reshape(l, 1)
        position_ids = (
            (row + offset)
            .reshape((1, l, m - 1))
            .tile((b, 1, 1))
            .reshape(b * l, m - 1)
            .contiguous()
        )
        position_ids = torch.concatenate([position_ids0, position_ids], dim=-1)

        x2 = x[:, :, 1:].reshape(b * l, m - 1)

        result = self.model(
            x2,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            output_hidden_states=True,
        )

        h = result.hidden_states[-1]
        h = h.reshape(b, l, h.size(1), h.size(2))
        h = h[:, :, -(self.lookahead_tokens + 1) : -1]

        return h

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

        causal_mask1 = causal_mask1.repeat(b, 1, 1, causal_mask2.size(3), 1)
        causal_mask2 = causal_mask2.repeat(b, 1, causal_mask1.size(2), 1, 1)
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .swapaxes(1, 2)
            .reshape(
                b * l,
                1,
                causal_mask1.size(3),
                causal_mask1.size(4) + causal_mask2.size(4),
            )
            .contiguous()
        )

        row = torch.arange(0, m, dtype=torch.int64, device=self.device).reshape(1, m)
        offset = torch.arange(0, l, dtype=torch.int64, device=self.device).reshape(l, 1)
        position_ids = (
            (row + offset)
            .reshape(1, l, m)
            .tile((b, 1, 1))
            .reshape(b * l, 1, m)
            .contiguous()
        )

        x = x.reshape(b * l, m)
        x = self.tok_emb(x)

        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.attn_norm(x)

            qkv = (
                layer.attn.qkv_proj(x)
                .reshape(
                    b * l,
                    m,
                    2 * self.num_kv_heads[i] + self.num_query_heads[i],
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
            )
            q, k, v = qkv.split(
                [self.num_query_heads[i], self.num_kv_heads[i], self.num_kv_heads[i]],
                dim=1,
            )

            q = layer.attn.q_norm(q)
            k = layer.attn.k_norm(k)

            q = q.reshape(b * l, self.num_query_heads[i], m, self.head_dim)
            k1 = (
                k[:, :, :1]
                .reshape(b * l, self.num_kv_heads[i], 1, self.head_dim)
                .repeat(1, 1, l, 1)
            )
            k2 = k[:, :, 1:].reshape(b * l, self.num_kv_heads[i], m - 1, self.head_dim)
            v1 = (
                v[:, :, :1]
                .reshape(b, l, self.num_kv_heads[i], 1, self.head_dim)
                .swapaxes(1, 3)
                .repeat(1, l, 1, 1, 1)
                .reshape(b * l, self.num_kv_heads[i], l, self.head_dim)
            )
            v2 = v[:, :, 1:].reshape(b * l, self.num_kv_heads[i], m - 1, self.head_dim)

            # apply rotary embedding
            cos, sin = self.rotary_emb(v1, seq_len=l + m)
            q_position_ids = position_ids.tile(1, self.num_query_heads[i], 1).reshape(
                b * l, self.num_query_heads[i], m
            )
            k_position_ids = position_ids.tile(1, self.num_kv_heads[i], 1).reshape(
                b * l, self.num_kv_heads[i], m
            )
            if i == 0:
                print("q_position_ids:", q_position_ids.shape, q_position_ids)
            q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
            k1 = self.apply_rotary_pos_emb(k1, cos, sin, k_position_ids[:, :, :1])
            k2 = self.apply_rotary_pos_emb(k2, cos, sin, k_position_ids[:, :, 1:])

            k1 = (
                k1.reshape(b, l, self.num_kv_heads[i], l, self.head_dim)
                .transpose(1, 3)
                .reshape(b * l, self.num_kv_heads[i], l, self.head_dim)
            )

            k = torch.concatenate([k1, k2], dim=-2)
            v = torch.concatenate([v1, v2], dim=-2)

            k = k.repeat_interleave(self.num_gqa_groups, dim=1)
            v = v.repeat_interleave(self.num_gqa_groups, dim=1)

            layer_causal_mask = causal_mask.repeat(
                1, self.num_kv_heads[i], 1, 1
            ).repeat_interleave(self.num_gqa_groups, dim=1)
            a = torch.nn.functional.softmax(
                (torch.matmul(q, k.transpose(-2, -1)) + layer_causal_mask)
                / math.sqrt(q.size(-1)),
                dim=-1,
            )

            attn_out = torch.matmul(a, v)
            attn_out = layer.attn.out_proj(
                attn_out.reshape(b * l, self.num_query_heads[i], m, self.head_dim)
                .permute([0, 2, 1, 3])
                .reshape(b * l, m, self.num_query_heads[i] * self.head_dim)
            )
            x = residual + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

        # (B * L, M, E)
        h = self.ln(x)

        h = h.reshape(b, l, m, -1)

        # only keep the hidden states which we care about
        h = h[:, :, -(self.lookahead_tokens + 1) : -1]

        return h

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
