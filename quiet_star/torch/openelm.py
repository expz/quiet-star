import math
import warnings

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from quiet_star.config import Config
from quiet_star.torch.pretrained import ForwardResult, PretrainedThoughtModel
from quiet_star.torch.utils import expand_dims


class OpenELMThoughtModel(PretrainedThoughtModel):
    def __init__(self, config: Config):
        pretrained_config = AutoConfig.from_pretrained(
            config.model.model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        if config.model.train_max_length > pretrained_config.max_context_length:
            warnings.warn(
                f"max_length was set to {config.model.train_max_length} which is "
                f"greater than the context window supported by the Qwen model "
                f"({pretrained_config.max_context_length})"
            )
            config.model.train_max_length = pretrained_config.max_context_length

        super().__init__(config)

        self.save_hyperparameters()  # saves the argument(s) of __init__

        self.eval_max_length = pretrained_config.max_context_length

        self.num_kv_heads: list[int] = pretrained_config.num_kv_heads
        self.num_query_heads: list[int] = pretrained_config.num_query_heads
        self.num_gqa_groups: int = pretrained_config.num_gqa_groups
        self.head_dim: int = pretrained_config.head_dim

        modules = dict(self.model.named_modules())

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
            seq_len=self.train_max_length, device=self.inv_freq.device, dtype=self.dtype
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
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_hidden_state: bool = False,
        return_key_value_cache: bool = False,
    ) -> ForwardResult:
        b, l = x.shape

        true_l = l
        if key_value_cache is not None:
            true_l += key_value_cache[0][0].shape[-2]

        causal_mask = torch.triu(
            torch.full(
                (true_l, true_l), float("-inf"), dtype=self._dtype, device=self.device
            ),
            diagonal=1,
        )[-l:, :]
        causal_mask = (
            causal_mask.unsqueeze(0).unsqueeze(1).repeat([b, 1, 1, 1]).contiguous()
        )

        use_cache = key_value_cache or return_key_value_cache
        past_key_values = (
            DynamicCache.from_legacy_cache(key_value_cache) if use_cache else None
        )

        x = self.tok_emb(x)
        for layer in self.layers:
            result = layer(
                x,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )
            x = result[0]

        if return_key_value_cache:
            new_key_value_cache = result[1]

        h = self.ln(x)
        logits = self.lm_head(h)

        if return_hidden_state:
            if return_key_value_cache:
                return ForwardResult(logits, h, new_key_value_cache)
            return ForwardResult(logits, h, None)
        if return_key_value_cache:
            return ForwardResult(logits, None, new_key_value_cache)
        return ForwardResult(logits, None, None)

    def lookahead_hidden_states(
        self, x: torch.Tensor, key_value_cache: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, A)
                We expect that A == self.lookahead_tokens
            key_value_cache: (list[tuple[torch.Tensor, torch.Tensor]])
                A list of (key, value) pairs for each layer in the model.
                It should cache keys for L + 1 + self.thought_length tokens

        """
        b, l, a = x.shape
        assert a == self.lookahead_tokens

        m = key_value_cache[0][0].size(2)
        assert m == l + 1 + self.thought_length

        # this part of the causal mask will prevent lookahead tokens from
        # attending to later tokens in the original (thought-free) sequence
        # this part of the mask actually governs attention to the tokens
        # represented by the key-value cache
        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = expand_dims(causal_mask1, (0, 1, 3))

        # this part of the causal mask allows all lookahead tokens to attend to
        # all thought tokens (and start thought token)
        causal_mask2 = torch.zeros(
            (b, 1, l, a, 1 + self.thought_length), dtype=self._dtype, device=self.device
        )
        # this part of the causal mask will prevent lookahead tokens from
        # attending to later tokens in the same thought
        causal_mask3 = torch.triu(
            torch.full((a, a), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask3 = expand_dims(causal_mask3, (0, 1, 2))

        # combine the three masks and absorb the sequence length dimension
        # into the batch dimension, for example, at the first token in the sequence
        # the mask for a three token sequence with thought length two and
        # four lookahead tokens will be
        #   [0., -inf, -inf, 0, 0, 0, 0, -inf, -inf, -inf]
        #   [0., -inf, -inf, 0, 0, 0, 0, 0, -inf, -inf]
        #   [0., -inf, -inf, 0, 0, 0, 0, 0, 0, -inf]
        #   [0., -inf, -inf, 0, 0, 0, 0, 0, 0, 0]
        causal_mask1 = causal_mask1.repeat(b, 1, 1, a, 1)
        causal_mask3 = causal_mask3.repeat(b, 1, l, 1, 1)
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2, causal_mask3], dim=-1)
            .transpose(1, 2)
            .reshape(b * l, 1, a, m + a)
            .contiguous()
        )
        print("causal_mask:", causal_mask.shape, causal_mask[0])

        # these are only the position ids for the tokens in x2
        # they do not include the positions for the tokens represented
        # by the key-value cache
        row = torch.arange(
            self.thought_length + 2,
            self.thought_length + 2 + a,
            dtype=torch.int64,
            device=self.device,
        ).reshape(1, a)
        offset = torch.arange(0, l, dtype=torch.int64, device=self.device).reshape(l, 1)
        position_ids = (
            (row + offset)
            .reshape((1, l, a))
            .tile((b, 1, 1))
            .reshape(b * l, 1, a)
            .contiguous()
        )

        x = x.reshape(b * l, a)
        x = self.tok_emb(x)

        cos, sin = self.rotary_emb(x, seq_len=m + a)
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.attn_norm(x)

            q_position_ids = position_ids.tile(1, self.num_query_heads[i], 1)
            k_position_ids = position_ids.tile(1, self.num_kv_heads[i], 1)
            total_heads = 2 * self.num_kv_heads[i] + self.num_query_heads[i]
            qkv = (
                layer.attn.qkv_proj(x)
                .reshape(b * l, a, total_heads, self.head_dim)
                .permute([0, 2, 1, 3])
            )
            q, k, v = qkv.split(
                [self.num_query_heads[i], self.num_kv_heads[i], self.num_kv_heads[i]],
                dim=1,
            )

            q = layer.attn.q_norm(q)
            k = layer.attn.k_norm(k)

            q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
            k = self.apply_rotary_pos_emb(k, cos, sin, k_position_ids)
            k = torch.concatenate([key_value_cache[i][0], k], dim=-2)
            v = torch.concatenate([key_value_cache[i][1], v], dim=-2)

            k = k.repeat_interleave(self.num_gqa_groups, dim=1)
            v = v.repeat_interleave(self.num_gqa_groups, dim=1)

            layer_causal_mask = causal_mask.repeat(
                1, self.num_kv_heads[i], 1, 1
            ).repeat_interleave(self.num_gqa_groups, dim=1)
            if i == 0:
                print("layer_causal_mask:", layer_causal_mask.shape)
                print("q:", q.shape)
                print("k:", k.shape)
            attn = torch.nn.functional.softmax(
                (torch.matmul(q, k.transpose(-2, -1)) + layer_causal_mask)
                / math.sqrt(q.size(-1)),
                dim=-1,
            )

            attn_out = torch.matmul(attn, v)
            attn_out = layer.attn.out_proj(
                attn_out.reshape(b * l, self.num_query_heads[i], a, self.head_dim)
                .permute([0, 2, 1, 3])
                .reshape(b * l, a, self.num_query_heads[i] * self.head_dim)
            )
            x = residual + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

        # (B * L, A, E)
        h = self.ln(x)

        h = h.reshape(b, l, a, self.embed_dim)

        return h

    def generate_next_thought_token(
        self,
        x: torch.Tensor,
        t: int,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        b, l, d = x.shape

        if key_value_cache is None:
            key_value_cache = []

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

        causal_mask1 = causal_mask1.repeat(b, 1, 1, causal_mask2.size(3), 1)
        causal_mask2 = causal_mask2.repeat(b, 1, causal_mask1.size(2), 1, 1)
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .swapaxes(1, 2)
            .reshape(b * l, 1, d, t + l)
            .contiguous()
        )

        row = torch.arange(0, d, dtype=torch.int64, device=self.device).reshape(1, d)
        offset = torch.arange(
            t - d + 1, l + t - d + 1, dtype=torch.int64, device=self.device
        ).reshape(l, 1)
        position_ids = (
            (row + offset)
            .reshape(1, l, d)
            .tile((b, 1, 1))
            .reshape(b * l, 1, d)
            .contiguous()
        )

        x = x.reshape(b * l, d)
        x = self.tok_emb(x)

        cos, sin = self.rotary_emb(x, seq_len=l + t + 1)
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.attn_norm(x)
            q_position_ids = position_ids.tile(1, self.num_query_heads[i], 1)
            k_position_ids = position_ids.tile(1, self.num_kv_heads[i], 1)
            total_heads = 2 * self.num_kv_heads[i] + self.num_query_heads[i]
            qkv = (
                layer.attn.qkv_proj(x)
                .reshape(b * l, d, total_heads, self.head_dim)
                .permute([0, 2, 1, 3])
            )
            q, k, v = qkv.split(
                [self.num_query_heads[i], self.num_kv_heads[i], self.num_kv_heads[i]],
                dim=1,
            )

            q = layer.attn.q_norm(q)
            k = layer.attn.k_norm(k)

            if len(key_value_cache) > i:
                q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
                k = self.apply_rotary_pos_emb(k, cos, sin, k_position_ids)
                k = torch.concatenate([key_value_cache[i][0], k], dim=-2)
                v = torch.concatenate([key_value_cache[i][1], v], dim=-2)

                key_value_cache[i] = (k, v)
            else:
                k1 = k[:, :, :1].repeat(1, 1, l, 1)
                k2 = k[:, :, 1:]
                v1 = (
                    v[:, :, :1]
                    .reshape(b, l, self.num_kv_heads[i], 1, self.head_dim)
                    .swapaxes(1, 3)
                    .repeat(1, l, 1, 1, 1)
                    .reshape(b * l, self.num_kv_heads[i], l, self.head_dim)
                )
                v2 = v[:, :, 1:]

                # apply rotary embedding
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

                key_value_cache.append((k, v))

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
                attn_out.reshape(b * l, self.num_query_heads[i], d, self.head_dim)
                .permute([0, 2, 1, 3])
                .reshape(b * l, d, self.num_query_heads[i] * self.head_dim)
            )
            x = residual + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

        # (B * L, M, E) before reshaping
        h = self.ln(x).reshape(b, l, d, -1)
        logits = self.lm_head(h)

        assert key_value_cache is not None

        return logits, key_value_cache

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
