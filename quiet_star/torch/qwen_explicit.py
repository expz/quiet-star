import math
import warnings

import torch
import torch.nn
import torch.utils.data
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from quiet_star.config import Config
from quiet_star.torch.pretrained import ForwardResult, PretrainedThoughtModel
from quiet_star.torch.utils import expand_dims


class QwenExplicitThoughtModel(PretrainedThoughtModel):
    """
    A version of the Qwen model that can predict thoughts at each token.

    This version of the model performs all the computations in an explicit
    and intuitive way that is easier to follow. It can be used to
    understand the version in qwen.py.
    """

    def __init__(self, config: Config):
        pretrained_config = AutoConfig.from_pretrained(config.model.model_name)

        if config.model.train_max_length > pretrained_config.max_position_embeddings:
            warnings.warn(
                f"max_length was set to {config.model.train_max_length} which is "
                f"greater than the context window supported by the Qwen model "
                f"({pretrained_config.max_position_embeddings})"
            )
            config.model.train_max_length = pretrained_config.max_position_embeddings

        super().__init__(config)

        self.save_hyperparameters()  # saves the argument(s) of __init__

        self.eval_max_length = pretrained_config.sliding_window

        self.num_kv_heads = pretrained_config.num_key_value_heads
        self.num_query_heads = pretrained_config.num_attention_heads
        self.num_gqa_groups = self.num_query_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_query_heads

        modules = dict(self.model.named_modules())

        self.layers = torch.nn.ModuleList(modules["model.layers"])

        self.ln = modules["model.norm"]

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
        position_ids = row.reshape(1, l).tile((b, 1)).unsqueeze(1)

        x = self.tok_emb(x)

        cos, sin = self.layers[0].self_attn.rotary_emb(x, seq_len=l)
        for layer in self.layers:
            residual = x
            x = layer.input_layernorm(x)

            q = (
                layer.self_attn.q_proj(x)
                .reshape(b, l, self.num_query_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                layer.self_attn.k_proj(x)
                .reshape(b, l, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                layer.self_attn.v_proj(x)
                .reshape(b, l, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )

            # apply rotary embedding
            q = self.apply_rotary_pos_emb(q, cos, sin, position_ids)
            k = self.apply_rotary_pos_emb(k, cos, sin, position_ids)

            k = (
                k.unsqueeze(2)
                .expand(b, self.num_kv_heads, self.num_gqa_groups, l, self.head_dim)
                .reshape(b, self.num_kv_heads * self.num_gqa_groups, l, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(b, self.num_kv_heads, self.num_gqa_groups, l, self.head_dim)
                .reshape(b, self.num_kv_heads * self.num_gqa_groups, l, self.head_dim)
            )

            a = torch.nn.functional.softmax(
                (torch.matmul(q, k.transpose(2, 3)) + causal_mask1)
                / math.sqrt(self.head_dim),
                dim=-1,
            )

            # attn_out is (B, H, L, E)
            attn_out = torch.matmul(a, v)
            attn_out = layer.self_attn.o_proj(
                attn_out.transpose(1, 2).reshape(b, l, self.embed_dim)
            )
            x = residual + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
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
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).repeat((b, 1, 1, 1))

        position_ids = (
            torch.arange(true_l - l, true_l, dtype=torch.int64, device=self.device)
            .unsqueeze(0)
            .repeat([b, 1])
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
                position_ids=position_ids,
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
            .reshape(b * l, a)
            .contiguous()
        )

        # x together with the key_value_cache makes this model evaluation
        # effectively of a tensor of a batch of B sequences of input IDs of
        # length L with thoughts at each location in the sequence. The thoughts
        # include the token from the sequence, the start thought token,
        # the thought, the end thought token, and the lookahead tokens.

        # For example, if there are two lookahead tokens:

        # token1 <|startofthought|> ... <|endofthought|> token2 token3
        # token2 <|startofthought|> ... <|endofthought|> token3 token4
        # token3 <|startofthought|> ... <|endofthought|> token4 token5
        # ...
        result = self.model(
            x.reshape(b * l, a),
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=key_value_cache,
            output_attentions=False,
            output_hidden_states=True,
        )

        h = result.hidden_states[-1]
        h = h.reshape(b, l, a, self.embed_dim)

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
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate new thought tokens for x at every position in the sequence
        given that there are already t thought tokens.

        This currently requires x to have no padding tokens. To add support for
        padding tokens at a later date, code similar to the following can be
        used:

        ```
        if activation_cache is None:
            # + 1 for dictionary to hold pad_mask
            activation_cache = [{} for _ in range(len(self.layers) + 1)]

        pad_mask = expand_dims(
            torch.full(x.shape, 0, dtype=self._dtype, device=self.device).masked_fill_(
                x != self.pad_token_id, 1.0
            ),
            (1, 4),
        ).tile((1, self.num_heads, 1, 1, 1))
        if activation_cache[-1]:
            pad_mask = torch.concatenate([activation_cache[-1]["pad_mask"], pad_mask], dim=3)
        activation_cache[-1]["pad_mask"] = pad_mask
        pad_mask1 = torch.matmul(pad_mask, pad_mask[:, :, :, :1].transpose(3, 4))
        pad_mask1 = pad_mask1.masked_fill_(
            pad_mask1 == 0.0, float("-inf")
        ).masked_fill_(pad_mask1 == 1.0, 0.0)
        pad_mask2 = torch.matmul(pad_mask, pad_mask[:, :, :, 1:].transpose(3, 4))
        pad_mask2 = pad_mask2.masked_fill_(
            pad_mask2 == 0.0, float("-inf")
        ).masked_fill_(pad_mask2 == 1.0, 0.0)
        ```

        Then add the masks in the lines where q is multiplied by k1 and k2.
        """
        b, l, d = x.shape

        if key_value_cache is None:
            key_value_cache = []

        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = (
            expand_dims(causal_mask1, (0, 2, 3))
            .repeat(b, 1, 1, 1, 1)
            .reshape(b * l, 1, 1, l)
        )
        causal_mask2 = torch.triu(
            torch.full(
                (t + 1, t + 1), float("-inf"), dtype=self._dtype, device=self.device
            ),
            diagonal=1,
        )
        causal_mask2 = expand_dims(causal_mask2[t - d + 1 :, 1:], (0, 1))

        row = torch.arange(0, d, dtype=torch.int64, device=self.device).reshape(1, d)
        offset = torch.arange(
            t - d + 1, l + t - d + 1, dtype=torch.int64, device=self.device
        ).reshape(l, 1)
        position_ids = (
            (row + offset).reshape(1, l, d).tile((b, 1, 1)).reshape(b * l, 1, d)
        )
        q_position_ids = (
            position_ids.tile(1, self.num_query_heads, 1)
            .reshape(b * l, self.num_query_heads, d)
            .contiguous()
        )
        k_position_ids = (
            position_ids.tile(1, self.num_kv_heads, 1)
            .reshape(b * l, self.num_kv_heads, d)
            .contiguous()
        )

        x = x.reshape(b * l, d)
        x = self.tok_emb(x)

        cos, sin = self.layers[0].self_attn.rotary_emb(x, seq_len=l + t + 1)

        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.input_layernorm(x)
            q = (
                layer.self_attn.q_proj(x)
                .reshape(b * l, d, self.num_query_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                layer.self_attn.k_proj(x)
                .reshape(b * l, d, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                layer.self_attn.v_proj(x)
                .reshape(b * l, d, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            if len(key_value_cache) > i:
                q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
                k = self.apply_rotary_pos_emb(k, cos, sin, k_position_ids)
                k = torch.concatenate([key_value_cache[i][0], k], dim=-2)
                v = torch.concatenate([key_value_cache[i][1], v], dim=-2)

                k1 = k[:, :, :l]
                k2 = k[:, :, l:]
                v1 = v[:, :, :l]
                v2 = v[:, :, l:]

                # cache activations for future forward passes
                key_value_cache[i] = (k, v)
            else:
                k1 = k[:, :, :1].repeat(1, 1, l, 1)
                k2 = k[:, :, 1:]
                v1 = (
                    v[:, :, :1]
                    .reshape(b, l, self.num_kv_heads, 1, self.head_dim)
                    .swapaxes(1, 3)
                    .repeat(1, l, 1, 1, 1)
                    .reshape(b * l, self.num_kv_heads, l, self.head_dim)
                )
                v2 = v[:, :, 1:]
                q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
                k1 = self.apply_rotary_pos_emb(k1, cos, sin, k_position_ids[:, :, :1])
                k2 = self.apply_rotary_pos_emb(k2, cos, sin, k_position_ids[:, :, 1:])

                k1 = (
                    k1.reshape(b, l, self.num_kv_heads, l, self.head_dim)
                    .transpose(1, 3)
                    .reshape(b * l, self.num_kv_heads, l, self.head_dim)
                )

                # cache activations for future forward passes
                k = torch.concatenate([k1, k2], dim=-2)
                v = torch.concatenate([v1, v2], dim=-2)
                key_value_cache.append((k, v))

            k1 = k1.repeat_interleave(self.num_gqa_groups, dim=1)
            k2 = k2.repeat_interleave(self.num_gqa_groups, dim=1)
            v1 = v1.repeat_interleave(self.num_gqa_groups, dim=1)
            v2 = v2.repeat_interleave(self.num_gqa_groups, dim=1)

            # For some reason, matmul causes a divergence between the naive
            # logits calculation and this optimized version for bfloat16.
            # This is not fixed by collapsing the batch dims into a single dim.
            a = torch.nn.functional.softmax(
                torch.concatenate(
                    [
                        # attend to tokens in original string
                        # (B * L, H, D, E) @ (B * L, H, E, L) => (B * L, H, D, L)
                        torch.matmul(q, k1.transpose(2, 3)) + causal_mask1,
                        # attend to thought tokens generated so far
                        # (B * L, H, D, E) @ (B * L, H, E, T) => (B * L, H, D, T)
                        torch.matmul(q, k2.transpose(2, 3)) + causal_mask2,
                    ],
                    dim=-1,
                )
                / math.sqrt(self.head_dim),
                dim=-1,
            ).nan_to_num()  # padding mask will cause NaNs where there should be zeros
            a1 = a[:, :, :, :l]
            a2 = a[:, :, :, l:]
            # attn_out is (B, H, L, D, E)
            attn_out = (
                # contributions of tokens in original string
                # (B * L, H, D, L) @ (B * L, H, L, E) => (B * L, H, D, E)
                torch.matmul(a1, v1)
                # contributions of thought tokens generated so far
                # (B * L, H, D, T) @ (B * L, H, T, E) => (B * L, H, D, E)
                + torch.matmul(a2, v2)
            )
            attn_out = layer.self_attn.o_proj(
                attn_out.transpose(1, 2).reshape(b * l, d, self.embed_dim),
            )
            x = residual + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        h = self.ln(x).reshape(b, l, d, self.embed_dim)
        logits = self.lm_head(h)

        assert key_value_cache is not None

        # (B, L, D, vocab_size)
        return logits, key_value_cache

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
