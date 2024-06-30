import warnings

import torch
import torch.nn
import torch.utils.data
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from quiet_star.config import Config
from quiet_star.torch.pretrained import ForwardResult, PretrainedThoughtModel
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

        self.save_hyperparameters()  # saves the argument(s) of __init__

        modules = dict(self.model.named_modules())

        self.num_kv_heads = pretrained_config.num_key_value_heads
        self.num_query_heads = pretrained_config.num_attention_heads
        self.num_gqa_groups = self.num_query_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_query_heads

        self.layers = torch.nn.ModuleList(modules["model.layers"])

        self.ln = modules["model.norm"]

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
    ) -> torch.Tensor:
        cos = cos[position_ids]
        sin = sin[position_ids]
        x_embed = (x * cos) + (cls.rotate_half(x) * sin)
        return x_embed

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, M)
                A tensor of a batch of B sequences of input IDs of length L
                with thoughts at each location in the sequence. The thoughts
                include the token from the sequence, the start thought token,
                the thought, the end thought token, and the lookahead tokens.

                For example, if B = 1 and there are two lookahead tokens:

                token1 <|startofthought|> ... <|endofthought|> token2 token3
                token2 <|startofthought|> ... <|endofthought|> token3 token4
                token3 <|startofthought|> ... <|endofthought|> token4 token5
                ...

                Sequences with padding are not supported.

        Returns:
            h: (B, L, D, E)
                A tensor of the output embeddings for each lookahead token.
                The embeddings are offset by 1, so the output embedding for
                the input <|endofthought|> token is the embedding of the first
                lookahead token.
        """
        # x is (B, L, M = 1 + T + 2 + D)
        b, l, m = x.shape

        # the batch of sequences without any thoughts
        x_init = x[:, :, :1].transpose(1, 2).reshape(b, l)

        # a vanilla causal mask preventing attention to previous tokens in each sequence
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

        # we evaluate the model on the sequences without thoughts
        # to get the key value caches
        result = self.model(
            x_init,
            attention_mask=init_causal_mask,
            position_ids=init_position_ids,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )

        # expand the key value caches to have one copy per position in each sequence
        past_key_values = tuple(
            (key.tile(l, 1, 1, 1), value.tile(l, 1, 1, 1))
            for key, value in result.past_key_values
        )

        # this part of the causal mask will prevent thought tokens from
        # attending to later tokens in the original (thought-free) sequence
        # this part of the mask actually governs attention to the tokens
        # represented by the key-value cache
        causal_mask1 = torch.triu(
            torch.full((l, l), float("-inf"), dtype=self._dtype, device=self.device),
            diagonal=1,
        )
        causal_mask1 = expand_dims(causal_mask1, (0, 1, 3))
        # this part of the causal mask will prevent thought tokens from
        # attending to later tokens in the same thought
        causal_mask2 = torch.triu(
            torch.full(
                (m - 1, m - 1), float("-inf"), dtype=self._dtype, device=self.device
            ),
            diagonal=1,
        )
        causal_mask2 = expand_dims(causal_mask2, (0, 1, 2))

        # combine the two masks and absorb the sequence length dimension
        # into the batch dimension, for example, at the first token in the sequence
        # the mask for a three token sequence will be
        #   [0., -inf, -inf, 0., -inf, -inf, -inf, ...]
        #   [0., -inf, -inf, 0., 0., -inf, -inf, ...]
        #   [0., -inf, -inf, 0., 0., 0., -inf, ...]
        #   ...
        causal_mask1 = causal_mask1.repeat(b, 1, 1, causal_mask2.size(3), 1)
        causal_mask2 = causal_mask2.repeat(b, 1, causal_mask1.size(2), 1, 1)
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .transpose(1, 2)
            .reshape(b * l, 1, causal_mask1.size(3), -1)
            .contiguous()
        )

        # these are only the position ids for the tokens in x2
        # they do not include the positions for the tokens represented
        # by the key-value cache
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

    def hidden_states_v2(self, x: torch.Tensor) -> torch.Tensor:
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
            .reshape(1, 1, l, m)
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
            b, self.num_kv_heads, 1, causal_mask2.size(3), 1
        )
        causal_mask2 = causal_mask2.repeat(
            b, self.num_kv_heads, causal_mask1.size(2), 1, 1
        )
        causal_mask = (
            torch.concatenate([causal_mask1, causal_mask2], dim=-1)
            .swapaxes(1, 2)
            .reshape(b * l, self.num_kv_heads, d, t + l)
            .repeat_interleave(self.num_gqa_groups, dim=1)
            .contiguous()
        )

        rows = torch.arange(0, d, dtype=torch.int64, device=self.device).reshape(1, d)
        row_offsets = torch.arange(
            t - d + 1, l + t - d + 1, dtype=torch.int64, device=self.device
        ).reshape(l, 1)
        position_ids = (
            (rows + row_offsets)
            .reshape((1, l, d))
            .tile((b, 1, 1))
            .reshape(b * l, 1, d)
            .contiguous()
        )
        q_position_ids = position_ids.tile(1, self.num_query_heads, 1).reshape(
            b * l, self.num_query_heads, d
        )
        k_position_ids = position_ids.tile(1, self.num_kv_heads, 1).reshape(
            b * l, self.num_kv_heads, d
        )

        x = x.reshape(b * l, d)
        x = self.tok_emb(x)

        for i, layer in enumerate(self.layers):
            residual = x
            x = layer.input_layernorm(x)
            cos, sin = layer.self_attn.rotary_emb(x, seq_len=l + t + 1)
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
            if activation_cache[i]:
                q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
                k = self.apply_rotary_pos_emb(k, cos, sin, k_position_ids)
                k = torch.concatenate([activation_cache[i]["k"], k], dim=-2)
                v = torch.concatenate([activation_cache[i]["v"], v], dim=-2)
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

                # apply rotary embedding
                q = self.apply_rotary_pos_emb(q, cos, sin, q_position_ids)
                k1 = self.apply_rotary_pos_emb(k1, cos, sin, k_position_ids[:, :, :1])
                k2 = self.apply_rotary_pos_emb(k2, cos, sin, k_position_ids[:, :, 1:])

                k1 = (
                    k1.reshape(b, l, self.num_kv_heads, l, self.head_dim)
                    .transpose(1, 3)
                    .reshape(b * l, self.num_kv_heads, l, self.head_dim)
                )

                k = torch.concatenate([k1, k2], dim=-2)
                v = torch.concatenate([v1, v2], dim=-2)

            activation_cache[i]["k"] = k
            activation_cache[i]["v"] = v

            k = k.repeat_interleave(self.num_gqa_groups, dim=1)
            v = v.repeat_interleave(self.num_gqa_groups, dim=1)

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask
            )

            attn_out = layer.self_attn.o_proj(
                attn_out.reshape(b * l, self.num_query_heads, d, self.head_dim)
                .permute([0, 2, 1, 3])
                .reshape(b * l, d, self.embed_dim)
            )
            x = residual + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        h = self.ln(x).reshape(b, l, d, -1)
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
