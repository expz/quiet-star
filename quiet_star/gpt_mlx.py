import math

import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils

from transformers import AutoTokenizer

from quiet_star.config import Config, ModelConfig
from quiet_star.mlx import MLXModule
from quiet_star.utils import mlx_dtype


START_THOUGHT_TOKEN = "<|startofthought|>"
END_THOUGHT_TOKEN = "<|endofthought|>"


class SelfAttentionBlock(mlx.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = mlx.nn.LayerNorm(config.embed_dim)
        self.ln2 = mlx.nn.LayerNorm(config.embed_dim)
        self.attn = mlx.nn.MultiHeadAttention(config.embed_dim, config.num_heads)
        self.mlp = mlx.nn.Sequential(
            mlx.nn.Linear(
                config.embed_dim,
                4 * config.embed_dim,
            ),
            mlx.nn.GELU(),
            mlx.nn.Linear(
                4 * config.embed_dim,
                config.embed_dim,
            ),
            mlx.nn.Dropout(config.dropout_attn),
        )
        self.mask = mlx.core.expand_dims(
            mlx.core.triu(
                mlx.core.full((config.max_length, config.max_length), float("-inf")),
                k=1,
            ),
            axis=(0, 1),
        )

    def __call__(self, x: mlx.core.array) -> mlx.core.array:
        """
        Do layer normalization before attention/MLP according to
        https://arxiv.org/pdf/2002.04745.pdf
        """
        t = x.shape[-2]

        x = self.ln1(x)
        x = x + self.attn(x, x, x, self.mask[:, :, :t, :t])
        x = x + self.mlp(self.ln2(x))
        return x


class _GPTModel(mlx.nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        model_config = config.model

        self.embed_dim = model_config.embed_dim
        self.num_heads = model_config.num_heads
        self.max_length = model_config.max_length
        self.thought_length = config.thought_length
        self.lookahead_tokens = config.lookahead_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

        special_tokens_dict = {
            "additional_special_tokens": [
                START_THOUGHT_TOKEN,
                END_THOUGHT_TOKEN,
            ],
        }

        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        assert (
            num_added_tokens == 2
        ), f"attempted to add 2 tokens but {num_added_tokens} were added"

        self.start_thought_token_id = self.tokenizer(
            START_THOUGHT_TOKEN, return_attention_mask=False
        )["input_ids"][0]
        self.end_thought_token_id = self.tokenizer(
            END_THOUGHT_TOKEN, return_attention_mask=False
        )["input_ids"][0]

        vocab_size = len(self.tokenizer)

        self.tok_emb = mlx.nn.Embedding(
            vocab_size,
            model_config.embed_dim,
        )
        self.pos_emb = mlx.nn.Embedding(
            model_config.max_length,
            model_config.embed_dim,
        )
        self.drop = mlx.nn.Dropout(model_config.dropout_embed)
        self.layers = [
            SelfAttentionBlock(model_config) for _ in range(model_config.num_layers)
        ]
        self.ln = mlx.nn.LayerNorm(model_config.embed_dim)
        self.lm_head = mlx.nn.Linear(
            model_config.embed_dim,
            vocab_size,
            bias=False,
        )
        # input embedding and logit output weights should be tied
        # do not reverse equality or initial loss will increase 10-fold
        self.tok_emb.weight = self.lm_head.weight

        self.lookahead_indices = mlx.core.array(
            [
                list(range(i, i + self.lookahead_tokens))
                for i in range(1, self.max_length + 1)
            ]
        )

    def __call__(
        self, x: mlx.core.array, return_hidden_state: bool = False
    ) -> mlx.core.array:
        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb(
            mlx.core.arange(0, token_embeddings.shape[-2], dtype=mlx.core.uint32)
        )

        x = self.drop(token_embeddings + position_embeddings)
        for layer in self.layers:
            x = layer(x)
        h = self.ln(x)
        logits = self.lm_head(h)

        if return_hidden_state:
            return logits, h
        return logits

    def hidden_states(self, x: mlx.core.array) -> mlx.core.array:
        # x is (B, L, M = 1 + T + 2 + D)
        b, l, m = x.shape

        causal_mask1 = mlx.core.triu(
            mlx.core.full((l, l), float("-inf"), dtype=x.dtype),
            k=1,
        )
        causal_mask1 = mlx.core.expand_dims(causal_mask1, (0, 1, 3))
        causal_mask2 = mlx.core.triu(
            mlx.core.full((m, m - 1), float("-inf"), dtype=x.dtype),
            k=0,
        )
        causal_mask2 = mlx.core.expand_dims(causal_mask2, (0, 1, 2))

        token_embeddings = self.tok_emb(x)
        row = mlx.core.arange(0, m, dtype=mlx.core.uint32).reshape(1, m)
        offset = mlx.core.arange(0, l, dtype=mlx.core.uint32).reshape(l, 1)
        pos = (row + offset).reshape(1, l, m)
        position_embeddings = self.pos_emb(pos)

        x = self.drop(token_embeddings + position_embeddings)
        for layer in self.layers:
            x = layer.ln1(x)
            q = (
                layer.leaf_modules()["attn"]["query_proj"](x)
                .reshape(b, l, m, layer.attn.num_heads, -1)
                .transpose([0, 3, 1, 2, 4])
            )
            k1 = (
                layer.leaf_modules()["attn"]["key_proj"](x[:, :, 0])
                .reshape(b, l, 1, layer.attn.num_heads, -1)
                .transpose([0, 3, 2, 1, 4])
            )
            k2 = (
                layer.leaf_modules()["attn"]["key_proj"](x[:, :, 1:])
                .reshape(b, l, m - 1, layer.attn.num_heads, -1)
                .transpose([0, 3, 1, 2, 4])
            )
            v1 = (
                layer.leaf_modules()["attn"]["value_proj"](x[:, :, 0])
                .reshape(b, l, 1, layer.attn.num_heads, -1)
                .transpose([0, 3, 2, 1, 4])
            )
            v2 = (
                layer.leaf_modules()["attn"]["value_proj"](x[:, :, 1:])
                .reshape(b, l, m - 1, layer.attn.num_heads, -1)
                .transpose([0, 3, 1, 2, 4])
            )
            a = mlx.core.softmax(
                mlx.core.concatenate(
                    [
                        # attend to tokens in original string
                        # (B, H, L, M, E) @ (B, H, 1, E, L) => (B, H, L, M, L)
                        mlx.core.matmul(q, k1.transpose([0, 1, 2, 4, 3]))
                        + causal_mask1,
                        # attend to thought and lookahead tokens
                        # (B, H, L, M, E) @ (B, H, L, E, M - 1) => (B, H, L, M, M - 1)
                        mlx.core.matmul(q, k2.transpose([0, 1, 2, 4, 3]))
                        + causal_mask2,
                    ],
                    axis=-1,
                )
                / math.sqrt(self.embed_dim / self.num_heads),
                axis=-1,
            )
            a1 = a[:, :, :, :, :l]
            a2 = a[:, :, :, :, l:]
            # attn_out is (B, H, L, D, E)
            attn_out = (
                # contributions of tokens in original string
                # (B, H, L, M, L) @ (B, H, 1, L, E) => (B, H, L, M, E)
                mlx.core.matmul(a1, v1)
                # contributions of thought and lookahead tokens
                # (B, H, L, M, M - 1) @ (B, H, L, M - 1, E) => (B, H, L, M, E)
                + mlx.core.matmul(a2, v2)
            )
            attn_out = layer.leaf_modules()["attn"]["out_proj"](
                attn_out.transpose([0, 2, 3, 1, 4]).reshape(b, l, m, self.embed_dim)
            )
            x = x + attn_out
            x = x + layer.mlp(layer.ln2(x))
        # (B, L, D, E)
        h = self.ln(x)

        # only keep the hidden states which we care about
        h = h[:, :, -(self.lookahead_tokens + 1) : -1]

        # drop the states where we don't have enough lookahead tokens
        return h[:, :l - (self.lookahead_tokens - 1)]

    def sample_next_tokens(
        self, logits: mlx.core.array, temp: float = 1.0
    ) -> mlx.core.array:
        logits = logits / temp
        return mlx.core.random.categorical(logits)

    def generate_thoughts(self, x: mlx.core.array) -> mlx.core.array:
        b, l = x.shape
        n = self.thought_length + 2 + self.lookahead_tokens

        start_token = mlx.core.full(
            (b, l, 1), self.start_thought_token_id, dtype=mlx.core.uint32
        )
        end_token = mlx.core.full(
            (b, l, 1), self.end_thought_token_id, dtype=mlx.core.uint32
        )
        padding = mlx.core.full(
            (b, n), self.tokenizer.pad_token_id, dtype=mlx.core.uint32
        )
        lookahead = mlx.core.take(
            mlx.core.concatenate([x, padding], axis=1),
            self.lookahead_indices[:l],
            axis=1,
        )
        print(
            "lookahead indices:", self.lookahead_indices.shape, self.lookahead_indices
        )
        print("lookahead:", lookahead.shape, lookahead)

        print("x:", x.shape, x)
        x = mlx.core.expand_dims(x[:, : self.max_length - n], axis=2)
        x = mlx.core.concatenate([x, start_token], axis=2)
        next_tokens = x
        print("x:", x.shape)

        activation_cache = None
        for t in range(1, self.thought_length + 1):
            logits, activation_cache = self.generate_next_thought_token(
                next_tokens, t, activation_cache
            )
            print("logits:", logits.shape)
            next_tokens = self.sample_next_tokens(logits[:, :, -1])
            next_tokens = mlx.core.expand_dims(next_tokens, axis=-1)
            print("x:", x.shape)
            print("next_tokens:", next_tokens.shape)
            x = mlx.core.concatenate([x, next_tokens], axis=-1)

        print("x:", x.shape)
        print("end_token:", end_token.shape)
        print("lookahead:", lookahead.shape)
        x = mlx.core.concatenate([x, end_token, lookahead], axis=-1)

        return x

    def generate_next_thought_token(
        self,
        x: mlx.core.array,
        t: int,
        activation_cache: list[dict[str, mlx.core.array]] | None = None,
    ) -> mlx.core.array:
        b, l, d = x.shape

        if activation_cache is None:
            activation_cache = [{} for _ in range(len(self.layers))]

        causal_mask1 = mlx.core.triu(
            mlx.core.full((l, l), float("-inf"), dtype=x.dtype),
            k=1,
        )
        causal_mask1 = mlx.core.expand_dims(causal_mask1, (0, 1, 3))
        causal_mask2 = mlx.core.triu(
            mlx.core.full((t + 1, t + 1), float("-inf"), dtype=x.dtype),
            k=1,
        )
        causal_mask2 = mlx.core.expand_dims(causal_mask2[t - d + 1 :, 1:], (0, 1, 2))

        rows = mlx.core.arange(0, d, dtype=mlx.core.uint32).reshape(1, d)
        row_offsets = mlx.core.arange(
            t - d + 1, l + t - d + 1, dtype=mlx.core.uint32
        ).reshape(l, 1)
        position_embeddings = self.pos_emb(rows + row_offsets)
        token_embeddings = self.tok_emb(x)

        x = self.drop(token_embeddings + position_embeddings)
        for i, layer in enumerate(self.layers):
            x = layer.ln1(x)
            if activation_cache[i]:
                q = (
                    layer.leaf_modules()["attn"]["query_proj"](x)
                    .reshape(b, l, 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 1, 2, 4])
                )
                k1 = activation_cache[i]["k1"]
                v1 = activation_cache[i]["v1"]
                k2 = (
                    layer.leaf_modules()["attn"]["key_proj"](x)
                    .reshape(b, l, 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 1, 2, 4])
                )
                v2 = (
                    layer.leaf_modules()["attn"]["value_proj"](x)
                    .reshape(b, l, 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 1, 2, 4])
                )
                k2 = mlx.core.concatenate([activation_cache[i]["k2"], k2], axis=-2)
                v2 = mlx.core.concatenate([activation_cache[i]["v2"], v2], axis=-2)
            else:
                q = (
                    layer.leaf_modules()["attn"]["query_proj"](x)
                    .reshape(b, l, d, layer.attn.num_heads, -1)
                    .transpose([0, 3, 1, 2, 4])
                )
                k1 = (
                    layer.leaf_modules()["attn"]["key_proj"](x[:, :, 0, :])
                    .reshape(b, l, 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 2, 1, 4])
                )
                k2 = (
                    layer.leaf_modules()["attn"]["key_proj"](x[:, :, 1:, :])
                    .reshape(b, l, d - 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 1, 2, 4])
                )
                v1 = (
                    layer.leaf_modules()["attn"]["value_proj"](x[:, :, 0, :])
                    .reshape(b, l, 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 2, 1, 4])
                )
                v2 = (
                    layer.leaf_modules()["attn"]["value_proj"](x[:, :, 1:, :])
                    .reshape(b, l, d - 1, layer.attn.num_heads, -1)
                    .transpose([0, 3, 1, 2, 4])
                )
                # cache activations for future forward passes
                activation_cache[i]["k1"] = k1
                activation_cache[i]["v1"] = v1

            # cache activations for future forward passes
            activation_cache[i]["k2"] = k2
            activation_cache[i]["v2"] = v2

            a = mlx.core.softmax(
                mlx.core.concatenate(
                    [
                        # attend to tokens in original string
                        # (B, H, L, D, E) @ (B, H, 1, E, L) => (B, H, L, D, L)
                        mlx.core.matmul(q, k1.transpose([0, 1, 2, 4, 3]))
                        + causal_mask1,
                        # attend to thought tokens generated so far
                        # (B, H, L, D, E) @ (B, H, L, E, T) => (B, H, L, D, T)
                        mlx.core.matmul(q, k2.transpose([0, 1, 2, 4, 3]))
                        + causal_mask2,
                    ],
                    axis=-1,
                )
                / math.sqrt(self.embed_dim / self.num_heads),
                axis=-1,
            )
            a1 = a[:, :, :, :, :l]
            a2 = a[:, :, :, :, l:]
            # attn_out is (B, H, L, D, E)
            attn_out = (
                # contributions of tokens in original string
                # (B, H, L, D, L) @ (B, H, 1, L, E) => (B, H, L, D, E)
                mlx.core.matmul(a1, v1)
                # contributions of thought tokens generated so far
                # (B, H, L, D, T) @ (B, H, L, T, E) => (B, H, L, D, E)
                + mlx.core.matmul(a2, v2)
            )
            attn_out = layer.leaf_modules()["attn"]["out_proj"](
                attn_out.transpose([0, 2, 3, 1, 4]).reshape(b, l, d, self.embed_dim)
            )
            x = x + attn_out
            x = x + layer.mlp(layer.ln2(x))
        x = self.ln(x)
        logits = self.lm_head(x)

        # (B, L, D, vocab_size)
        return logits, activation_cache


class GPTModel(MLXModule):
    def __init__(self, config: Config):
        super().__init__(_GPTModel(config))

        self.learning_rate = config.learning_rate
        self.betas = config.betas
        self.weight_decay = config.weight_decay

        self.tokenizer = self.model.tokenizer

        # Count parameters without double counting tied embeddings
        num_params = sum(
            arr.size
            for name, arr in mlx.utils.tree_flatten(self.model.trainable_parameters())
            if "embedding" not in name
        )
        print("number of parameters: %.2fM" % (num_params / 1e6,))

    @staticmethod
    def calculate_loss(
        logits: mlx.core.array, targets: mlx.core.array
    ) -> mlx.core.array:
        b, t, _ = logits.shape

        # Reshape to calculate cross entropy; the 2D version runs at 50% of the speed of the below version
        loss = mlx.nn.losses.cross_entropy(
            logits.reshape(b * t, -1),
            targets.reshape(-1),
            reduction="mean",
            # ignore_index=self.tokenizer.pad_token_id,
        )

        return loss

    def training_step(self, batch: dict[str, mlx.core.array], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        loss = self._training_step(x, y)
        mlx.core.eval(self.state)
        return loss.tolist()

    def validation_step(
        self, batch: dict[str, mlx.core.array], batch_idx: int
    ) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        logits = self(x)
        loss = self.calculate_loss(logits, y)
        return loss.tolist()

    def configure_optimizers(self):
        return mlx.optimizers.AdamW(
            learning_rate=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
