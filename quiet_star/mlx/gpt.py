import math

import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
from transformers import AutoTokenizer

from quiet_star.config import GPTConfig, GPTModelConfig
from quiet_star.constants import END_THOUGHT_TOKEN, START_THOUGHT_TOKEN
from quiet_star.mlx.framework import MLXModule
from quiet_star.mlx.utils import assert_shape


class SelfAttentionBlock(mlx.nn.Module):
    def __init__(self, config: GPTModelConfig):
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
    def __init__(self, config: GPTConfig):
        super().__init__()

        model_config = config.model

        self.embed_dim = model_config.embed_dim
        self.num_heads = model_config.num_heads
        self.max_length = model_config.max_length
        self.num_thoughts = config.num_thoughts
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

        self.mixing_mlp = mlx.nn.Sequential(
            mlx.nn.Linear(2 * model_config.embed_dim, 2 * model_config.embed_dim),
            mlx.nn.ReLU(),
            mlx.nn.Linear(2 * model_config.embed_dim, 1),
            mlx.nn.Sigmoid(),
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
        return h[:, : l - (self.lookahead_tokens - 1)]

    def sample_next_tokens(
        self, logits: mlx.core.array, temp: float = 1.0
    ) -> mlx.core.array:
        logits = logits / temp
        return mlx.core.random.categorical(logits)

    @staticmethod
    def shift_and_stack(
        x: mlx.core.array, rows: int, cols: int, col_offset: int = 0
    ) -> mlx.core.array:
        y = mlx.core.stack(
            [x[:, i : i + cols] for i in range(col_offset, rows + col_offset)], axis=1
        )

        # row_vec = mlx.core.arange(0, cols, dtype=mlx.core.uint32).reshape(1, -1)
        # offset_vec = mlx.core.arange(col_offset, rows + col_offset, dtype=mlx.core.uint32).reshape(-1, 1)
        # indices = row_vec + offset_vec
        # yp = mlx.core.take(x, indices=indices, axis=1)
        # assert mlx.core.allclose(y, yp, atol=1e-7).item()

        return y

    def generate_thoughts(
        self, x: mlx.core.array
    ) -> tuple[mlx.core.array, mlx.core.array]:
        b, l = x.shape
        n = self.thought_length + 2 + self.lookahead_tokens

        start_token = mlx.core.full(
            (b, min(l, self.max_length - n), 1),
            self.start_thought_token_id,
            dtype=mlx.core.uint32,
        )
        end_token = mlx.core.full(
            (b, min(l, self.max_length - n), 1),
            self.end_thought_token_id,
            dtype=mlx.core.uint32,
        )
        padding = mlx.core.full(
            (b, n), self.tokenizer.pad_token_id, dtype=mlx.core.uint32
        )
        lookahead = self.shift_and_stack(
            mlx.core.concatenate([x, padding], axis=1),
            rows=min(l, self.max_length - n),
            cols=self.lookahead_tokens,
            col_offset=1,
        )

        x = mlx.core.expand_dims(x[:, : self.max_length - n], axis=2)
        x = mlx.core.concatenate([x, start_token], axis=2)
        next_tokens = x

        activation_cache = None
        thought_logits = None
        for t in range(1, self.thought_length + 1):
            logits, activation_cache = self.generate_next_thought_token(
                next_tokens, t, activation_cache
            )
            if t == 1:
                thought_logits = logits[:, :, -1:]
            else:
                thought_logits = mlx.core.concatenate(
                    [thought_logits, logits[:, :, -1:]], axis=2
                )
            next_tokens = self.sample_next_tokens(logits[:, :, -1])
            next_tokens = mlx.core.expand_dims(next_tokens, axis=-1)
            x = mlx.core.concatenate([x, next_tokens], axis=-1)

        # (B, L, 1 + T + 2 + A)
        x = mlx.core.concatenate([x, end_token, lookahead], axis=-1)

        # (B, L, T)
        assert thought_logits is not None

        return thought_logits, x

    def generate_next_thought_token(
        self,
        x: mlx.core.array,
        t: int,
        activation_cache: list[dict[str, mlx.core.array]] | None = None,
    ) -> tuple[mlx.core.array, list[dict[str, mlx.core.array]]]:
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

        assert activation_cache is not None

        # (B, L, D, vocab_size)
        return logits, activation_cache

    def mixing_head(
        self, h: mlx.core.array, h_thought: mlx.core.array
    ) -> mlx.core.array:
        x = mlx.core.concatenate([h, h_thought], axis=-1)
        w = self.mixing_mlp(x)
        return w


class GPTModel(MLXModule):
    def __init__(self, config: GPTConfig):
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
        logits: mlx.core.array, targets: mlx.core.array, reduce: bool = True
    ) -> mlx.core.array:
        v = logits.shape[-1]
        # Reshape to calculate cross entropy; the 2D version runs at 50% of the speed of the below version
        loss = mlx.nn.losses.cross_entropy(
            logits.reshape(-1, v),
            targets.reshape(-1),
            axis=-1,
            reduction="none",
            # ignore_index=self.tokenizer.pad_token_id,
        ).reshape(*targets.shape)
        if reduce:
            return mlx.core.mean(loss)
        return loss

    @classmethod
    def forward_pass(
        cls, model: _GPTModel, inputs: mlx.core.array, targets: mlx.core.array
    ) -> mlx.core.array:
        # Shortcut variables for asserting tensor shapes
        b = inputs.shape[0]
        n = model.num_thoughts
        t = model.thought_length
        a = model.lookahead_tokens
        l = inputs.shape[1]
        lp = min(l, model.max_length - (t + 2 + a))
        lpp = min(l - (a - 1), model.max_length - (t + 2 + a) - (a - 1))
        e = model.embed_dim
        v = len(model.tokenizer)

        assert_shape(inputs, (b, l))
        assert_shape(targets, (b, l))

        offset_max = min(
            inputs.shape[-1] - (model.lookahead_tokens - 1),
            model.max_length
            - (model.thought_length + 2 + model.lookahead_tokens)
            - (model.lookahead_tokens - 1),
        )

        # Calculate logits without thoughts
        targets = model.shift_and_stack(targets, offset_max, model.lookahead_tokens)
        assert_shape(targets, (b, lpp, a))

        logits, h = model(inputs, return_hidden_state=True)
        assert_shape(logits, (b, l, v))
        assert_shape(h, (b, l, e))
        h = model.shift_and_stack(h, offset_max, model.lookahead_tokens)
        logits = model.shift_and_stack(logits, offset_max, model.lookahead_tokens)
        assert_shape(h, (b, lpp, a, e))
        assert_shape(logits, (b, lpp, a, v))

        # Calculate logits with thoughts
        inputs = mlx.core.repeat(inputs, repeats=model.num_thoughts, axis=0)
        assert_shape(inputs, (b * n, l))

        logits_thought, input_with_thoughts = model.generate_thoughts(inputs)
        input_with_thoughts = mlx.core.stop_gradient(input_with_thoughts)
        assert_shape(logits_thought, (b * n, lp, t, v))
        assert_shape(input_with_thoughts, (b * n, lp, 1 + t + 2 + a))
        h_thought = model.hidden_states(input_with_thoughts)
        assert_shape(h_thought, (b * n, lpp, a, e))
        logits_lookahead = model.lm_head(h_thought)
        assert_shape(logits_lookahead, (b * n, lpp, a, v))

        # Calculate mixing weight
        h = mlx.core.repeat(h, repeats=model.num_thoughts, axis=0)
        assert_shape(h, (b * n, lpp, a, e))
        w = model.mixing_head(h, h_thought)
        assert_shape(w, (b * n, lpp, a, 1))

        # Calculate final logits
        # logits: (B * N, L - A, A, V), N = num thoughts, A = lookahead length
        logits = mlx.core.repeat(logits, repeats=model.num_thoughts, axis=0)
        logits_final = w * logits + (1.0 - w) * logits_lookahead
        assert_shape(logits, (b * n, lpp, a, v))
        assert_shape(logits_final, (b * n, lpp, a, v))

        # Calculate negative log likelihood
        # loss: (B, N, L - A), N = num thoughts, A = lookahead length
        targets = mlx.core.repeat(targets, repeats=model.num_thoughts, axis=0)
        loss = cls.calculate_loss(logits_final, targets, reduce=False)
        loss = mlx.core.mean(loss, axis=-1).reshape(b, model.num_thoughts, -1)
        assert_shape(loss, (b, n, lpp))

        # Calculate REINFORCE loss
        r = -loss
        r_mean = r.mean(axis=1, keepdims=True)
        reward = mlx.core.stop_gradient(mlx.nn.relu(r - r_mean))
        assert_shape(reward, (b, n, lpp))

        logits_thought = logits_thought.reshape(
            b, model.num_thoughts, logits_thought.shape[1], logits_thought.shape[2], -1
        )
        input_with_thoughts = input_with_thoughts.reshape(
            b, model.num_thoughts, input_with_thoughts.shape[1], -1
        )
        thought_targets = input_with_thoughts[:, :, :, 2 : 2 + model.thought_length]
        assert_shape(logits_thought, (b, n, lp, t, v))
        assert_shape(input_with_thoughts, (b, n, lp, 1 + t + 2 + a))
        assert_shape(thought_targets, (b, n, lp, t))

        policy_loss = reward * mlx.core.mean(
            cls.calculate_loss(
                logits_thought[:, :, : -model.lookahead_tokens + 1],
                thought_targets[:, :, : -model.lookahead_tokens + 1],
                reduce=False,
            ),
            axis=-1,
        )
        assert_shape(policy_loss, (b, n, lpp))

        # Calculate total loss averaging across batch, thought number and token position
        total_loss = mlx.core.mean(loss + policy_loss)

        return total_loss

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

    def configure_optimizers(self) -> mlx.optimizers.Optimizer:
        return mlx.optimizers.AdamW(
            learning_rate=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
