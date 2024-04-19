import math

import lightning
import lightning.pytorch
import torch
import torch.nn
import torch.utils.data
import transformers.modeling_utils
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from quiet_star.config import Config
from quiet_star.constants import END_THOUGHT_TOKEN, START_THOUGHT_TOKEN
from quiet_star.torch.utils import assert_shape, expand_dims, torch_dtype


class PretrainedThoughtModel(lightning.LightningModule):
    def __init__(
        self, model: transformers.modeling_utils.PreTrainedModel, config: Config
    ):
        super().__init__()

        modules = dict(model.named_modules())

        model_config = config.model

        pretrained_config = AutoConfig.from_pretrained(model_config.model_name)

        self._dtype = torch_dtype(pretrained_config.torch_dtype)
        self.embed_dim = pretrained_config.hidden_size
        assert (
            pretrained_config.num_key_value_heads
            == pretrained_config.num_attention_heads
        )
        self.num_heads = pretrained_config.num_attention_heads
        self.max_length = min(
            pretrained_config.max_position_embeddings, model_config.max_length
        )
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
        init_embedding_token_ids = self.tokenizer(
            config.embedding_init_token, return_attention_mask=False
        )["input_ids"][0]
        init_token_embedding = (
            model.get_input_embeddings().weight[init_embedding_token_ids].detach()
        )
        init_token_embedding = init_token_embedding.mean(dim=0)

        vocab_size = len(self.tokenizer)

        self.tok_emb = torch.nn.Embedding(
            vocab_size,
            self.embed_dim,
            device=model_config.device,
            dtype=self._dtype,
        )
        self.tok_emb.weight[: pretrained_config.vocab_size, :] = (
            model.get_input_embeddings().weight
        )
        self.tok_emb.weight[self.start_thought_token_id, :] = init_token_embedding
        self.tok_emb.weight[self.end_thought_token_id, :] = init_token_embedding

        trainability_mask = torch.zeros_like(self.tok_emb.weight)
        trainability_mask[-2:] = config.embedding_grad_weight
        self.tok_emb.weight.register_hook(lambda grad: grad * trainability_mask)

        self.layers = list(modules["model.layers"])

        self.ln = modules["model.norm"]
        self.lm_head = torch.nn.Linear(
            model_config.embed_dim,
            vocab_size,
            bias=False,
            device=model_config.device,
            dtype=torch_dtype(model_config.dtype),
        )
        # Tie output to input weights
        self.lm_head.weight = self.tok_emb.weight

        self.mixing_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                2 * self.embed_dim,
                2 * self.embed_dim,
                device=self.device,
                dtype=self._dtype,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                2 * model_config.embed_dim,
                1,
                device=model_config.device,
                dtype=self._dtype,
            ),
            torch.nn.Sigmoid(),
        )

        self.learning_rate = config.learning_rate
        self.betas = config.betas
        self.weight_decay = config.weight_decay

        num_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (num_params / 1e6,))

    def forward(
        self, x: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor:
        x = self.tok_emb(x)
        for layer in self.layers:
            x = layer(x)
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
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
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

        row = torch.arange(0, m, dtype=torch.int64, device=self.device).reshape(1, m)
        offset = torch.arange(0, l, dtype=torch.int64, device=self.device).reshape(l, 1)
        position_ids = (row + offset).reshape(1, l, m)

        x = self.tok_emb(x)

        for layer in self.layers:
            x = layer.input_layernorm(x)

            q = (
                layer.self_attn.q_proj(x)
                .reshape(b, l, m, self.num_heads, -1)
                .permute([0, 3, 1, 2, 4])
            )
            k1 = (
                layer.self_attn.k_proj(x[:, :, 0])
                .reshape(b, l, 1, self.num_heads, -1)
                .permute([0, 3, 2, 1, 4])
            )
            k2 = (
                layer.self_attn.k_proj(x[:, :, 1:])
                .reshape(b, l, m - 1, self.num_heads, -1)
                .permute([0, 3, 1, 2, 4])
            )
            v1 = (
                layer.self_attn.v_proj(x[:, :, 0])
                .reshape(b, l, 1, self.num_heads, -1)
                .permute([0, 3, 2, 1, 4])
            )
            v2 = (
                layer.self_attn.v_proj(x[:, :, 1:])
                .reshape(b, l, m - 1, self.num_heads, -1)
                .permute([0, 3, 1, 2, 4])
            )

            # TODO: Fix me
            cos1, sin1 = layer.rotary_emb(v1, seq_len=x.shape[1])
            cos2, sin2 = layer.rotary_emb(v2, seq_len=x.shape[1])
            q = apply_rotary_pos_emb(q, cos1, sin1, position_ids)
            k1 = apply_rotary_pos_emb(k1, cos1, sin1, position_ids[:, :, 0])
            k2 = apply_rotary_pos_emb(k2, cos2, sin2, position_ids[:, :, 1:])

            a = torch.nn.functional.softmax(
                torch.concatenate(
                    [
                        # attend to tokens in original string
                        # (B, H, L, M, E) @ (B, H, 1, E, L) => (B, H, L, M, L)
                        torch.matmul(q, k1.permute([0, 1, 2, 4, 3])) + causal_mask1,
                        # attend to thought and lookahead tokens
                        # (B, H, L, M, E) @ (B, H, L, E, M - 1) => (B, H, L, M, M - 1)
                        torch.matmul(q, k2.permute([0, 1, 2, 4, 3])) + causal_mask2,
                    ],
                    dim=-1,
                )
                / math.sqrt(self.embed_dim / self.num_heads),
                dim=-1,
            )
            a1 = a[:, :, :, :, :l]
            a2 = a[:, :, :, :, l:]
            # attn_out is (B, H, L, D, E)
            attn_out = (
                # contributions of tokens in original string
                # (B, H, L, M, L) @ (B, H, 1, L, E) => (B, H, L, M, E)
                torch.matmul(a1, v1)
                # contributions of thought and lookahead tokens
                # (B, H, L, M, M - 1) @ (B, H, L, M - 1, E) => (B, H, L, M, E)
                + torch.matmul(a2, v2)
            )
            attn_out = layer.self_attn.o_proj(
                attn_out.permute([0, 2, 3, 1, 4]).reshape(b, l, m, self.embed_dim)
            )
            x = x + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # (B, L, D, E)
        h = self.ln(x)

        # only keep the hidden states which we care about
        h = h[:, :, -(self.lookahead_tokens + 1) : -1]

        # drop the states where we don't have enough lookahead tokens
        return h[:, : l - (self.lookahead_tokens - 1)]

    def sample_next_tokens(
        self, logits: torch.Tensor, temp: float = 1.0
    ) -> torch.Tensor:
        logits = logits / temp
        return torch.distributions.categorical.Categorical(logits=logits).sample()

    @staticmethod
    def shift_and_stack(
        x: torch.Tensor, rows: int, cols: int, col_offset: int = 0
    ) -> torch.Tensor:
        y = torch.stack(
            [x[:, i : i + cols] for i in range(col_offset, rows + col_offset)], dim=1
        )

        # row_vec = torch.arange(0, cols, dtype=torch.int64, device=self.device).reshape(1, -1)
        # offset_vec = torch.arange(col_offset, rows + col_offset, dtype=torch.int64, device=self.device).reshape(-1, 1)
        # indices = row_vec + offset_vec
        # yp = torch.take_along_dim(x, indices=indices, dim=1)
        # assert torch.allclose(y, yp, atol=1e-7).item()

        return y

    def generate_thoughts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = x.shape
        n = self.thought_length + 2 + self.lookahead_tokens

        start_token = torch.full(
            (b, min(l, self.max_length - n), 1),
            self.start_thought_token_id,
            device=x.device,
            dtype=torch.int64,
        )
        end_token = torch.full(
            (b, min(l, self.max_length - n), 1),
            self.end_thought_token_id,
            device=x.device,
            dtype=torch.int64,
        )
        padding = torch.full(
            (b, n), self.tokenizer.pad_token_id, device=x.device, dtype=torch.int64
        )
        lookahead = self.shift_and_stack(
            torch.concatenate([x, padding], dim=1),
            rows=min(l, self.max_length - n),
            cols=self.lookahead_tokens,
            col_offset=1,
        )

        x = torch.unsqueeze(x[:, : self.max_length - n], dim=2)
        x = torch.concatenate([x, start_token], dim=2)
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
                thought_logits = torch.concatenate(
                    [thought_logits, logits[:, :, -1:]], dim=2
                )
            next_tokens = self.sample_next_tokens(logits[:, :, -1])
            next_tokens = torch.unsqueeze(next_tokens, -1)
            x = torch.concatenate([x, next_tokens], dim=-1)

        # (B, L, 1 + T + 2 + A)
        x = torch.concatenate([x, end_token, lookahead], dim=-1)

        # (B, L, T)
        assert thought_logits is not None

        return thought_logits, x

    def generate_next_thought_token(
        self,
        x: torch.Tensor,
        t: int,
        activation_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
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

        rows = torch.arange(0, d, dtype=torch.int64, device=self.device).reshape(1, d)
        row_offsets = torch.arange(
            t - d + 1, l + t - d + 1, dtype=torch.int64, device=self.device
        ).reshape(l, 1)
        position_ids = rows + row_offsets

        x = self.tok_emb(x)

        for i, layer in enumerate(self.layers):
            x = layer.input_layernorm(x)
            if activation_cache[i]:
                q = (
                    layer.self_attn.q_proj(x)
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                )
                k1 = activation_cache[i]["k1"]
                v1 = activation_cache[i]["v1"]
                k2 = (
                    layer.self_attn.k_proj(x)
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                )
                v2 = (
                    layer.self_attn.v_proj(x)
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                )
                k2 = torch.concatenate([activation_cache[i]["k2"], k2], dim=-2)
                v2 = torch.concatenate([activation_cache[i]["v2"], v2], dim=-2)
            else:
                q = (
                    layer.self_attn.q_proj(x)
                    .reshape(b, l, d, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                )
                k1 = (
                    layer.self_attn.k_proj(x[:, :, 0, :])
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 2, 1, 4])
                )
                k2 = (
                    layer.self_attn.k_proj(x[:, :, 1:, :])
                    .reshape(b, l, d - 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                )
                v1 = (
                    layer.self_attn.v_proj(x[:, :, 0, :])
                    .reshape(b, l, 1, self.num_heads, -1)
                    .permute([0, 3, 2, 1, 4])
                )
                v2 = (
                    layer.self_attn.v_proj(x[:, :, 1:, :])
                    .reshape(b, l, d - 1, self.num_heads, -1)
                    .permute([0, 3, 1, 2, 4])
                )
                # cache activations for future forward passes
                activation_cache[i]["k1"] = k1
                activation_cache[i]["v1"] = v1

            # cache activations for future forward passes
            activation_cache[i]["k2"] = k2
            activation_cache[i]["v2"] = v2

            # TODO: Fix me, including interaction with activation_cache
            cos1, sin1 = layer.rotary_emb(v1, seq_len=x.shape[1])
            cos2, sin2 = layer.rotary_emb(v2, seq_len=x.shape[1])
            q = apply_rotary_pos_emb(q, cos1, sin1, position_ids)
            k1 = apply_rotary_pos_emb(k1, cos1, sin1, position_ids[:, :, 0])
            k2 = apply_rotary_pos_emb(k2, cos2, sin2, position_ids[:, :, 1:])

            a = torch.nn.functional.softmax(
                torch.concatenate(
                    [
                        # attend to tokens in original string
                        # (B, H, L, D, E) @ (B, H, 1, E, L) => (B, H, L, D, L)
                        torch.matmul(q, k1.permute([0, 1, 2, 4, 3])) + causal_mask1,
                        # attend to thought tokens generated so far
                        # (B, H, L, D, E) @ (B, H, L, E, T) => (B, H, L, D, T)
                        torch.matmul(q, k2.permute([0, 1, 2, 4, 3])) + causal_mask2,
                    ],
                    dim=-1,
                )
                / math.sqrt(self.embed_dim / self.num_heads),
                dim=-1,
            )
            a1 = a[:, :, :, :, :l]
            a2 = a[:, :, :, :, l:]
            # attn_out is (B, H, L, D, E)
            attn_out = (
                # contributions of tokens in original string
                # (B, H, L, D, L) @ (B, H, 1, L, E) => (B, H, L, D, E)
                torch.matmul(a1, v1)
                # contributions of thought tokens generated so far
                # (B, H, L, D, T) @ (B, H, L, T, E) => (B, H, L, D, E)
                + torch.matmul(a2, v2)
            )
            attn_out = layer.self_attn.o_proj(
                attn_out.permute([0, 2, 3, 1, 4]).reshape(b, l, d, self.embed_dim)
            )
            x = x + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.ln(x)
        logits = self.lm_head(x)

        assert activation_cache is not None

        # (B, L, D, vocab_size)
        return logits, activation_cache

    def mixing_head(self, h: torch.Tensor, h_thought: torch.Tensor) -> torch.Tensor:
        x = torch.concatenate([h, h_thought], dim=-1)
        w = self.mixing_mlp(x)
        return w

    def calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, reduce: bool = True
    ) -> float:
        v = logits.shape[-1]

        # Reshape to calculate cross entropy; the 2D version runs at 50% of the speed of the below version
        loss = F.cross_entropy(
            logits.reshape(-1, v),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction="none",
        ).reshape(*targets.shape)
        if reduce:
            return torch.mean(loss)
        return loss

    def forward_pass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Shortcut variables for asserting tensor shapes
        b = inputs.shape[0]
        n = self.num_thoughts
        t = self.thought_length
        a = self.lookahead_tokens
        l = inputs.shape[1]
        lp = min(l, self.max_length - (t + 2 + a))
        lpp = min(l - (a - 1), self.max_length - (t + 2 + a) - (a - 1))
        e = self.embed_dim
        v = len(self.tokenizer)

        assert_shape(inputs, (b, l))
        assert_shape(targets, (b, l))

        offset_max = min(
            inputs.shape[-1] - (self.lookahead_tokens - 1),
            self.max_length
            - (self.thought_length + 2 + self.lookahead_tokens)
            - (self.lookahead_tokens - 1),
        )

        # Calculate logits without thoughts
        targets = self.shift_and_stack(targets, offset_max, self.lookahead_tokens)
        assert_shape(targets, (b, lpp, a))

        logits, h = self.forward(inputs, return_hidden_state=True)
        assert_shape(logits, (b, l, v))
        assert_shape(h, (b, l, e))
        h = self.shift_and_stack(h, offset_max, self.lookahead_tokens)
        logits = self.shift_and_stack(logits, offset_max, self.lookahead_tokens)
        assert_shape(h, (b, lpp, a, e))
        assert_shape(logits, (b, lpp, a, v))

        # Calculate logits with thoughts
        inputs = inputs.repeat(self.num_thoughts, 1)
        assert_shape(inputs, (b * n, l))

        logits_thought, input_with_thoughts = self.generate_thoughts(inputs)
        input_with_thoughts = input_with_thoughts.detach()
        assert_shape(logits_thought, (b * n, lp, t, v))
        assert_shape(input_with_thoughts, (b * n, lp, 1 + t + 2 + a))
        h_thought = self.hidden_states(input_with_thoughts)
        assert_shape(h_thought, (b * n, lpp, a, e))
        logits_lookahead = self.lm_head(h_thought)
        assert_shape(logits_lookahead, (b * n, lpp, a, v))

        # Calculate mixing weight
        h = h.repeat(self.num_thoughts, 1, 1, 1)
        assert_shape(h, (b * n, lpp, a, e))
        w = self.mixing_head(h, h_thought)
        assert_shape(w, (b * n, lpp, a, 1))

        # Calculate final logits
        # logits: (B * N, L - A, A, V), N = num thoughts, A = lookahead length
        logits = logits.repeat(self.num_thoughts, 1, 1, 1)
        logits_final = w * logits + (1.0 - w) * logits_lookahead
        assert_shape(logits, (b * n, lpp, a, v))
        assert_shape(logits_final, (b * n, lpp, a, v))

        # Calculate negative log likelihood
        # loss: (B, N, L - A), N = num thoughts, A = lookahead length
        targets = targets.repeat(self.num_thoughts, 1, 1)
        loss = self.calculate_loss(logits_final, targets, reduce=False)
        loss = torch.mean(loss, dim=-1).reshape(b, self.num_thoughts, -1)
        assert_shape(loss, (b, n, lpp))

        # Calculate REINFORCE loss
        r = -loss
        r_mean = torch.mean(r, dim=1, keepdims=True)
        reward = torch.nn.functional.relu(r - r_mean).detach()
        assert_shape(reward, (b, n, lpp))

        logits_thought = logits_thought.reshape(
            b, self.num_thoughts, logits_thought.shape[1], logits_thought.shape[2], -1
        )
        input_with_thoughts = input_with_thoughts.reshape(
            b, self.num_thoughts, input_with_thoughts.shape[1], -1
        )
        thought_targets = input_with_thoughts[:, :, :, 2 : 2 + self.thought_length]
        assert_shape(logits_thought, (b, n, lp, t, v))
        assert_shape(input_with_thoughts, (b, n, lp, 1 + t + 2 + a))
        assert_shape(thought_targets, (b, n, lp, t))

        policy_loss = reward * torch.mean(
            self.calculate_loss(
                logits_thought[:, :, : -self.lookahead_tokens + 1],
                thought_targets[:, :, : -self.lookahead_tokens + 1],
                reduce=False,
            ),
            dim=-1,
        )
        assert_shape(policy_loss, (b, n, lpp))

        # Calculate total loss averaging across batch, thought number and token position
        total_loss = torch.mean(loss + policy_loss)

        return total_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        loss = self.forward_pass(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        logits = self.forward(x)
        loss = self.calculate_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

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
