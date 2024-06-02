import abc

import lightning
import lightning.pytorch
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from quiet_star.config import Config
from quiet_star.constants import END_THOUGHT_TOKEN, START_THOUGHT_TOKEN
from quiet_star.torch.utils import assert_shape, torch_dtype


class PretrainedThoughtModel(lightning.LightningModule, abc.ABC):
    def __init__(self, config: Config):
        super().__init__()

        self.to(config.model.device)

        self._dtype = torch_dtype(config.model.dtype)

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        self.model = model

        self.num_thoughts = config.num_thoughts
        self.thought_length = config.thought_length
        self.lookahead_tokens = config.lookahead_tokens

        self.max_length = config.model.max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name,
            trust_remote_code=True,
        )

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

        self.pad_token_id = self.tokenizer.pad_token_id
        self.start_thought_token_id = self.tokenizer(
            START_THOUGHT_TOKEN, return_attention_mask=False
        )["input_ids"][
            -1
        ]  # -1 because there might be a BOS token
        self.end_thought_token_id = self.tokenizer(
            END_THOUGHT_TOKEN, return_attention_mask=False
        )["input_ids"][-1]
        init_embedding_token_ids = self.tokenizer(
            config.embedding_init_token, return_attention_mask=False
        )["input_ids"][-1]
        init_token_embedding = (
            model.get_input_embeddings().weight[init_embedding_token_ids].detach()
        )
        init_token_embedding = init_token_embedding.mean(dim=0)

        self.tok_emb = self.model.get_input_embeddings()

        e, d = self.tok_emb.weight.shape
        with torch.no_grad():
            if self.start_thought_token_id >= e or self.end_thought_token_id >= e:
                new_embedding_count = (
                    max(self.start_thought_token_id, self.end_thought_token_id) - e + 1
                )
                new_embeddings = torch.zeros(
                    (new_embedding_count, d),
                    dtype=self.tok_emb.weight.dtype,
                    device=self.device,
                )
                self.tok_emb.weight = torch.nn.Parameter(
                    torch.cat([self.tok_emb.weight, new_embeddings])
                )
            self.tok_emb.weight[self.start_thought_token_id, :] = init_token_embedding
            self.tok_emb.weight[self.end_thought_token_id, :] = init_token_embedding

        trainability_mask = torch.zeros_like(self.tok_emb.weight, device=self.device)
        trainability_mask[self.start_thought_token_id] = config.embedding_grad_weight
        trainability_mask[self.end_thought_token_id] = config.embedding_grad_weight
        self.tok_emb.weight.register_hook(lambda grad: grad * trainability_mask)

        # WARNING: The vocab size / size of embedding weights can be larger than
        #          len(tokenizer). The extra weights correspond to dummy tokens.
        self.vocab_size, self.embed_dim = self.tok_emb.weight.shape

        self.lm_head = self.model.lm_head
        if self.lm_head is None:
            self.lm_head = torch.nn.Linear(self.embed_dim, self.vocab_size)
            self.lm_head.weight = self.tok_emb.weight
        with torch.no_grad():
            init_lm_head_weight = self.lm_head.weight[
                init_embedding_token_ids, :
            ].detach()
            init_lm_head_weight = init_lm_head_weight.mean(dim=0)
            self.lm_head.weight[self.start_thought_token_id, :] = init_lm_head_weight
            self.lm_head.weight[self.end_thought_token_id, :] = init_lm_head_weight

        self.mixing_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                2 * self.embed_dim,
                2 * self.embed_dim,
                device=self.device,
                dtype=self._dtype,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                2 * self.embed_dim,
                1,
                device=self.device,
                dtype=self._dtype,
            ),
            torch.nn.Sigmoid(),
        )

        self.learning_rate = config.learning_rate
        self.betas = config.betas
        self.weight_decay = config.weight_decay

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def generate_next_thought_token(
        self,
        x: torch.Tensor,
        t: int,
        activation_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        pass

    def generate_thoughts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = x.shape

        # only generate thoughts for tokens which have enough lookahead tokens
        n = self.thought_length + 2 + self.lookahead_tokens
        lpp = min(self.max_length - n, l - (self.lookahead_tokens - 1))

        start_token = torch.full(
            (b, lpp, 1),
            self.start_thought_token_id,
            device=x.device,
            dtype=torch.int64,
        )
        end_token = torch.full(
            (b, lpp, 1),
            self.end_thought_token_id,
            device=x.device,
            dtype=torch.int64,
        )
        padding = torch.full(
            (b, n), self.pad_token_id, device=x.device, dtype=torch.int64
        )
        lookahead = self.shift_and_stack(
            torch.concatenate([x, padding], dim=1),
            rows=lpp,
            cols=self.lookahead_tokens,
            col_offset=1,
        )

        x = x[:, :lpp]
        x = torch.unsqueeze(x, dim=2)
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

    def mixing_head(self, h: torch.Tensor, h_thought: torch.Tensor) -> torch.Tensor:
        x = torch.concatenate([h, h_thought], dim=-1)
        w = self.mixing_mlp(x)
        return w

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

    def calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, reduce: bool = True
    ) -> float:
        v = logits.shape[-1]

        # Reshape to calculate cross entropy; the 2D version runs at 50% of the speed of the below version
        loss = F.cross_entropy(
            logits.reshape(-1, v),
            targets.reshape(-1),
            ignore_index=self.pad_token_id,
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
        lp = min(l - (a - 1), self.max_length - (t + 2 + a))
        e = self.embed_dim
        v = self.vocab_size

        assert_shape(inputs, (b, l))
        assert_shape(targets, (b, l))

        offset_max = min(
            inputs.shape[-1] - (self.lookahead_tokens - 1),
            self.max_length - (self.thought_length + 2 + self.lookahead_tokens),
        )

        targets = self.shift_and_stack(targets, offset_max, self.lookahead_tokens)
        assert_shape(targets, (b, lp, a))

        # Calculate logits without thoughts
        logits, h = self.forward(inputs, return_hidden_state=True)
        assert_shape(logits, (b, l, v))
        assert_shape(h, (b, l, e))
        h = self.shift_and_stack(h, offset_max, self.lookahead_tokens)
        logits = self.shift_and_stack(logits, offset_max, self.lookahead_tokens)
        assert_shape(h, (b, lp, a, e))
        assert_shape(logits, (b, lp, a, v))

        # Calculate logits with thoughts
        inputs = inputs.repeat(self.num_thoughts, 1)
        assert_shape(inputs, (b * n, l))

        logits_thought, input_with_thoughts = self.generate_thoughts(inputs)
        input_with_thoughts = input_with_thoughts.detach()
        assert_shape(logits_thought, (b * n, lp, t, v))
        assert_shape(input_with_thoughts, (b * n, lp, 1 + t + 2 + a))
        h_thought = self.hidden_states(input_with_thoughts)
        assert_shape(h_thought, (b * n, lp, a, e))
        logits_lookahead = self.lm_head(h_thought)
        assert_shape(logits_lookahead, (b * n, lp, a, v))

        # Calculate mixing weight
        h = h.repeat(self.num_thoughts, 1, 1, 1)
        assert_shape(h, (b * n, lp, a, e))
        w = self.mixing_head(h, h_thought)
        assert_shape(w, (b * n, lp, a, 1))

        del inputs
        del h
        del h_thought

        # Calculate final logits
        # logits: (B * N, L - A, A, V), N = num thoughts, A = lookahead length
        logits = logits.repeat(self.num_thoughts, 1, 1, 1)
        logits_final = w * logits + (1.0 - w) * logits_lookahead
        assert_shape(logits, (b * n, lp, a, v))
        assert_shape(logits_final, (b * n, lp, a, v))

        # Calculate negative log likelihood
        # loss: (B, N, L - A), N = num thoughts, A = lookahead length
        targets = targets.repeat(self.num_thoughts, 1, 1)
        loss = self.calculate_loss(logits_final.float(), targets, reduce=False)
        loss = torch.mean(loss, dim=-1).reshape(b, self.num_thoughts, -1)
        assert_shape(loss, (b, n, lp))

        # Calculate REINFORCE loss
        r = -loss
        r_mean = torch.mean(r, dim=1, keepdims=True)
        reward = torch.nn.functional.relu(r - r_mean).detach()
        assert_shape(reward, (b, n, lp))

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
                logits_thought.float(),
                thought_targets,
                reduce=False,
            ),
            dim=-1,
        )
        assert_shape(policy_loss, (b, n, lp))

        # Calculate total loss averaging across batch, thought number and token position
        total_loss = torch.mean(loss + policy_loss).to(self._dtype)

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
