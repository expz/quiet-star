import abc
from typing import Callable, NamedTuple

import lightning
import lightning.pytorch
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

import wandb
from quiet_star.config import Config
from quiet_star.constants import END_THOUGHT_TOKEN, START_THOUGHT_TOKEN
from quiet_star.torch.log import MetricLogger
from quiet_star.torch.utils import assert_shape, torch_dtype


class ForwardResult(NamedTuple):
    logits: torch.Tensor
    hidden_state: torch.Tensor | None
    key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None


class PretrainedThoughtModel(lightning.LightningModule, abc.ABC):
    def __init__(self, config: Config):
        super().__init__()

        self.to(config.model.device)

        self._dtype = torch_dtype(config.model.dtype)

        self.metric_logger = MetricLogger(config.logger, config)

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        self.model = model
        self.model_name = config.model.model_name
        self.tokenizer_name = config.model.tokenizer_name

        self.num_thoughts = config.num_thoughts
        self.thought_length = config.thought_length
        self.lookahead_tokens = config.lookahead_tokens
        self.temperature = config.temperature

        self.train_max_length = config.model.train_max_length
        self.eval_max_length = self.train_max_length

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
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.bos_token_id
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            self.tokenizer.pad_token = self.tokenizer.bos_token
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

        self.tok_emb = self.model.get_input_embeddings().to(self.device)

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
            self.lm_head = torch.nn.Linear(self.embed_dim, self.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight
        else:
            e_h, d_h = self.lm_head.weight.shape
            with torch.no_grad():
                # If the predefined lm_head is too small, expand it to include start
                # and end thought tokens
                if (
                    self.start_thought_token_id >= e_h
                    or self.end_thought_token_id >= e_h
                ):
                    new_embedding_count = (
                        max(self.start_thought_token_id, self.end_thought_token_id)
                        - e_h
                        + 1
                    )
                    new_embeddings = torch.zeros(
                        (new_embedding_count, d_h),
                        dtype=self.lm_head.weight.dtype,
                        device=self.device,
                    )
                    self.lm_head.weight = torch.nn.Parameter(
                        torch.cat([self.lm_head.weight, new_embeddings])
                    )

                init_lm_head_weight = self.lm_head.weight[
                    init_embedding_token_ids, :
                ].detach()
                init_lm_head_weight = init_lm_head_weight.mean(dim=0)
                self.lm_head.weight[self.start_thought_token_id, :] = (
                    init_lm_head_weight
                )
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
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_hidden_state: bool = False,
        return_key_value_cache: bool = False,
    ) -> ForwardResult:
        pass

    @abc.abstractmethod
    def generate_next_thought_token(
        self,
        x: torch.Tensor,
        t: int,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        pass

    def generate_thoughts(
        self,
        x: torch.Tensor,
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L)
                A tensor of token ids, where B is the batch size and L is the length of the sequence
        Returns:
            thought_logits: (B, L, T, vocab_size)
            h_lookahead: (B, L, T, embed_dim)
            x: (B, L, 1 + T + 2 + A)
        """
        b, l = x.shape

        # only generate thoughts for tokens which have enough lookahead tokens
        n = self.thought_length + 2 + self.lookahead_tokens
        lpp = min(self.train_max_length - n, l - self.lookahead_tokens)

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

        key_value_cache = None
        thought_logits = None
        # we do one extra iteration to pass the final thought token
        # through the model to update the key_value_cache, but we do not
        # save the output because we force the subsequent token to be an
        # end thought token
        for t in range(1, self.thought_length + 2):
            logits, key_value_cache = self.generate_next_thought_token(
                next_tokens, t, key_value_cache
            )
            if t <= self.thought_length:
                if t == 1:
                    thought_logits = logits[:, :, -1:]
                else:
                    thought_logits = torch.concatenate(
                        [thought_logits, logits[:, :, -1:]], dim=2
                    )
                torch.use_deterministic_algorithms(False)
                next_tokens = self.sample_next_token(
                    logits[:, :, -1, :],
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                ).detach()
                torch.use_deterministic_algorithms(True)
                x = torch.concatenate([x, next_tokens], dim=-1)

        # (B, L, 1 + T + 2 + A)
        x = torch.concatenate([x, end_token, lookahead], dim=-1)

        # (B, L, T)
        assert thought_logits is not None
        assert key_value_cache is not None

        lookahead_input = torch.concatenate([end_token, lookahead[:, :, :-1]], dim=-1)
        h_lookahead = self.lookahead_hidden_states(lookahead_input, key_value_cache)

        return thought_logits, h_lookahead, x

    @abc.abstractmethod
    def lookahead_hidden_states(
        self, x: torch.Tensor, key_value_cache: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        pass

    def mixing_head(self, h: torch.Tensor, h_thought: torch.Tensor) -> torch.Tensor:
        x = torch.concatenate([h, h_thought], dim=-1)
        w = self.mixing_mlp(x)
        return w

    @staticmethod
    def sample_next_token(
        logits: torch.Tensor,
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        suppress: list[int] = [],
    ) -> torch.Tensor:
        """
        Samples a token from the logits output by a language model.

        Args:
            logits: The logits output by the language model.
            top_k: The number of top tokens to consider. If None, all tokens are considered.
            top_p: The probability mass to consider. If None, all tokens are considered.
            temperature: The temperature to apply to the logits.

        Returns:
            The sampled token.
        """

        if suppress:
            for s in suppress:
                logits[..., s] = float("-inf")

        if do_sample:
            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                k_values, k_indices = torch.topk(logits, top_k, dim=-1)
                masked_logits = torch.ones_like(logits) * float("-inf")
                masked_logits = masked_logits.scatter_(-1, k_indices, k_values)

            # Top-p sampling
            if top_p is not None:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, dim=-1, descending=True
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                probs = torch.zeros_like(probs).scatter_(
                    -1, sorted_indices, sorted_probs
                )
                masked_logits = torch.log(probs)

            # Sample from the distribution
            return (
                torch.distributions.categorical.Categorical(logits=logits)
                .sample()
                .unsqueeze(-1)
            )

        return torch.argmax(logits, dim=-1, keepdim=True)

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
    ) -> torch.Tensor:
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
        b, l = inputs.shape
        n = self.num_thoughts
        t = self.thought_length
        a = self.lookahead_tokens
        lp = min(l - a, self.train_max_length - (t + 2 + a))
        e = self.embed_dim
        v = self.vocab_size

        assert_shape(inputs, (b, l))
        assert_shape(targets, (b, l))

        offset_max = min(
            inputs.size(1) - self.lookahead_tokens,
            self.train_max_length - (self.thought_length + 2 + self.lookahead_tokens),
        )

        targets = self.shift_and_stack(targets, offset_max, self.lookahead_tokens)
        assert_shape(targets, (b, lp, a))

        # Calculate logits without thoughts
        logits, h, _ = self.forward(inputs, return_hidden_state=True)  # type: ignore
        assert_shape(logits, (b, l, v))
        assert_shape(h, (b, l, e))
        h = self.shift_and_stack(h, offset_max, self.lookahead_tokens)
        logits = self.shift_and_stack(logits, offset_max, self.lookahead_tokens)
        assert_shape(h, (b, lp, a, e))
        assert_shape(logits, (b, lp, a, v))

        # Calculate logits with thoughts
        inputs = inputs.repeat(self.num_thoughts, 1)
        assert_shape(inputs, (b * n, l))

        logits_thought, h_lookahead, input_with_thoughts = self.generate_thoughts(
            inputs, do_sample=True, temperature=self.temperature
        )
        input_with_thoughts = input_with_thoughts.detach()
        assert_shape(logits_thought, (b * n, lp, t, v))
        assert_shape(input_with_thoughts, (b * n, lp, 1 + t + 2 + a))
        assert_shape(h_lookahead, (b * n, lp, a, e))
        logits_lookahead = self.lm_head(h_lookahead)
        assert_shape(logits_lookahead, (b * n, lp, a, v))

        # Calculate mixing weight
        h = h.repeat(self.num_thoughts, 1, 1, 1)
        assert_shape(h, (b * n, lp, a, e))
        w = self.mixing_head(h, h_lookahead)
        assert_shape(w, (b * n, lp, a, 1))

        del inputs
        del h
        del h_lookahead

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

        self.metric_logger.log(
            {
                "mixing_weight_avg": w.mean().item(),
                "mixing_weight_min": w.min().item(),
                "mixing_weight_max": w.max().item(),
                "loss_avg": loss.mean().item(),
                "loss_min": loss.min().item(),
                "loss_max": loss.max().item(),
                "r_mean_avg": r_mean.mean().item(),
                "r_mean_min": r_mean.min().item(),
                "r_mean_max": r_mean.max().item(),
                "reward_avg": reward.mean().item(),
                "reward_min": reward.min().item(),
                "reward_max": reward.max().item(),
                "policy_loss_avg": policy_loss.mean().item(),
                "policy_loss_min": policy_loss.min().item(),
                "policy_loss_max": policy_loss.max().item(),
                "total_loss": total_loss.item(),
            }
        )

        return total_loss

    def generate_token(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        logits, _, key_value_cache = self.forward(
            x,
            attention_mask=attention_mask,
            key_value_cache=key_value_cache,
            return_key_value_cache=True,
        )
        assert key_value_cache is not None
        return (
            self.sample_next_token(
                logits[:, -1, :],
                do_sample,
                top_k,
                top_p,
                temperature,
                suppress=[self.start_thought_token_id, self.end_thought_token_id],
            ),
            key_value_cache,
        )

    def thoughtful_forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_key_value_cache: bool = False,
        return_hidden_state: bool = False,
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> ForwardResult:
        b, l = x.shape
        if key_value_cache:
            l = key_value_cache[0][0].size(2)

        start_thought_token = torch.full(
            (b, 1),
            self.start_thought_token_id,
            device=x.device,
            dtype=torch.int64,
        )
        end_thought_token = torch.full(
            (b, 1),
            self.end_thought_token_id,
            device=x.device,
            dtype=torch.int64,
        )
        x = torch.cat([x, start_thought_token], dim=1)

        # generate a first thought token and save the key value cache
        logits, _, key_value_cache = self.forward(
            x,
            attention_mask,
            key_value_cache=key_value_cache,
            return_key_value_cache=True,
        )
        next_token = self.sample_next_token(
            logits[:, -1, :],
            do_sample,
            top_k,
            top_p,
            temperature,
            suppress=[self.start_thought_token_id, self.end_thought_token_id],
        )

        # sample self.thought_length - 1 more thought tokens
        for _ in range(self.thought_length - 1):
            logits, _, key_value_cache = self.forward(
                next_token,
                attention_mask,
                key_value_cache=key_value_cache,
                return_key_value_cache=True,
            )
            next_token = self.sample_next_token(
                logits[:, -1, :],
                do_sample,
                top_k,
                top_p,
                temperature,
                suppress=[self.start_thought_token_id, self.end_thought_token_id],
            )

        # add the end thought token and generate the token to keep
        final_tokens = torch.cat([next_token, end_thought_token], dim=1)
        logits, h, key_value_cache = self.forward(
            final_tokens,
            attention_mask,
            key_value_cache=key_value_cache,
            return_key_value_cache=return_key_value_cache,
            return_hidden_state=return_hidden_state,
        )
        if return_key_value_cache:
            assert key_value_cache is not None

            key_value_cache = [
                (
                    torch.cat([kvs[0][:, :, :l], kvs[0][:, :, -1:]], dim=2),
                    torch.cat([kvs[1][:, :, :l], kvs[1][:, :, -1:]], dim=2),
                )
                for kvs in key_value_cache
            ]
        return ForwardResult(logits, h, key_value_cache)

    def generate_thoughtful_token(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate a thought and then a token, returning just the token.

        This method takes an input sequence of shape (batch_size, sequence_length)
        and generates a thought sequence using the model's `forward` method. It
        then samples a token from the distribution of the last token in the thought
        sequence.

        Args:
            x: The input sequence of tokens.
            do_sample: Whether to sample from the distribution or take the argmax.
            top_k: The number of top tokens to consider during sampling.
            top_p: The probability mass to consider during sampling.
            temperature: The temperature to apply to the logits during sampling.

        Returns:
            The generated tokens of shape (batch_size,).
        """
        logits, h, key_value_cache = self.forward(
            x,
            attention_mask,
            key_value_cache,
            return_key_value_cache=True,
            return_hidden_state=True,
        )
        logits_thought, h_thought, _ = self.thoughtful_forward(
            x,
            attention_mask,
            key_value_cache,
            return_key_value_cache=False,
            return_hidden_state=True,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # appease mypy
        assert h is not None
        assert h_thought is not None
        assert key_value_cache is not None

        w = self.mixing_head(h[:, -1, :], h_thought[:, -1, :])
        logits_final = w * logits[:, -1, :] + (1.0 - w) * logits_thought[:, -1, :]

        return (
            self.sample_next_token(
                logits_final,
                do_sample,
                top_k,
                top_p,
                temperature,
                suppress=[self.start_thought_token_id, self.end_thought_token_id],
            ),
            key_value_cache,
        )

    def detect_final_tokens(
        self, x: torch.Tensor, final_tokens_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Detect which sequences in x end in a sequence of tokens given by final_tokens.

        Returns:
            A boolean tensor of shape (batch_size, 1) with True for sequences that end in
            final_tokens and False otherwise.
        """
        batch_size = x.size(0)

        final_token_match = torch.zeros(
            (batch_size,), dtype=torch.bool, device=x.device
        )
        for final_tokens in final_tokens_list:
            assert len(final_tokens.shape) == 1
            final_token_count = final_tokens.size(0)
            batch_ending = final_tokens.unsqueeze(0).repeat(batch_size, 1)
            final_token_match = final_token_match | (
                x[:, -final_token_count:] == batch_ending
            ).all(dim=1)

        return final_token_match

    def _generate(
        self,
        token_generator: Callable,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        stop: list[str] = [],
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temp: float = 1.0,
    ) -> torch.Tensor:
        if x.size(1) == 0:
            return torch.tensor(
                [[self.pad_token_id] * x.size(0)], dtype=x.dtype, device=x.device
            )

        stop_token_ids = [
            self.tokenizer.encode(s, return_tensors="pt", add_special_tokens=False)[
                0
            ].to(x.device)
            for s in stop
        ]

        unfinished_sequences = torch.ones(
            (x.size(0),), device=self.device, dtype=x.dtype
        )

        x_completed = x
        new_token = x
        key_value_cache = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                new_token, key_value_cache = token_generator(
                    new_token,
                    attention_mask,
                    key_value_cache,
                    do_sample,
                    top_k,
                    top_p,
                    temp,
                )
                new_token = new_token * unfinished_sequences + self.pad_token_id * (
                    1 - unfinished_sequences
                )
                x_completed = torch.cat([x_completed, new_token], dim=1)
                unfinished_sequences = unfinished_sequences & ~self.detect_final_tokens(
                    x_completed, stop_token_ids
                )
                if unfinished_sequences.max() == 0:
                    break
        print(self.tokenizer.batch_decode(x_completed))
        return x_completed

    def generate(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        use_thoughts: bool = True,
        stop: list[str] = [],
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate new tokens using the model.

        This method takes an input sequence of shape (batch_size, sequence_length) and generates
        new tokens using the model's `generate_token` or `generate_thoughtful_token` method. It
        continues generating tokens until either `max_new_tokens` tokens have been generated
        or the model outputs the end-of-sequence token.

        Args:
            x: The input sequence of tokens.
            attention_mask: The attention mask for the input sequence.
            max_new_tokens: The maximum number of new tokens to generate.
            use_thoughts: Whether to generate a thought before generating each next token.
            stop: A list of strings to stop generation at.
            do_sample: Whether to sample from the distribution or take the argmax.
            top_k: The number of top tokens to consider during sampling.
            top_p: The probability mass to consider during sampling.
            temp: The temperature to apply to the logits during sampling.

        Returns:
            The generated tokens of shape (batch_size, sequence_length + max_new_tokens).
        """
        if use_thoughts:
            generator_func = self.generate_thoughtful_token
        else:
            generator_func = self.generate_token

        return self._generate(
            generator_func,
            x,
            attention_mask,
            max_new_tokens,
            stop,
            do_sample,
            top_k,
            top_p,
            temperature,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        loss = self.forward_pass(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        with torch.no_grad():
            logits, _, _ = self.forward(x)
            loss = self.calculate_loss(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
