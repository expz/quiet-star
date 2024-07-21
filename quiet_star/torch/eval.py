import copy
import glob
import json
import os
from typing import Any, Optional, Type

import lm_eval.api.instance
import lm_eval.api.model
import lm_eval.models.utils
import lm_eval.tasks
import lm_eval.utils
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoTokenizer

from quiet_star.config import (  # Used by load_from_checkpoint()
    Config,
    ModelConfig,
    QwenDefaultConfig,
    QwenDefaultModelConfig,
)
from quiet_star.torch.openelm import OpenELMThoughtModel
from quiet_star.torch.pretrained import PretrainedThoughtModel
from quiet_star.torch.qwen import QwenThoughtModel


class LMEvalWrapper(lm_eval.api.model.LM):
    def __init__(
        self,
        model: PretrainedThoughtModel,
        batch_size: int = 1,
        add_bos_token: bool = False,
        prefix_token_id: int | None = None,
        max_length: int | None = None,
        truncation: bool = False,
        logits_cache: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.tokenizer_name,
            trust_remote_code=True,
        )
        self.batch_size = batch_size
        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self.truncation = truncation
        self.prefix_token_id = prefix_token_id
        self.logits_cache = logits_cache

    @property
    def tokenizer_name(self) -> str:
        return self.model.tokenizer_name

    @property
    def eos_token(self) -> str:
        return self.model.tokenizer.eos_token

    @property
    def max_gen_tokens(self) -> int:
        return 512

    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        return self.model.eval_max_length

    @property
    def rank(self) -> int:
        # for simplicity we do not currently support multiple GPUs
        return 0

    @property
    def world_size(self) -> int:
        # for simplicity we do not currently support multiple GPUs
        return 1

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _model_generate(
        self, context: torch.Tensor, max_length: int, **generation_kwargs: Any
    ) -> torch.Tensor:
        # Use HuggingFace's logic to make sure we interpret edge cases the same way
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        if generation_kwargs["temperature"] == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs["temperature"] == 0.0:
            generation_kwargs.pop("temperature")

        return self.model.generate(
            context, max_new_tokens=max_length, **generation_kwargs
        )

    def tok_encode(
        self, string: str, left_truncate_len: int | None = None
    ) -> list[int]:
        """
        Code adapted from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L738
        """
        # by default for CausalLM - false or self.add_bos_token is set
        special_tokens_kwargs = {"add_special_tokens": self.add_bos_token}

        encoding = self.model.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.model.tokenizer.padding_side
        self.model.tokenizer.padding_side = padding_side

        add_special_tokens = {"add_special_tokens": self.add_bos_token}

        encoding = self.model.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.model.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def _model_call(
        self,
        inps: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            return self.model(inps)

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int, inplen: int
    ) -> torch.Tensor:
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(
            req: tuple[tuple[str, str], list[int], list[int]]
        ) -> tuple[int, tuple[int, ...]]:
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(
            req: tuple[tuple[str, str], list[int], list[int]]
        ) -> list[int]:
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = lm_eval.models.utils.Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts" if self.logits_cache else None,
            group_fn=_lookup_one_token_cont,
        )

        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm.tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.model.device,
                )
                inplen: int
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)  # type: ignore[call-overload]
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs: dict[str, Any] = {}
            batched_inps = lm_eval.models.utils.pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.model.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.model.tokenizer.decode(tokens, skip_special_tokens=True)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(
        self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        new_reqs = []
        context_enc: list[int]
        continuation_enc: list[int]
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                assert self.prefix_token_id is not None
                context_enc = [self.prefix_token_id]
                continuation_enc = self.tok_encode(continuation)
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(
        self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False
    ) -> list[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    lm_eval.utils.make_disjoint_window,
                    lm_eval.utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            ll_tokens = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True,
            )

            # discard is_greedy
            string_nll = [x[0] for x in ll_tokens]

            loglikelihoods.append(sum(string_nll))

        return loglikelihoods

    def generate_until(
        self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """
        Code adapted from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L1164
        """
        res = []

        def _collate(req: tuple[str, dict]) -> tuple[int, str]:
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm.tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = lm_eval.models.utils.Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts: list[str]
            all_gen_kwargs: list[dict[str, Any]]
            contexts, all_gen_kwargs = zip(*chunk)  # type: ignore[assignment]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            if not until:
                until = [self.eos_token]
            else:
                until.append(self.eos_token)
            if "max_gen_toks" in kwargs.keys():
                max_gen_tokens = kwargs.pop("max_gen_toks")
            else:
                max_gen_tokens = self.max_gen_tokens

            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_tokens

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.model.device)
            attn_masks = attn_masks.to(self.model.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_tokens

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks
                cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res


def get_checkpoint_path(
    version: int, epoch: int | None = None, step: int | None = None
) -> str:
    if step is not None:
        assert (
            epoch is not None
        ), "if you specify a step, you also need to specify an epoch"
        return f"lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt"

    if epoch is not None:
        path = f"lightning_logs/version_{version}/checkpoints/epoch={epoch}-*"
    else:
        path = f"lightning_logs/version_{version}/checkpoints/epoch=*"

    fnames = glob.glob(path)
    if len(fnames) > 1:
        print(
            f"INFO: There were multiple checkpoints for epoch {epoch}. Choosing the most recent one."
        )
    fnames.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return fnames[0]


def load_status_quo_model(
    cls: Type[PretrainedThoughtModel], model_name: str, tokenizer_name: str
) -> PretrainedThoughtModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        model=ModelConfig(
            device=device,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
        ),
    )
    return cls(config).to(config.model.device)


def eval_pretrained(
    cls: Type[PretrainedThoughtModel],
    version: int,
    epoch: int | None = None,
    step: int | None = None,
    limit: int | None = None,
    model_name: str | None = None,
    tokenizer_name: str | None = None,
) -> None:
    use_thoughts = version == -1
    if use_thoughts:
        print(f"loading untrained model")
        assert model_name is not None
        assert tokenizer_name is not None
        model = load_status_quo_model(cls, model_name, tokenizer_name)
    else:
        checkpoint = get_checkpoint_path(version, epoch, step)
        print(f"loading checkpoint from {checkpoint}")
        model = cls.load_from_checkpoint(checkpoint)

    model.eval()

    model = model.to(model.device)

    eval_model = LMEvalWrapper(model, batch_size=1, add_bos_token=True)

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=["gsm8k"],
        num_fewshot=0,
        task_manager=task_manager,
        device="cuda:0",
        limit=limit,
        log_samples=True,
        apply_chat_template=True,
        system_instruction="You are a helpful and confident assistant. Think step-by-step.\n",
        gen_kwargs=f"use_thoughts={use_thoughts}",
        # gen_kwargs="do_sample=True,temperature=0.7",
    )

    print(json.dumps(results, indent=4, sort_keys=True))


def eval_openelm(
    version: int,
    epoch: int | None = None,
    step: int | None = None,
    limit: int | None = None,
    model_name: str | None = None,
    tokenizer_name: str | None = None,
) -> None:
    return eval_pretrained(
        OpenELMThoughtModel, version, epoch, step, limit, model_name, tokenizer_name
    )


def eval_qwen(
    version: int,
    epoch: int | None = None,
    step: int | None = None,
    limit: int | None = None,
    model_name: str | None = None,
    tokenizer_name: str | None = None,
) -> None:
    return eval_pretrained(
        QwenThoughtModel, version, epoch, step, limit, model_name, tokenizer_name
    )
