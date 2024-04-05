import math
import torch
import torch.nn
from torch.nn import functional as F
import lightning

from transformers import AutoTokenizer

import lightning.pytorch
import torch.utils.data

from quiet_star.config import Config, ModelConfig
from quiet_star.attention_triton import TritonCausalSelfAttention
from quiet_star.attention_torch import TorchCausalSelfAttention


class GELU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(input)


class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(
            config.embed_dim, device=config.device, dtype=config.dtype
        )
        self.ln2 = torch.nn.LayerNorm(
            config.embed_dim, device=config.device, dtype=config.dtype
        )
        if config.attn_type == "torch":
            self.self_attn = TorchCausalSelfAttention(
                config.embed_dim,
                config.num_heads,
                config.max_length,
                config.dropout_attn,
                device=config.device,
                dtype=config.dtype,
            )
        elif config.attn_type == "triton":
            self.self_attn = TritonCausalSelfAttention(
                config.embed_dim,
                config.num_heads,
                config.max_length,
                config.dropout_attn,
                device=config.device,
                dtype=config.dtype,
            )
        else:
            raise ValueError("Unrecognized attention type:", config.attn_type)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                config.embed_dim,
                4 * config.embed_dim,
                device=config.device,
                dtype=config.dtype,
            ),
            GELU(),
            torch.nn.Linear(
                4 * config.embed_dim,
                config.embed_dim,
                device=config.device,
                dtype=config.dtype,
            ),
            torch.nn.Dropout(config.dropout_attn),
        )

    def forward(
        self, x: torch.Tensor, padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Do layer normalization before attention/MLP according to
        https://arxiv.org/pdf/2002.04745.pdf
        """
        x = self.ln1(x)
        # padding is for the key
        x = x + self.self_attn(x, padding)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(lightning.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        model_config = config.model

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        vocab_size = len(self.tokenizer)

        self.tok_emb = torch.nn.Embedding(
            vocab_size,
            model_config.embed_dim,
            device=model_config.device,
            dtype=model_config.dtype,
        )
        self.pos_emb = torch.nn.Parameter(
            torch.randn(
                1,
                model_config.max_length,
                model_config.embed_dim,
                device=model_config.device,
                dtype=model_config.dtype,
            )
        )
        self.drop = torch.nn.Dropout(model_config.dropout_embed)
        self.layers = torch.nn.Sequential(
            *[SelfAttentionBlock(model_config) for _ in range(model_config.num_layers)]
        )
        self.ln = torch.nn.LayerNorm(
            model_config.embed_dim, device=model_config.device, dtype=model_config.dtype
        )
        self.lm_head = torch.nn.Linear(
            model_config.embed_dim,
            vocab_size,
            bias=False,
            device=model_config.device,
            dtype=model_config.dtype,
        )
        # input embedding and logit output weights should be tied
        # do not reverse equality or initial loss will increase 10-fold
        self.tok_emb.weight = self.lm_head.weight

        self.learning_rate = config.learning_rate
        self.betas = config.betas
        self.weight_decay = config.weight_decay

        num_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (num_params / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.layers(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        return logits

    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        b, t, _ = logits.size()

        # Reshape to calculate cross entropy; the 2D version runs at 50% of the speed of the below version
        loss = F.cross_entropy(
            logits.reshape(b * t, -1),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        logits = self.forward(x)
        loss = self.calculate_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> float:
        x = batch["input_ids"][:, :-1]
        y = batch["input_ids"][:, 1:]

        logits = self.forward(x)
        loss = self.calculate_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
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
