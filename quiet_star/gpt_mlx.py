import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils

from transformers import AutoTokenizer

from quiet_star.config import Config, ModelConfig
from quiet_star.mlx import MLXModule
from quiet_star.utils import mlx_dtype


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
        self.mask = mlx.core.triu(
            mlx.core.full((config.max_length, config.max_length), float("-inf")),
            k=1,
        )

    def __call__(self, x: mlx.core.array) -> mlx.core.array:
        """
        Do layer normalization before attention/MLP according to
        https://arxiv.org/pdf/2002.04745.pdf
        """
        b, t, e = x.shape

        x = self.ln1(x)
        x = x + self.attn(x, x, x, self.mask[:t, :t])
        x = x + self.mlp(self.ln2(x))
        return x


class _GPTModel(mlx.nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        model_config = config.model

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        vocab_size = len(self.tokenizer)

        self.tok_emb = mlx.nn.Embedding(
            vocab_size,
            model_config.embed_dim,
        )
        self.pos_emb = mlx.core.random.normal(
            [
                1,
                model_config.max_length,
                model_config.embed_dim,
            ],
            dtype=mlx_dtype(model_config.dtype),
        )
        self.drop = mlx.nn.Dropout(model_config.dropout_embed)
        self.layers = mlx.nn.Sequential(
            *[SelfAttentionBlock(model_config) for _ in range(model_config.num_layers)]
        )
        self.ln = mlx.nn.LayerNorm(model_config.embed_dim)
        self.lm_head = mlx.nn.Linear(
            model_config.embed_dim,
            vocab_size,
            bias=False,
        )
        # input embedding and logit output weights should be tied
        # do not reverse equality or initial loss will increase 10-fold
        self.tok_emb.weight = self.lm_head.weight

    def __call__(self, x: mlx.core.array) -> mlx.core.array:
        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.layers(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        return logits


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
