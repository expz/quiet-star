import abc

import datasets
import mlx.core  # type: ignore
import mlx.nn
import mlx.optimizers
import mlx.utils
import tqdm


class MLXModule(abc.ABC):
    def __init__(self, model: mlx.nn.Module):
        self.model = model

    def __call__(self, x: mlx.core.array) -> mlx.core.array:
        return self.model(x)

    def compile(self) -> None:
        mlx.core.eval(self.model)

        optimizer = self.configure_optimizers()

        self.state = [self.model.state, optimizer.state]

        def _step(inputs: mlx.core.array, targets: mlx.core.array) -> mlx.core.array:
            loss_and_grad_fn = mlx.nn.value_and_grad(self.model, self.forward_pass)
            loss, grads = loss_and_grad_fn(self.model, inputs, targets)
            optimizer.update(self.model, grads)
            return loss

        self._training_step = mlx.core.compile(
            _step, inputs=self.state, outputs=self.state
        )

    @classmethod
    def forward_pass(
        cls, model: mlx.nn.Module, inputs: mlx.core.array, targets: mlx.core.array
    ) -> mlx.core.array:
        logits = model(inputs)
        loss = cls.calculate_loss(logits, targets)
        return loss

    @staticmethod
    @abc.abstractmethod
    def calculate_loss(
        logits: mlx.core.array, targets: mlx.core.array
    ) -> mlx.core.array:
        pass

    @abc.abstractmethod
    def training_step(self, batch: dict[str, mlx.core.array], batch_idx: int) -> float:
        pass

    @abc.abstractmethod
    def validation_step(
        self, batch: dict[str, mlx.core.array], batch_idx: int
    ) -> float:
        pass

    @abc.abstractmethod
    def configure_optimizers(self) -> mlx.optimizers.Optimizer:
        pass


class MLXIterator:
    def __init__(self, dataset: datasets.Dataset, batch_size: int):
        self.iter = dataset.iter(batch_size=batch_size)
        full_batches = len(dataset) // batch_size
        partial_batch = 1 if len(dataset) % batch_size else 0
        self.len = full_batches + partial_batch

    def __iter__(self) -> "MLXIterator":
        return self

    def __next__(self) -> mlx.core.array:
        return {"input_ids": mlx.core.array(next(self.iter)["input_ids"])}

    def __len__(self) -> int:
        return self.len


class MLXDataLoader:
    def __init__(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.dataset.shuffle()

    def __iter__(self) -> MLXIterator:
        return MLXIterator(self.dataset, self.batch_size)

    def __len__(self) -> int:
        full_batches = len(self.dataset) // self.batch_size
        partial_batch = 1 if len(self.dataset) % self.batch_size else 0
        return full_batches + partial_batch


class MLXTrainer:
    def __init__(self, max_epochs: int = 1):
        self.max_epochs = max_epochs

    def fit(
        self,
        model: MLXModule,
        train_dataloader: MLXDataLoader,
        test_dataloader: MLXDataLoader,
    ) -> None:
        for epoch in range(1, self.max_epochs + 1):
            train_bar = tqdm.tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                leave=True,
                desc=f"epoch {epoch}/{self.max_epochs}",
            )
            for batch_idx, batch in train_bar:
                train_loss = model.training_step(batch, batch_idx)
                train_bar.set_postfix_str(f"train_loss={train_loss:0.4f}")

            test_bar = tqdm.tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
                leave=True,
                desc=f"epoch {epoch}/{self.max_epochs}",
            )
            for batch_idx, batch in test_bar:
                valid_loss = model.validation_step(batch, batch_idx)
                test_bar.set_postfix_str(f"validation loss = {valid_loss:0.4f}")
