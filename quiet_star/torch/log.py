import enum
from typing import Any

import torch.utils.tensorboard

import wandb
from quiet_star.config import Config
from quiet_star.constants import MetricLoggerBackend


class MetricLogger:
    def __init__(self, backend: MetricLoggerBackend, config: Config | None = None):
        self.backend = backend
        if backend == MetricLoggerBackend.WANDB:
            wandb.init(  # type: ignore[attr-defined]
                project="quiet-star",
                config=config.as_dict() if config is not None else None,
            )
        elif backend == MetricLoggerBackend.TENSORBOARD:
            self.writer = torch.utils.tensorboard.SummaryWriter()
            self.step = 0
        elif backend == MetricLoggerBackend.NONE:
            pass
        else:
            raise ValueError(f"Unsupported metric logger backend: {backend}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.backend == MetricLoggerBackend.WANDB:
            self._wandb_log(metrics, step)
        elif self.backend == MetricLoggerBackend.TENSORBOARD:
            self._tensorboard_log(metrics, step)

    def _wandb_log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        wandb.log(metrics, step=step)  # type: ignore[attr-defined]

    def _tensorboard_log(
        self, metrics: dict[str, Any], step: int | None = None
    ) -> None:
        if step is None:
            step = self.step
            self.step += 1
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
