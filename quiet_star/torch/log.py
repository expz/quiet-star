import torch.utils.tensorboard

import wandb
from quiet_star.config import Config
from quiet_star.constants import MetricLoggerBackend


class MetricLogger:
    def __init__(self, backend: MetricLoggerBackend, config: Config):
        self.backend = backend
        self.step = -1
        self.prev_step = -2
        self.log_interval = config.log_interval
        self.reset_log_interval()
        if backend == MetricLoggerBackend.WANDB:
            wandb.init(  # type: ignore[attr-defined]
                project="quiet-star",
                config=config.as_dict(),
            )
        elif backend == MetricLoggerBackend.TENSORBOARD:
            self.writer = torch.utils.tensorboard.SummaryWriter()
        elif backend == MetricLoggerBackend.NONE:
            pass
        else:
            raise ValueError(f"Unsupported metric logger backend: {backend}")

    def reset_log_interval(self) -> None:
        self.interval_metrics: list[dict[str, float]] = [
            {} for _ in range(self.log_interval)
        ]

    @staticmethod
    def mean(metrics_list: list[dict[str, float]]) -> dict[str, float]:
        mean_metrics: dict[str, float] = {}
        len_metrics: dict[str, float] = {}
        for metrics in metrics_list:
            for metric, value in metrics.items():
                if metric in mean_metrics:
                    mean_metrics[metric] += value
                    len_metrics[metric] += 1
                else:
                    mean_metrics[metric] = value
                    len_metrics[metric] = 1
        for metric, total_value in mean_metrics.items():
            mean_metrics[metric] = total_value / len_metrics[metric]
        return mean_metrics

    def update_interval_metrics(self, i: int, metrics: dict[str, float]) -> None:
        self.interval_metrics[i].update(**metrics)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if step is None:
            self.prev_step = self.step
            self.step += 1
            step = self.step
        else:
            self.prev_step = self.step
            self.step = step
        intervals_stepped = (self.step + 1) // self.log_interval - (
            self.prev_step + 1
        ) // self.log_interval
        if intervals_stepped > 0:
            skipped_last_interval_step = ((self.step + 1) % self.log_interval != 0) or (
                intervals_stepped > 1
            )
            if not skipped_last_interval_step:
                self.update_interval_metrics(step % self.log_interval, metrics)

            mean_metrics = self.mean(self.interval_metrics)
            if self.backend == MetricLoggerBackend.WANDB:
                self._wandb_log(mean_metrics, (step + 1) // self.log_interval)
            elif self.backend == MetricLoggerBackend.TENSORBOARD:
                self._tensorboard_log(mean_metrics, (step + 1) // self.log_interval)

            self.reset_log_interval()
            if skipped_last_interval_step:
                self.update_interval_metrics(step % self.log_interval, metrics)
        else:
            self.update_interval_metrics(step % self.log_interval, metrics)

    def _wandb_log(self, metrics: dict[str, float], step: int | None = None) -> None:
        wandb.log(metrics, step=step)  # type: ignore[attr-defined]

    def _tensorboard_log(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        if step is None:
            step = self.step
            self.step += 1
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
