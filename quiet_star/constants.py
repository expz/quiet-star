import enum


class MetricLoggerBackend(enum.Enum):
    NONE = 0
    WANDB = 1
    TENSORBOARD = 2


START_THOUGHT_TOKEN = "<|startofthought|>"
END_THOUGHT_TOKEN = "<|endofthought|>"
