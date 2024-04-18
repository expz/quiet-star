import torch


def torch_dtype(dtype: str) -> torch.dtype:
    dt = getattr(torch, dtype)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a torch dtype")
    return dt


def assert_shape(t: torch.Tensor, shape: tuple) -> None:
    assert t.shape == shape, f"expected shape: {shape}, actual shape: {t.shape}"


def expand_dims(t: torch.Tensor, dims: tuple) -> torch.Tensor:
    for dim in dims:
        t = torch.unsqueeze(t, dim=dim)
    return t
