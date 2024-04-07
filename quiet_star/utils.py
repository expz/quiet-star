import mlx.core


def mlx_dtype(dtype: str) -> mlx.core.Dtype:
    dt = getattr(mlx.core, dtype)
    if not isinstance(dt, mlx.core.Dtype):
        raise ValueError(f"{dtype} is not an mlx dtype")
    return dt
