import mlx.core  # type: ignore


def mlx_dtype(dtype: str) -> mlx.core.Dtype:
    dt = getattr(mlx.core, dtype)
    if not isinstance(dt, mlx.core.Dtype):
        raise ValueError(f"{dtype} is not an mlx dtype")
    return dt


def assert_shape(t: mlx.core.array, shape: tuple) -> None:
    assert t.shape == shape, f"expected shape: {shape}, actual shape: {t.shape}"
