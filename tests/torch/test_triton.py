import torch.nn.functional
import torch.random

from quiet_star.torch.attention_triton import scaled_dot_product_attention

batch_size = 3
seq_length = 5
num_heads = 4
head_dim = 16
dtype = torch.float16

torch.manual_seed(123)


def triton_and_torch_outputs() -> (
    tuple[torch.Tensor, torch.Tensor, torch.nn.Linear, torch.nn.Linear]
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.randn(
        [batch_size, num_heads, seq_length, head_dim],
        dtype=dtype,
        device=device,
    )

    in_proj1 = torch.nn.Linear(
        head_dim, 3 * head_dim, bias=False, device=device, dtype=dtype
    )
    in_proj2 = torch.nn.Linear(
        head_dim, 3 * head_dim, bias=False, device=device, dtype=dtype
    )
    in_proj2.weight = torch.nn.Parameter(in_proj1.weight.clone().detach())

    q1, k1, v1 = torch.split(in_proj1(x), head_dim, dim=-1)
    q2, k2, v2 = torch.split(in_proj2(x), head_dim, dim=-1)

    triton_out = scaled_dot_product_attention(q1, k1, v1)
    torch_out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)

    return triton_out, torch_out, in_proj1, in_proj2


def test_triton_foward() -> None:
    triton_out, torch_out, _, _ = triton_and_torch_outputs()

    assert (
        triton_out.shape == torch_out.shape
    ), f"triton: {triton_out.shape}, torch: {torch_out.shape}"
    assert torch.allclose(
        triton_out, torch_out, atol=1e-3
    ), f"triton: {triton_out.shape}, {triton_out}\ntorch: {torch_out.shape}, {torch_out}"


def test_triton_backward() -> None:
    device = torch.device("cuda")
    y1 = torch.randn(
        [batch_size, num_heads, seq_length, head_dim],
        dtype=dtype,
        device=device,
    )
    y2 = y1.clone().detach()

    triton_out, torch_out, in_proj1, in_proj2 = triton_and_torch_outputs()

    triton_out.backward(y1)
    torch_out.backward(y2)

    triton_grad = in_proj1.weight.grad
    torch_grad = in_proj2.weight.grad

    assert (
        triton_grad.shape == torch_grad.shape
    ), f"triton: {triton_grad.shape}, torch: {torch_grad.shape}"
    assert torch.allclose(
        triton_grad, torch_grad, atol=1e-3
    ), f"triton: {triton_grad.shape}, {triton_grad}\ntorch: {torch_grad.shape}, {torch_grad}"
