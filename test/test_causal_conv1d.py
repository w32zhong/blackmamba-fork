import torch
from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref

def test_causal_conv1d(dim, seqlen, width, has_bias, silu_activation, itype, channel_last):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 2 
    # batch = 1
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, activation=activation)
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias_ref, activation=activation)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")

    assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)

test_causal_conv1d(64, 8, 2, False, False, torch.float32, False)
