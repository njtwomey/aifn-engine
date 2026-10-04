"""Golden values for aifn-compute/nn/quantise from torch: per-tensor and per-channel quantisation (integer
representations),
min–max observer parameters, and fake quantisation with its straight-through gradient."""

import numpy as np
import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(6, 10)) * np.linspace(0.2, 2, 10)
    xt = torch.tensor(x, dtype=torch.float64)
    # Exact halves, to pin round-half-to-even.
    halves = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=torch.float64)

    affine = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    affine(xt.float())
    a_scale, a_zp = affine.calculate_qparams()
    symmetric = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    symmetric(xt.float())
    s_scale, s_zp = symmetric.calculate_qparams()
    channel = PerChannelMinMaxObserver(ch_axis=1, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    channel(xt.float())
    c_scale, c_zp = channel.calculate_qparams()

    scale, zp = 0.1, 7
    q = torch.quantize_per_tensor(xt.float(), scale, zp, torch.quint8).int_repr()
    qh = torch.quantize_per_tensor(halves.float(), 1.0, 10, torch.quint8).int_repr()
    qc = torch.quantize_per_channel(xt.float(), c_scale.to(torch.float64), c_zp, 1, torch.qint8).int_repr()

    xg = xt.clone().requires_grad_(True)
    fq = torch.fake_quantize_per_tensor_affine(xg, 0.05, 3, 0, 15)
    fq.backward(torch.ones_like(fq))
    assert xg.grad is not None
    return {
        "x": x,
        "observer": {
            "affine": {"scale": float(a_scale), "zeroPoint": int(a_zp)},
            "symmetric": {"scale": float(s_scale), "zeroPoint": int(s_zp)},
            "channel": {"scale": c_scale.numpy(), "zeroPoint": c_zp.numpy()},
        },
        "quantise": {"scale": scale, "zeroPoint": zp, "q": q.numpy(), "halves": halves.numpy(), "qHalves": qh.numpy()},
        "quantisePerChannel": {"q": qc.numpy()},
        "fakeQuantise": {
            "scale": 0.05,
            "zeroPoint": 3,
            "qmin": 0,
            "qmax": 15,
            "y": fq.detach().numpy(),
            "grad": xg.grad.numpy(),
        },
    }
