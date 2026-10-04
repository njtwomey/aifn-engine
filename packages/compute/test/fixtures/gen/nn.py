"""Golden forward values and gradients for aifn-compute/nn layers, from torch in float64.

Each case gives inputs, parameters (in aifn's layout: dense weights [in, out], recurrent weights [in, gates * H]),
the output and the gradients of a weighted sum of the output, sum(w * y), with respect to the inputs and parameters.
"""

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def t(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=True)


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().numpy()


def grads(y: torch.Tensor, rng: np.random.Generator, *xs: torch.Tensor) -> tuple[np.ndarray, list[np.ndarray]]:
    w = rng.normal(size=tuple(y.shape))
    out = (torch.tensor(w) * y).sum()
    gs = torch.autograd.grad(out, xs)
    return w, [arr(g) for g in gs]


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    # The recurrent cells take torch's default initialisation, drawn from torch's generator: seed it, so regenerating
    # gives the same weights (and the same goldens).
    torch.manual_seed(3)
    out: dict[str, object] = {}

    # conv2d with stride, padding and dilation.
    for name, stride, padding, dilation in [
        ("conv2d_plain", (1, 1), (0, 0), (1, 1)),
        ("conv2d_strided", (2, 1), (1, 2), (1, 1)),
        ("conv2d_dilated", (1, 2), (2, 1), (2, 1)),
    ]:
        x = rng.normal(size=(2, 3, 7, 6))
        k = rng.normal(size=(4, 3, 3, 2))
        xt, kt = t(x), t(k)
        y = F.conv2d(xt, kt, stride=stride, padding=padding, dilation=dilation)
        w, (gx, gk) = grads(y, rng, xt, kt)
        out[name] = {
            "x": x,
            "k": k,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "y": arr(y),
            "w": w,
            "gx": gx,
            "gk": gk,
        }

    # conv1d.
    x = rng.normal(size=(2, 2, 9))
    k = rng.normal(size=(3, 2, 3))
    xt, kt = t(x), t(k)
    y = F.conv1d(xt, kt, stride=2, padding=1, dilation=2)
    w, (gx, gk) = grads(y, rng, xt, kt)
    out["conv1d"] = {"x": x, "k": k, "y": arr(y), "w": w, "gx": gx, "gk": gk}

    # Pooling.
    x = rng.normal(size=(2, 2, 6, 5))
    xt = t(x)
    y = F.max_pool2d(xt, 2, stride=2, padding=1)
    w, (gx,) = grads(y, rng, xt)
    out["maxpool2d"] = {"x": x, "y": arr(y), "w": w, "gx": gx}
    xt = t(x)
    y = F.avg_pool2d(xt, 3, stride=2, padding=1)
    w, (gx,) = grads(y, rng, xt)
    out["avgpool2d"] = {"x": x, "y": arr(y), "w": w, "gx": gx}

    # Layer norm over the last axis.
    x = rng.normal(size=(3, 4, 5)) * 2 + 1
    g = rng.normal(size=5)
    b = rng.normal(size=5)
    xt, gt, bt = t(x), t(g), t(b)
    y = F.layer_norm(xt, (5,), gt, bt, eps=1e-5)
    w, (gx, gg, gb) = grads(y, rng, xt, gt, bt)
    out["layernorm"] = {"x": x, "gamma": g, "beta": b, "y": arr(y), "w": w, "gx": gx, "ggamma": gg, "gbeta": gb}

    # Batch norm (training statistics) on [N, C, H, W].
    x = rng.normal(size=(4, 3, 2, 2))
    g = rng.normal(size=3)
    b = rng.normal(size=3)
    xt, gt, bt = t(x), t(g), t(b)
    y = F.batch_norm(xt, None, None, gt, bt, training=True, eps=1e-5)
    w, (gx, gg, gb) = grads(y, rng, xt, gt, bt)
    out["batchnorm"] = {"x": x, "gamma": g, "beta": b, "y": arr(y), "w": w, "gx": gx, "ggamma": gg, "gbeta": gb}

    # Batch norm running statistics: torch.nn.BatchNorm2d (momentum 0.1) on three training batches, then evaluation.
    # Its own generator, so that the cases after it keep their draws.
    rb = np.random.default_rng(31)
    bn = torch.nn.BatchNorm2d(3)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor(g))
        bn.bias.copy_(torch.tensor(b))
    batches = [rb.normal(loc=1.5, scale=2.0, size=(4, 3, 2, 2)) for _ in range(3)]
    bn.train()
    train_y = [arr(bn(torch.tensor(xb))) for xb in batches]
    assert bn.running_mean is not None and bn.running_var is not None  # track_running_stats defaults to True
    running = [arr(bn.running_mean).copy(), arr(bn.running_var).copy()]
    bn.eval()
    x = rb.normal(size=(2, 3, 2, 2))
    xt = t(x)
    y = bn(xt)
    w, (gx, gg) = grads(y, rb, xt, bn.weight)
    out["batchnorm_running"] = {
        "gamma": g,
        "beta": b,
        "batches": batches,
        "trainY": train_y,
        "runningMean": running[0],
        "runningVariance": running[1],
        "x": x,
        "y": arr(y),
        "w": w,
        "gx": gx,
        "ggamma": gg,
    }

    # Multi-head attention (self-attention, causal), with torch's in-projection split into q, k, v.
    T, d, h = 5, 8, 2
    x = rng.normal(size=(2, T, d))
    mha = torch.nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        for p in mha.parameters():
            p.copy_(torch.tensor(rng.normal(size=tuple(p.shape)) * 0.4))
    xt = t(x)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    y, attn = mha(xt, xt, xt, attn_mask=mask, need_weights=True, average_attn_weights=False)
    w, (gx,) = grads(y, rng, xt)
    win = arr(mha.in_proj_weight)
    bin_ = arr(mha.in_proj_bias)
    out["mha"] = {
        "x": x,
        "heads": h,
        "wq": win[:d].T,
        "wk": win[d : 2 * d].T,
        "wv": win[2 * d :].T,
        "bq": bin_[:d],
        "bk": bin_[d : 2 * d],
        "bv": bin_[2 * d :],
        "wo": arr(mha.out_proj.weight).T,
        "bo": arr(mha.out_proj.bias),
        "y": arr(y),
        "weights": arr(attn),
        "w": w,
        "gx": gx,
    }

    # Recurrent cells: one step from a non-zero state.
    n_in, H, B = 3, 4, 2
    x = rng.normal(size=(B, n_in))
    h0 = rng.normal(size=(B, H))
    c0 = rng.normal(size=(B, H))

    lstm = torch.nn.LSTMCell(n_in, H)
    xt, ht, ct = t(x), t(h0), t(c0)
    h1, c1 = lstm(xt, (ht, ct))
    w, (gx, gh, gc) = grads(h1 + 0.5 * c1, rng, xt, ht, ct)
    out["lstm"] = {
        "x": x,
        "h": h0,
        "c": c0,
        "inputWeight": arr(lstm.weight_ih).T,
        "hiddenWeight": arr(lstm.weight_hh).T,
        "bias": arr(lstm.bias_ih) + arr(lstm.bias_hh),
        "h1": arr(h1),
        "c1": arr(c1),
        "w": w,
        "gx": gx,
        "gh": gh,
        "gc": gc,
    }

    gru = torch.nn.GRUCell(n_in, H)
    xt, ht = t(x), t(h0)
    h1 = gru(xt, ht)
    w, (gx, gh) = grads(h1, rng, xt, ht)
    out["gru"] = {
        "x": x,
        "h": h0,
        "inputWeight": arr(gru.weight_ih).T,
        "hiddenWeight": arr(gru.weight_hh).T,
        "bias": arr(gru.bias_ih),
        "hiddenBias": arr(gru.bias_hh),
        "h1": arr(h1),
        "w": w,
        "gx": gx,
        "gh": gh,
    }

    rnn = torch.nn.RNNCell(n_in, H)
    xt, ht = t(x), t(h0)
    h1 = rnn(xt, ht)
    w, (gx, gh) = grads(h1, rng, xt, ht)
    out["rnn"] = {
        "x": x,
        "h": h0,
        "inputWeight": arr(rnn.weight_ih).T,
        "hiddenWeight": arr(rnn.weight_hh).T,
        "bias": arr(rnn.bias_ih) + arr(rnn.bias_hh),
        "h1": arr(h1),
        "w": w,
        "gx": gx,
        "gh": gh,
    }

    # Activations.
    x = np.linspace(-4, 4, 17)
    out["activations"] = {
        "x": x,
        "gelu": arr(F.gelu(torch.tensor(x))),
        "geluTanh": arr(F.gelu(torch.tensor(x), approximate="tanh")),
        "silu": arr(F.silu(torch.tensor(x))),
        "elu": arr(F.elu(torch.tensor(x))),
        "leakyRelu": arr(F.leaky_relu(torch.tensor(x), 0.01)),
    }

    # RMS norm over the last axis (Zhang & Sennrich, 2019), torch's F.rms_norm.
    x = rng.normal(size=(3, 6))
    g = rng.normal(size=6)
    xt, gt = t(x), t(g)
    y = F.rms_norm(xt, (6,), gt, eps=1e-6)
    w, (gx, gg) = grads(y, rng, xt, gt)
    out["rmsnorm"] = {"x": x, "gamma": g, "y": arr(y), "w": w, "gx": gx, "ggamma": gg}
    return out
