"""Golden values for aifn-compute/nn/attention, from torch in float64.

References are torch's `scaled_dot_product_attention` (with explicit boolean or additive masks, and `enable_gqa` for
shared key-value heads), `rms_norm`, `layer_norm`, `gelu`, `silu`, and transcriptions of the reference code of each
scheme: the rotate-half RoPE of GPT-NeoX/LLaMA (Hugging Face `apply_rotary_pos_emb`), the complex-number RoPE of
Meta's LLaMA (`apply_rotary_emb`), Hugging Face's linear and YaRN frequency rescalings, BLOOM's ALiBi slopes
(`build_alibi_tensor`), T5's `_relative_position_bucket`, and DeepSeek-V2's latent attention. Parameters are in aifn's
layout: Linear weights [in, out], applied as x W + b. Gradients are of the weighted sum sum(r * y).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def t(a: np.ndarray, grad: bool = False) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().numpy()


def lin(rng: np.random.Generator, i: int, o: int, bias: bool = True) -> dict[str, np.ndarray]:
    p = {"weight": rng.normal(size=(i, o)) / np.sqrt(i)}
    if bias:
        p["bias"] = 0.1 * rng.normal(size=o)
    return p


def apply_lin(h: torch.Tensor, p: dict[str, np.ndarray]) -> torch.Tensor:
    y = h @ t(p["weight"])
    return y + t(p["bias"]) if "bias" in p else y


# ── Positions (transcriptions of the reference implementations) ──────────────────────────────────────────────────────


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_half(x: torch.Tensor, positions: np.ndarray, inv_freq: np.ndarray, mscale: float = 1.0) -> torch.Tensor:
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = t(np.cos(emb) * mscale), t(np.sin(emb) * mscale)
    return x * cos + rotate_half(x) * sin


def rope_interleaved(x: torch.Tensor, positions: np.ndarray, inv_freq: np.ndarray) -> torch.Tensor:
    # Meta's LLaMA: view pairs (2i, 2i + 1) as complex numbers and multiply by e^{i p θ}.
    freqs = torch.polar(torch.ones(len(positions), len(inv_freq)), t(np.outer(positions, inv_freq)))
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(xc * freqs).flatten(-2)


def inv_freq_default(dim: int, base: float = 10000.0) -> np.ndarray:
    return 1.0 / (base ** (np.arange(0, dim, 2) / dim))


def inv_freq_yarn(dim: int, base: float, factor: float, original: int, beta_fast: float = 32, beta_slow: float = 1):
    """Hugging Face `_compute_yarn_parameters` (truncate=True)."""

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    pos_freqs = base ** (np.arange(0, dim, 2) / dim)
    extrapolation_factor = 1 - ramp
    inv = (1 / (factor * pos_freqs)) * (1 - extrapolation_factor) + (1 / pos_freqs) * extrapolation_factor
    return inv, 0.1 * math.log(factor) + 1.0


def alibi_slopes(n: int) -> list[float]:
    """BLOOM's `build_alibi_tensor` slopes."""
    closest = 2 ** math.floor(math.log2(n))
    base = 2 ** (-(2 ** -(math.log2(closest) - 3)))
    slopes = [base ** (i + 1) for i in range(closest)]
    if closest != n:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest) - 3)))
        num_rem = min(closest, n - closest)
        slopes += [extra_base**i for i in range(1, 1 + 2 * num_rem, 2)]
    return slopes


def t5_bucket(relative_position: torch.Tensor, bidirectional: bool, num_buckets: int, max_distance: int):
    """Hugging Face `T5Attention._relative_position_bucket`."""
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.min(large, torch.full_like(large, num_buckets - 1))
    relative_buckets += torch.where(is_small, relative_position, large)
    return relative_buckets


# ── Attention ────────────────────────────────────────────────────────────────────────────────────────────────────────


def sdpa_case(rng, tq: int, tk: int, mask: np.ndarray | None, bias: np.ndarray | None, cap: float | None):
    b, h, d = 2, 3, 4
    q, k, v = rng.normal(size=(b, h, tq, d)), rng.normal(size=(b, h, tk, d)), rng.normal(size=(b, h, tk, d))
    qt, kt, vt = t(q, True), t(k, True), t(v, True)
    scores = qt @ kt.transpose(-1, -2) / math.sqrt(d)
    if bias is not None:
        scores = scores + t(bias)
    if cap is not None:
        scores = cap * torch.tanh(scores / cap)
    if mask is not None:
        scores = scores.masked_fill(torch.tensor(mask) == 0, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    y = w @ vt
    if cap is None and bias is None:
        # Cross-check the hand-written path against torch's kernel.
        ref = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None if mask is None else torch.tensor(mask) == 1)
        assert torch.allclose(ref, y)
    r = rng.normal(size=y.shape)
    gq, gk, gv = torch.autograd.grad((t(r) * y).sum(), (qt, kt, vt))
    out = {"q": q, "k": k, "v": v, "y": arr(y), "weights": arr(w), "r": r, "gq": arr(gq), "gk": arr(gk), "gv": arr(gv)}
    if mask is not None:
        out["mask"] = mask
    if bias is not None:
        out["bias"] = bias
    if cap is not None:
        out["softCap"] = cap
    return out


def position_mask(qp, kp, causal: bool, window: int | None) -> np.ndarray:
    m = np.ones((len(qp), len(kp)))
    for i, p in enumerate(qp):
        for j, q in enumerate(kp):
            if (causal and q > p) or (window is not None and p - q >= window):
                m[i, j] = 0
    return m


def mha(p, x: torch.Tensor, heads: int, kv_heads: int, causal: bool, rope: str | None, qk_norm: bool):
    d_head = p["query"]["weight"].shape[1] // heads
    split = lambda u, n: u.reshape(*u.shape[:-1], n, d_head).transpose(-3, -2)  # noqa: E731
    q = split(apply_lin(x, p["query"]), heads)
    k = split(apply_lin(x, p["key"]), kv_heads)
    v = split(apply_lin(x, p["value"]), kv_heads)
    if qk_norm:
        q = F.rms_norm(q, (d_head,), weight=t(p["queryNorm"]["gamma"]), eps=1e-6)
        k = F.rms_norm(k, (d_head,), weight=t(p["keyNorm"]["gamma"]), eps=1e-6)
    positions = np.arange(x.shape[-2])
    if rope == "half":
        q, k = rope_half(q, positions, inv_freq_default(d_head)), rope_half(k, positions, inv_freq_default(d_head))
    elif rope == "interleaved":
        inv = inv_freq_default(d_head)
        q, k = rope_interleaved(q, positions, inv), rope_interleaved(k, positions, inv)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=kv_heads != heads)
    return apply_lin(o.transpose(-3, -2).reshape(*x.shape[:-1], heads * d_head), p["output"])


def mha_cases(rng):
    out = []
    d, T = 16, 6
    for heads, kv_heads, rope, qk_norm in [(4, 2, "half", True), (4, 1, "interleaved", False), (4, 4, None, True)]:
        dh = d // heads
        p = {
            "query": lin(rng, d, heads * dh),
            "key": lin(rng, d, kv_heads * dh),
            "value": lin(rng, d, kv_heads * dh),
            "output": lin(rng, heads * dh, d),
        }
        if qk_norm:
            p["queryNorm"] = {"gamma": 1 + 0.2 * rng.normal(size=dh)}
            p["keyNorm"] = {"gamma": 1 + 0.2 * rng.normal(size=dh)}
        x = rng.normal(size=(2, T, d))
        xt = t(x, True)
        y = mha(p, xt, heads, kv_heads, True, rope, qk_norm)
        r = rng.normal(size=y.shape)
        (gx,) = torch.autograd.grad((t(r) * y).sum(), (xt,))
        out.append(
            {
                "heads": heads,
                "kvHeads": kv_heads,
                "rope": rope,
                "qkNorm": qk_norm,
                "params": p,
                "x": x,
                "y": arr(y),
                "r": r,
                "gx": arr(gx),
            }
        )
    return out


def mla_case(rng):
    d, heads, dh, dc, dr, dq, T = 16, 2, 4, 6, 4, 8, 5
    p = {
        "queryDown": lin(rng, d, dq, False),
        "query": lin(rng, dq, heads * dh, False),
        "queryRope": lin(rng, dq, heads * dr, False),
        "kvDown": lin(rng, d, dc, False),
        "kvNorm": {"gamma": 1 + 0.2 * rng.normal(size=dc)},
        "keyUp": lin(rng, dc, heads * dh, False),
        "valueUp": lin(rng, dc, heads * dh, False),
        "keyRope": lin(rng, d, dr, False),
        "output": lin(rng, heads * dh, d, False),
    }
    x = rng.normal(size=(T, d))
    xt = t(x, True)
    pos = np.arange(T)
    inv = inv_freq_default(dr)
    split = lambda u, w: u.reshape(T, heads, w).transpose(0, 1)  # noqa: E731
    cq = apply_lin(xt, p["queryDown"])
    q_c = split(apply_lin(cq, p["query"]), dh)
    q_r = rope_half(split(apply_lin(cq, p["queryRope"]), dr), pos, inv)
    c = F.rms_norm(apply_lin(xt, p["kvDown"]), (dc,), weight=t(p["kvNorm"]["gamma"]), eps=1e-6)
    k_c = split(apply_lin(c, p["keyUp"]), dh)
    v = split(apply_lin(c, p["valueUp"]), dh)
    k_r = rope_half(apply_lin(xt, p["keyRope"]), pos, inv).unsqueeze(0).expand(heads, T, dr)
    q = torch.cat([q_c, q_r], -1)
    k = torch.cat([k_c, k_r], -1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = apply_lin(o.transpose(0, 1).reshape(T, heads * dh), p["output"])
    r = rng.normal(size=y.shape)
    (gx,) = torch.autograd.grad((t(r) * y).sum(), (xt,))
    return {"heads": heads, "params": p, "x": x, "y": arr(y), "r": r, "gx": arr(gx)}


# ── Feed-forward and blocks ──────────────────────────────────────────────────────────────────────────────────────────


def ffn(p, h, kind: str):
    up = apply_lin(h, p["up"])
    if kind == "mlp":
        return apply_lin(F.gelu(up), p["down"])
    act = {"swiglu": F.silu, "geglu": F.gelu, "reglu": F.relu}[kind]
    return apply_lin(act(apply_lin(h, p["gate"])) * up, p["down"])


def ffn_cases(rng):
    out = []
    d, hidden = 6, 10
    for kind in ["mlp", "swiglu", "geglu", "reglu"]:
        p = {"up": lin(rng, d, hidden), "down": lin(rng, hidden, d)}
        if kind != "mlp":
            p["gate"] = lin(rng, d, hidden)
        x = rng.normal(size=(3, d))
        xt = t(x, True)
        y = ffn(p, xt, kind)
        r = rng.normal(size=y.shape)
        (gx,) = torch.autograd.grad((t(r) * y).sum(), (xt,))
        out.append({"kind": kind, "params": p, "x": x, "y": arr(y), "r": r, "gx": arr(gx)})
    return out


D, HEADS, HIDDEN, BATCH, T = 8, 2, 16, 2, 5


def block_params(rng, norm: str, ff: str, parallel: bool, kv_heads: int, bias: bool):
    def nrm():
        g = {"gamma": 1 + 0.2 * rng.normal(size=D)}
        if norm == "layer":
            g["beta"] = 0.1 * rng.normal(size=D)
        return g

    dh = D // HEADS
    p = {
        "attentionNorm": nrm(),
        "attention": {
            "query": lin(rng, D, D, bias),
            "key": lin(rng, D, kv_heads * dh, bias),
            "value": lin(rng, D, kv_heads * dh, bias),
            "output": lin(rng, D, D, bias),
        },
        "feedForward": {"up": lin(rng, D, HIDDEN, bias), "down": lin(rng, HIDDEN, D, bias)},
    }
    if ff != "mlp":
        p["feedForward"]["gate"] = lin(rng, D, HIDDEN, bias)
    if not parallel:
        p["feedForwardNorm"] = nrm()
    return p


def block(p, x, placement, norm, causal, ff="mlp", parallel=False, kv_heads=HEADS, rope=None, alibi=False):
    def normalise(np_, h):
        if norm == "rms":
            return F.rms_norm(h, (D,), weight=t(np_["gamma"]), eps=1e-6)
        return F.layer_norm(h, (D,), weight=t(np_["gamma"]), bias=t(np_["beta"]), eps=1e-5)

    def attention(h):
        if alibi:
            a = p["attention"]
            dh = D // HEADS
            split = lambda u: u.reshape(*u.shape[:-1], HEADS, dh).transpose(-3, -2)  # noqa: E731
            q, k, v = (split(apply_lin(h, a[n])) for n in ("query", "key", "value"))
            pos = np.arange(h.shape[-2])
            bias = np.array([[[-m * abs(i - j) for j in pos] for i in pos] for m in alibi_slopes(HEADS)])
            mask = position_mask(pos, pos, causal, None)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=t(np.where(mask == 1, bias, -np.inf)))
            return apply_lin(o.transpose(-3, -2).reshape(*h.shape), a["output"])
        return mha(p["attention"], h, HEADS, kv_heads, causal, rope, False)

    if parallel:
        h = normalise(p["attentionNorm"], x)
        return x + attention(h) + ffn(p["feedForward"], h, ff)
    if placement == "pre":
        h = x + attention(normalise(p["attentionNorm"], x))
        return h + ffn(p["feedForward"], normalise(p["feedForwardNorm"], h), ff)
    h = normalise(p["attentionNorm"], x + attention(x))
    return normalise(p["feedForwardNorm"], h + ffn(p["feedForward"], h, ff))


def block_cases(rng):
    x = rng.normal(size=(BATCH, T, D))
    r = rng.normal(size=(BATCH, T, D))
    out = {"x": x, "r": r, "dModel": D, "heads": HEADS, "hidden": HIDDEN, "blocks": []}
    variants = [
        dict(placement="pre", norm="layer", causal=False),
        dict(placement="post", norm="layer", causal=True),
        dict(placement="pre", norm="rms", causal=True),
        dict(placement="pre", norm="rms", causal=True, ff="swiglu", kv_heads=1, rope="half", bias=False),
        dict(placement="pre", norm="layer", causal=True, parallel=True),
        dict(placement="pre", norm="layer", causal=True, alibi=True),
    ]
    for v in variants:
        ff = v.get("ff", "mlp")
        parallel = v.get("parallel", False)
        kv = v.get("kv_heads", HEADS)
        p = block_params(rng, v["norm"], ff, parallel, kv, v.get("bias", True))
        xt = t(x, True)
        y = block(p, xt, v["placement"], v["norm"], v["causal"], ff, parallel, kv, v.get("rope"), v.get("alibi", False))
        (t(r) * y).sum().backward()
        assert xt.grad is not None
        out["blocks"].append(
            {
                "placement": v["placement"],
                "norm": v["norm"],
                "causal": v["causal"],
                "feedForward": ff,
                "parallel": parallel,
                "kvHeads": kv,
                "position": "rope" if v.get("rope") else "alibi" if v.get("alibi") else "none",
                "bias": v.get("bias", True),
                "params": p,
                "y": arr(y),
                "gradX": xt.grad.numpy(),
            }
        )
    return out


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    out: dict[str, object] = {}

    # Scaled dot-product attention with each mask and bias.
    qp, kp = np.arange(4, 7), np.arange(7)
    out["sdpa"] = {
        "plain": sdpa_case(rng, 3, 7, None, None, None),
        "causal": sdpa_case(rng, 3, 7, position_mask(qp, kp, True, None), None, None),
        "window": sdpa_case(rng, 3, 7, position_mask(qp, kp, True, 3), None, None),
        "padding": sdpa_case(
            rng, 3, 7, np.array([[[[1, 1, 1, 1, 1, 0, 0]]], [[[1, 1, 1, 1, 1, 1, 1]]]], dtype=float), None, None
        ),
        "alibi": sdpa_case(
            rng,
            3,
            7,
            position_mask(qp, kp, True, None),
            np.array([[[-m * abs(p - q) for q in kp] for p in qp] for m in alibi_slopes(3)]),
            None,
        ),
        "softcap": sdpa_case(rng, 3, 7, None, None, 2.0),
    }
    out["mha"] = mha_cases(rng)
    out["mla"] = mla_case(rng)
    out["ffn"] = ffn_cases(rng)
    out["block"] = block_cases(rng)

    # Positions.
    pos = np.arange(9)
    x = rng.normal(size=(2, 9, 8))
    out["rope"] = {
        "x": x,
        "half": arr(rope_half(t(x), pos, inv_freq_default(8))),
        "interleaved": arr(rope_interleaved(t(x), pos, inv_freq_default(8))),
        "base500": arr(rope_half(t(x), pos, inv_freq_default(8, 500.0))),
    }
    yarn_inv, yarn_mscale = inv_freq_yarn(64, 10000.0, 4.0, 2048)
    out["ropeFrequencies"] = {
        "default": inv_freq_default(64),
        "linear": inv_freq_default(64) / 4.0,
        "ntk": inv_freq_default(64, 10000.0 * 4.0 ** (64 / 62)),
        "yarn": yarn_inv,
        "yarnMagnitude": yarn_mscale,
    }
    pe = np.zeros((10, 6))
    for p in range(10):
        for i in range(3):
            pe[p, 2 * i] = math.sin(p / 10000 ** (2 * i / 6))
            pe[p, 2 * i + 1] = math.cos(p / 10000 ** (2 * i / 6))
    out["sinusoidal"] = pe
    out["alibiSlopes"] = {str(n): alibi_slopes(n) for n in [1, 2, 3, 4, 6, 8, 12, 16]}
    rel = torch.arange(-300, 301)
    out["t5"] = {
        "relative": rel.numpy(),
        "bidirectional": t5_bucket(rel, True, 32, 128).numpy(),
        "causal": t5_bucket(rel, False, 32, 128).numpy(),
        "small": t5_bucket(rel, True, 8, 20).numpy(),
    }
    return out
