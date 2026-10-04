"""Golden values for aifn-compute/nn/decoding's logit processors: transcriptions in torch of Hugging Face's
`RepetitionPenaltyLogitsProcessor`, `TemperatureLogitsWarper`, `TopKLogitsWarper` and `TopPLogitsWarper`, applied
alone and in that order, on random logits with heavy-tailed values and a prefix with repeats."""

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def repetition(scores: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
    score = torch.gather(scores, 1, ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return scores.scatter(1, ids, score)


def top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.size(-1))
    remove = scores < torch.topk(scores, k)[0][..., -1, None]
    return scores.masked_fill(remove, -float("inf"))


def top_p(scores: torch.Tensor, p: float, min_keep: int = 1) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove_sorted = cumulative <= (1 - p)
    remove_sorted[..., -min_keep:] = 0
    remove = remove_sorted.scatter(1, sorted_indices, remove_sorted)
    return scores.masked_fill(remove, -float("inf"))


def cases() -> dict[str, object]:
    rng = np.random.default_rng(11)
    out = []
    for _ in range(6):
        V = 12
        logits = rng.standard_t(3, size=V) * 2
        prefix = rng.integers(0, V, size=5)
        s = torch.tensor(logits)[None]
        ids = torch.tensor(prefix)[None]
        k, p, temp, pen = int(rng.integers(1, V)), float(rng.uniform(0.3, 0.95)), float(rng.uniform(0.4, 2)), 1.3
        combined = top_p(top_k(repetition(s, ids, pen) / temp, k), p)
        out.append(
            {
                "logits": logits,
                "prefix": prefix,
                "k": k,
                "p": p,
                "temperature": temp,
                "penalty": pen,
                "topK": top_k(s, k)[0].numpy(),
                "topP": top_p(s, p)[0].numpy(),
                "repetition": repetition(s, ids, pen)[0].numpy(),
                "combined": combined[0].numpy(),
                "filtered": combined.softmax(-1)[0].numpy(),
            }
        )
    return {"cases": out}
