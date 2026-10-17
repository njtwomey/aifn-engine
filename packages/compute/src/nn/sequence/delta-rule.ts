/**
 * The delta-rule fast-weight recurrence of DeltaNet and its gated form (Gated DeltaNet), in recurrent mode. The state
 * $\Smat_t \in \reals^{d_v \times d_k}$ maps keys to values. Each token takes one gradient step of size $\beta_t$ on
 * the regression loss $\tfrac12\lVert \Smat\kvec_t - \vvec_t \rVert^2$, after an optional decay $\alpha_t$:
 *
 * $$
 * \Smat_t = \alpha_t \Smat_{t-1} (\Imat - \beta_t \kvec_t \kvec_t^\top) + \beta_t \vvec_t \kvec_t^\top
 * = \alpha_t \Smat_{t-1} + \beta_t (\vvec_t - \alpha_t \Smat_{t-1} \kvec_t) \kvec_t^\top,
 * \qquad \ovec_t = \Smat_t \qvec_t.
 * $$
 *
 * With $\alpha_t = 1$ it is DeltaNet; with the additive write $\Smat_t = \Smat_{t-1} + \vvec_t\kvec_t^\top$ in place
 * of the correction it is causal linear attention without a normaliser. The state is stored values by keys,
 * $d_v \times d_k$ as in the papers, which is the transpose of the $d_k \times d_v$ state of
 * `linearAttentionRecurrent`. The loop runs token by token (the reference form, as Mamba's `selective_scan_ref`); the
 * chunked WY form of Yang et al. (2024) computes the same states with matrix products.
 *
 * Sources: Schlag, Irie and Schmidhuber (2021), "Linear Transformers Are Secretly Fast Weight Programmers"
 * (arXiv:2102.11174); Yang et al. (2024), "Parallelizing Linear Transformers with the Delta Rule over Sequence
 * Length" (arXiv:2406.06484), §2; Yang, Kautz and Hatamizadeh (2025), "Gated Delta Networks" (arXiv:2412.06464), §3.
 * Keys are used as given: the papers L2-normalise them first (pass normalised keys for the stable regime
 * $\beta_t \in (0, 2)$).
 */

import {
  add,
  get,
  matmul,
  mul,
  outer,
  shapeOfValue,
  slice,
  stack,
  sub,
  zeros,
  type Value,
} from 'aifn-compute/foundation/tensor'

/** Options of `deltaRule`. */
export type DeltaRuleOptions = {
  /** The decay $\alpha_t \in (0, 1]$: one number for every step, or $T$ values (default 1, DeltaNet). */
  decay?: number | Value
  /** The state before the first token, $d_v \times d_k$ (default zeros). */
  initial?: Value
}

/**
 * The outputs of `deltaRule`: `outputs` $\ovec_t = \Smat_t\qvec_t$, $[T, d_v]$, and every state `states`,
 * $[T, d_v, d_k]$.
 */
export type DeltaRuleResult = {
  /** The reads $\ovec_t = \Smat_t \qvec_t$, $T \times d_v$. */
  outputs: Value
  /** Every state $\Smat_1, \dots, \Smat_T$, $[T, d_v, d_k]$. */
  states: Value
}

/**
 * The (gated) delta rule over one sequence, token by token:
 * $\Smat_t = \alpha_t \Smat_{t-1} + \beta_t (\vvec_t - \alpha_t \Smat_{t-1} \kvec_t) \kvec_t^\top$ and
 * $\ovec_t = \Smat_t \qvec_t$. Differentiable in every argument (shapes are read with `shapeOfValue`, so it runs under
 * `grad` and `vmap`).
 *
 * @param q The queries $\qvec_t$, $T \times d_k$.
 * @param k The keys $\kvec_t$, $T \times d_k$ (normally of unit length).
 * @param v The values $\vvec_t$, $T \times d_v$.
 * @param beta The writing strengths $\beta_t$: a number for every step, or $T$ values (normally in $(0, 1]$; 1 replaces
 *   the value stored at a unit key outright).
 * @param options The decay $\alpha_t$ and the initial state.
 * @returns The outputs $[T, d_v]$ and every state $[T, d_v, d_k]$.
 *
 * @example Orthonormal keys with beta 1 overwrite: rewriting the first key replaces its value
 * const k = tensor([[1, 0], [0, 1], [1, 0]])
 * const v = tensor([[1, 2], [3, 4], [5, 6]])
 * const { states } = deltaRule(k, k, v, 1)
 * print('S_3 =', slice(states, 2))
 *
 * @example A key that overlaps two stored ones corrects both
 * const r = Math.SQRT1_2
 * const k = tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [r, r, 0]])
 * const { states, outputs } = deltaRule(k, k, tensor([[1], [2], [3], [4]]), 1)
 * print('S_4 =', slice(states, 3))
 * print('reads =', outputs)
 */
export function deltaRule(
  q: Value,
  k: Value,
  v: Value,
  beta: number | Value,
  options: DeltaRuleOptions = {},
): DeltaRuleResult {
  const [T, dk] = shapeOfValue(k)
  const dv = shapeOfValue(v)[1]
  const at = (x: number | Value, t: number): number | Value => (typeof x === 'number' ? x : get(x, t))
  let S: Value = options.initial ?? zeros([dv, dk])
  const states: Value[] = []
  const outputs: Value[] = []
  for (let t = 0; t < T; t++) {
    const kt = slice(k, t)
    const vt = slice(v, t)
    const alpha = at(options.decay ?? 1, t)
    const b = at(beta, t)
    const decayed = mul(alpha, S)
    // Prediction error of the decayed memory at this key, then one gradient step of size β_t.
    const error = sub(vt, matmul(decayed, kt))
    S = add(decayed, mul(b, outer(error, kt)))
    states.push(S)
    outputs.push(matmul(S, slice(q, t)))
  }
  return { outputs: stack(outputs), states: stack(states) }
}
