/**
 * The matrix-memory cell of xLSTM (mLSTM) with exponential input gating and the max-based stabiliser, in its two
 * equal forms. Per token, with input gate $i_t = \exp(\tilde\imath_t)$ and forget gate $f_t = \sigma(\tilde f_t)$ or
 * $\exp(\tilde f_t)$,
 *
 * $$
 * \Cmat_t = f_t \Cmat_{t-1} + i_t \vvec_t \kvec_t^\top, \qquad \nvec_t = f_t \nvec_{t-1} + i_t \kvec_t, \qquad
 * \tilde\hvec_t = \frac{\Cmat_t \qvec_t}{\max(\lvert \nvec_t^\top \qvec_t \rvert, 1)}.
 * $$
 *
 * The stabiliser $m_t = \max(\log f_t + m_{t-1}, \tilde\imath_t)$ is the log of the largest weight
 * $i_s \prod_{j=s+1}^{t} f_j$ in the sums, so the stabilised states $\Cmat_t e^{-m_t}$ and $\nvec_t e^{-m_t}$ never
 * overflow; the threshold 1 becomes $e^{-m_t}$ and the output is unchanged.
 *
 * - `mlstmRecurrent`: the recurrence, every state at once. The stabiliser is the max-plus scan
 *   $m_t = F_t + \max_{s \le t}(\tilde\imath_s - F_s)$ with $F_t = \sum_{j \le t} \log f_j$ (an associative scan
 *   under `maximum`); the stabilised gates $f'_t = e^{\log f_t + m_{t-1} - m_t}$ and $i'_t = e^{\tilde\imath_t - m_t}$
 *   then drive `linearRecurrence`.
 * - `mlstmParallel`: the attention-like form, with the $T \times T$ gate matrix
 *   $D_{ts} = \exp(F_t - F_s + \tilde\imath_s - m_t)$ for $s \le t$ and 0 above the diagonal:
 *   $\tilde\Hmat = ((\Qmat\Kmat^\top) \circ \Dmat)\Vmat$, each row divided by the larger of its sum's magnitude
 *   and $e^{-m_t}$.
 *
 * Source: Beck et al. (2024), "xLSTM: Extended Long Short-Term Memory", §2.3 (mLSTM) and Appendix A (stabilisation and
 * the parallel forward pass), arXiv:2405.04517. The output gate $o_t$ and the projections are left to the caller (the
 * result is $\tilde\hvec_t$); keys are taken as given (the paper scales them by $1/\sqrt{d}$ in their projection).
 */

import {
  abs,
  add,
  arange,
  associativeScan,
  concat,
  cumsum,
  div,
  einsum,
  exp,
  expandDims,
  greaterEqual,
  matmul,
  max,
  maximum,
  mul,
  neg,
  shapeOfValue,
  slice,
  sub,
  sum,
  transpose,
  where,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { logSigmoid } from 'aifn-compute/numerics/special'
import { linearRecurrence } from './ssm'

/** The forget-gate activation: `sigmoid` ($f_t \in (0, 1)$) or `exp` ($f_t > 0$). */
export type MlstmForget = 'sigmoid' | 'exp'

/** Options of `mlstmRecurrent` and `mlstmParallel`. */
export type MlstmOptions = {
  /** The forget-gate activation (default `sigmoid`). */
  forget?: MlstmForget
}

/** The outputs of `mlstmRecurrent` and `mlstmParallel`. */
export type MlstmResult = {
  /** The cell outputs $\tilde\hvec_t$, $[T, d_v]$ (before the output gate). */
  outputs: Value
  /** The stabiliser $m_t$, $T$ values: the log of the largest weight $i_s \prod_{j=s+1}^{t} f_j$. */
  stabiliser: Value
  /** The cumulative log forget gates $F_t$, $T$ values. */
  logForget: Value
}

/**
 * The log forget gates $\log f_t$ from their pre-activations: $\log\sigma(\tilde f_t)$ for `sigmoid`, $\tilde f_t$
 * itself for `exp`.
 *
 * @param fPre The pre-activations $\tilde f_t$, $T$ values.
 * @param forget The forget-gate activation.
 * @returns $\log f_t$, $T$ values.
 */
function logForgetGates(fPre: Value, forget: MlstmForget): Value {
  return forget === 'exp' ? fPre : logSigmoid(fPre)
}

/**
 * The mLSTM recurrence, stabilised, every step at once: the stabiliser by a max-plus associative scan, then the
 * memory $\Cmat_t$ and normaliser $\nvec_t$ by `linearRecurrence` with the stabilised gates. One sequence, time
 * first, scalar gates (one head). Differentiable; equal to `mlstmParallel`.
 *
 * @param q The queries $\qvec_t$, $T \times d_k$.
 * @param k The keys $\kvec_t$, $T \times d_k$.
 * @param v The values $\vvec_t$, $T \times d_v$.
 * @param iPre The input-gate pre-activations $\tilde\imath_t$, $T$ values ($i_t = e^{\tilde\imath_t}$).
 * @param fPre The forget-gate pre-activations $\tilde f_t$, $T$ values.
 * @param options The forget-gate activation.
 * @returns The outputs $\tilde\hvec_t$ ($T \times d_v$), the stabiliser $m_t$ and the cumulative log forget gates
 *   $F_t$.
 *
 * @example Three steps: the third key rewrites the first with input gate 4 after a forget gate of one half
 * const e = tensor([[1, 0], [0, 1], [1, 0]])
 * const v = tensor([[1, 0], [0, 2], [3, 1]])
 * const iPre = tensor([0, 0, Math.log(4)])
 * const fPre = tensor([0, 0, Math.log(0.5)])
 * const { outputs, stabiliser } = mlstmRecurrent(e, e, v, iPre, fPre, { forget: 'exp' })
 * print('h =', outputs)
 * print('m =', stabiliser)
 */
export function mlstmRecurrent(
  q: Value,
  k: Value,
  v: Value,
  iPre: Value,
  fPre: Value,
  options: MlstmOptions = {},
): MlstmResult {
  const logF = logForgetGates(fPre, options.forget ?? 'sigmoid')
  const F = cumsum(logF)
  const m = add(F, associativeScan(maximum, sub(iPre, F)))
  // m_{t−1}, with m_0 := m_1 − log f_1 so that f'_1 = 1 (it multiplies the zero state C_0, so its value is unused).
  const first = sub(slice(m, [0, 1]), slice(logF, [0, 1]))
  const mPrev = concat([first, slice(m, [0, -1])])
  const fStab = exp(sub(add(logF, mPrev), m))
  const iStab = exp(sub(iPre, m))
  const writes = mul(expandDims(expandDims(iStab, -1), -1), einsum('tv,tk->tvk', v, k))
  const C = linearRecurrence(expandDims(expandDims(fStab, -1), -1), writes)
  const n = linearRecurrence(expandDims(fStab, -1), mul(expandDims(iStab, -1), k))
  const num = einsum('tvk,tk->tv', C, q)
  const den = maximum(abs(sum(mul(n, q), -1)), exp(neg(m)))
  return { outputs: div(num, expandDims(den, -1)), stabiliser: m, logForget: F }
}

/**
 * The mLSTM in its parallel form: a stabilised, gated attention matrix
 * $D_{ts} = \exp(F_t - F_s + \tilde\imath_s - m_t)$ (zero above the diagonal) times the scores,
 * $\Smat = (\Qmat\Kmat^\top) \circ \Dmat$, then
 * $\tilde\hvec_t = \sum_s S_{ts} \vvec_s / \max(\lvert \sum_s S_{ts} \rvert, e^{-m_t})$. Equal to `mlstmRecurrent`.
 * $O(T^2)$ time and memory. Differentiable.
 *
 * @param q The queries $\qvec_t$, $T \times d_k$.
 * @param k The keys $\kvec_t$, $T \times d_k$.
 * @param v The values $\vvec_t$, $T \times d_v$.
 * @param iPre The input-gate pre-activations $\tilde\imath_t$, $T$ values.
 * @param fPre The forget-gate pre-activations $\tilde f_t$, $T$ values.
 * @param options The forget-gate activation.
 * @returns The outputs $\tilde\hvec_t$ ($T \times d_v$), the stabiliser $m_t$ and the cumulative log forget gates
 *   $F_t$.
 *
 * @example The parallel and recurrent forms agree, even with gates far beyond float64's range
 * const s = stream(9)
 * const [q, k, v] = [normals(s, [6, 2]), normals(s, [6, 2]), normals(s, [6, 2])]
 * const iPre = add(normals(s, [6]), 30)
 * const fPre = add(normals(s, [6]), 20)
 * const par = mlstmParallel(q, k, v, iPre, fPre, { forget: 'exp' })
 * const rec = mlstmRecurrent(q, k, v, iPre, fPre, { forget: 'exp' })
 * print('log of the largest weight, m_T =', slice(par.stabiliser, 5))
 * print('parallel =', par.outputs)
 * print('recurrent =', rec.outputs)
 */
export function mlstmParallel(
  q: Value,
  k: Value,
  v: Value,
  iPre: Value,
  fPre: Value,
  options: MlstmOptions = {},
): MlstmResult {
  const F = cumsum(logForgetGates(fPre, options.forget ?? 'sigmoid'))
  // log D_ts = F_t − F_s + ĩ_s for s ≤ t, and −∞ above the diagonal.
  const logD = add(sub(expandDims(F, -1), expandDims(F, 0)), expandDims(iPre, 0))
  const T = shapeOfValue(F)[0]
  const steps = arange(T)
  const causal = greaterEqual(expandDims(steps, -1), expandDims(steps, 0))
  const masked = where(causal, logD, -Infinity)
  const m = max(masked, -1)
  const D = exp(sub(masked, expandDims(m, -1)))
  const scores = mul(matmul(q, transpose(k)), D)
  const den = maximum(abs(sum(scores, -1)), exp(neg(m)))
  return { outputs: div(matmul(scores, v), expandDims(den, -1)), stabiliser: m, logForget: F }
}
