/**
 * Linear state-space layers (Gu et al., 2022, S4): a continuous system x′(t) = A x(t) + B u(t), y(t) = C x(t) + D u(t)
 * is discretised with a step Δ into x_k = Ā x_{k−1} + B̄ u_k, y_k = C x_k + D u_k, which can be run two ways that give
 * the same outputs: as a recurrence (a linear scan, O(L) sequential or O(log L) parallel rounds) or as a causal
 * convolution of the input with the kernel K̄ = (CB̄, CĀB̄, …, CĀ^{L−1}B̄). HiPPO-LegS supplies the A and B that make the
 * state track an optimal Legendre-polynomial summary of the whole input history (Gu et al., 2020).
 *
 * Everything here is a composition of primitives (the matrix exponential, solves, the associative scan, gathers), so
 * it differentiates in A, B, C, D and Δ, as a learned layer needs.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import {
  abs,
  add,
  associativeScan,
  broadcastTo,
  concat,
  div,
  exp,
  expandDims,
  expm1,
  fromData,
  gather,
  less,
  matmul,
  mul,
  reshape,
  shapeOfValue,
  slice,
  squeeze,
  stack,
  sub,
  sum,
  zeros,
  unwrap,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { DiscreteSsm, SsmDiscretisation } from 'aifn-compute/systems'

/**
 * Elementwise discretisation of a diagonal system (S4D, Mamba): a, b and Δ broadcast together; with `zoh`
 * Ā = e^{Δa} and B̄ = (e^{Δa} − 1)/a·b, with `bilinear` Ā = (1 + Δa/2)/(1 − Δa/2) and B̄ = Δb/(1 − Δa/2), with `euler`
 * Ā = 1 + Δa and B̄ = Δb. (Mamba uses Ā = e^{Δa} with the Euler B̄ = Δb, `selectiveScan`.)
 */
export function discretiseDiagonal(a: Value, b: Value, step: Value, method: SsmDiscretisation = 'zoh'): DiscreteSsm {
  const da = mul(step, a)
  if (method === 'euler') return { A: add(1, da), B: mul(step, b) }
  if (method === 'bilinear') {
    const denom = sub(1, mul(0.5, da))
    return { A: div(add(1, mul(0.5, da)), denom), B: div(mul(step, b), denom) }
  }
  // B̄ = Δ·φ(Δa)·b with φ(z) = (e^z − 1)/z = expm1(z)/z: expm1 avoids the cancellation of e^z − 1 for small Δa, and
  // φ(0) = 1 is taken where Δa = 0 (the replaced divisor keeps the unused branch finite, so gradients stay finite).
  const zero = less(abs(unwrap(da)), Number.MIN_VALUE)
  const safe = where(zero, 1, da)
  const phi = where(zero, 1, div(expm1(safe), safe))
  return { A: exp(da), B: mul(mul(step, phi), b) }
}

/**
 * The HiPPO-LegS matrices of order N (Gu et al., 2020, Theorem 2; S4's `transition('legs')`): A[n, k] =
 * −√(2n+1)√(2k+1) for n > k, −(n + 1) on the diagonal and 0 above; B[n] = √(2n+1). With x′ = A x/t + B u/t the state
 * holds the coefficients of the best Legendre approximation of u on [0, t]; S4 uses A and B with a learned step.
 */
export function hippoLegS(order: Size): { A: Tensor; B: Tensor } {
  const A = new Float64Array(order * order)
  const B = new Float64Array(order)
  for (let n = 0; n < order; n++) {
    B[n] = Math.sqrt(2 * n + 1)
    for (let k = 0; k <= n; k++) A[n * order + k] = n === k ? -(n + 1) : -Math.sqrt(2 * n + 1) * Math.sqrt(2 * k + 1)
  }
  return { A: fromData(A, [order, order]), B: fromData(B, [order, 1]) }
}

/**
 * The convolution kernel K̄ = (C B̄, C Ā B̄, …, C Ā^{L−1} B̄) of length L of a single-input discrete system: Ā [N, N],
 * B̄ [N, 1] (or [N]), C [P, N] (or [N]); shape [L, P] (or [L]). y = K̄ * u (causal) equals the recurrence from a zero
 * state (S4, §2.4). Computed by L matrix–vector products, so it differentiates in Ā, B̄ and C; S4's fast Cauchy-kernel
 * algorithm gives the same values.
 */
export function ssmKernel(A: Value, B: Value, C: Value, length: Size): Value {
  const b = shapeOfValue(B).length === 1 ? expandDims(B, 1) : B
  const vectorC = shapeOfValue(C).length === 1
  const c = vectorC ? expandDims(C, 0) : C
  const taps: Value[] = []
  let v = b
  for (let k = 0; k < length; k++) {
    taps.push(squeeze(matmul(c, v), 1))
    if (k + 1 < length) v = matmul(A, v)
  }
  const K = stack(taps, 0)
  return vectorC ? reshape(K, [length]) : K
}

/**
 * The causal convolution y_t = Σ_{k ≤ t} K_k u_{t−k} of inputs u [L] (or [L, ...] with K broadcasting over the rest)
 * with a kernel K [L] (or [L, ...]): a lower-triangular Toeplitz matrix gathered from K, times u. Differentiable in K
 * and u. This is the convolutional mode of a state-space layer.
 */
export function causalConvolution(kernel: Value, u: Value): Value {
  const L = shapeOfValue(u)[0]
  const ks = shapeOfValue(kernel)
  const rest = ks.slice(1)
  const w = rest.reduce((p, q) => p * q, 1)
  // T[t, s] = K[t − s] for s ≤ t, else a padded zero row (index L).
  const padded = concat([kernel, zeros([1, ...rest])], 0)
  const ids = new Int32Array(L * L * w)
  for (let t = 0; t < L; t++)
    for (let s = 0; s < L; s++) for (let j = 0; j < w; j++) ids[(t * L + s) * w + j] = (s <= t ? t - s : L) * w + j
  const toeplitz = gather(reshape(padded, [-1]), ids, [L, L, ...rest])
  if (rest.length === 0) return squeeze(matmul(toeplitz, expandDims(u, 1)), 1)
  // Per channel: y[t, …] = Σ_s T[t, s, …] u[s, …].
  return sum(mul(toeplitz, expandDims(u, 0)), 1)
}

/** Options of `linearRecurrence`. */
export type RecurrenceOptions = {
  /** The state before the first step (default zeros). */
  initial?: Value
}

/**
 * The linear recurrence h_t = a_t ⊙ h_{t−1} + b_t (elementwise; a and b [T, ...] broadcasting together), every h_t at
 * once, by the associative scan of the pairs (a_t, b_t) under (a, b) ∘ (a′, b′) = (a a′, a′ b + b′) (Blelloch, 1990,
 * §1.4; Martin and Cundy, 2018). Diagonal state-space layers, linear attention with decay and Mamba's selective scan
 * are this recurrence.
 */
export function linearRecurrence(a: Value, b: Value, options: RecurrenceOptions = {}): Value {
  const shape = shapeOfValue(b)
  const A = broadcastTo(a, shape)
  let B = b
  if (options.initial !== undefined) {
    const first = add(mul(slice(A, [0, 1]), expandDims(options.initial, 0)), slice(b, [0, 1]))
    B = concat([first, slice(b, [1, shape[0]])], 0)
  }
  const [, h] = associativeScan(
    ([a1, b1]: readonly Value[], [a2, b2]: readonly Value[]) => [mul(a1, a2), add(mul(a2, b1), b2)],
    [A, B],
  )
  return h
}

/**
 * The matrix recurrence h_t = A_t h_{t−1} + b_t with A_t [N, N] (A [T, N, N], or one [N, N] for every step) and b
 * [T, N], every h_t at once by the associative scan of (A_t, b_t) under (A, b) ∘ (A′, b′) = (A′A, A′b + b′). The
 * recurrent mode of a dense state-space layer, `ssmKernel`'s convolution in the other mode.
 */
export function matrixRecurrence(A: Value, b: Value, options: RecurrenceOptions = {}): Value {
  const [T, n] = shapeOfValue(b)
  const As = shapeOfValue(A).length === 2 ? broadcastTo(A, [T, n, n]) : A
  let B = b
  if (options.initial !== undefined) {
    const first = add(squeeze(matmul(slice(As, 0), expandDims(options.initial, 1)), 1), slice(b, 0))
    B = concat([expandDims(first, 0), slice(b, [1, T])], 0)
  }
  const [, h] = associativeScan(
    ([A1, b1]: readonly Value[], [A2, b2]: readonly Value[]) => [
      matmul(A2, A1),
      add(squeeze(matmul(A2, expandDims(b1, -1)), -1), b2),
    ],
    [As, B],
  )
  return h
}

/**
 * Run a discrete single-input system over inputs u [L] in recurrent mode: x_k = Ā x_{k−1} + B̄ u_k (by
 * `matrixRecurrence`), y_k = C x_k + D u_k. Returns the outputs [L] and the states [L, N]. Equal to
 * `causalConvolution(ssmKernel(Ā, B̄, C, L), u) + D u`.
 */
export function ssmRecurrent(system: DiscreteSsm, C: Value, u: Value, D: Value = 0): { outputs: Value; states: Value } {
  const b = reshape(system.B, [-1])
  const states = matrixRecurrence(system.A, mul(expandDims(u, 1), b))
  const outputs = add(squeeze(matmul(states, reshape(C, [-1, 1])), 1), mul(D, u))
  return { outputs, states }
}
