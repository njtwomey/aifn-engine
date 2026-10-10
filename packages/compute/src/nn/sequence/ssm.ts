/**
 * Linear state-space layers (Gu et al., 2022, S4): a continuous system
 * $\xvec'(t) = \Amat\xvec(t) + \Bmat u(t)$, $y(t) = \Cmat\xvec(t) + D u(t)$ is discretised with a step $\Delta$ into
 * $\xvec_k = \bar{\Amat}\xvec_{k-1} + \bar{\Bmat} u_k$, $y_k = \Cmat\xvec_k + D u_k$, which can be run two ways
 * that give the same outputs: as a recurrence (a linear scan, $O(L)$ sequential or $O(\log L)$ parallel rounds) or as a
 * causal convolution of the input with the kernel
 * $\bar{\Kmat} = (\Cmat\bar{\Bmat}, \Cmat\bar{\Amat}\bar{\Bmat}, \dots, \Cmat\bar{\Amat}^{L-1}\bar{\Bmat})$.
 * HiPPO-LegS supplies the $\Amat$ and $\Bmat$ that make the state track an optimal Legendre-polynomial summary of the
 * whole input history (Gu et al., 2020).
 *
 * Everything here is a composition of primitives (the matrix exponential, solves, the associative scan, gathers), so
 * it differentiates in $\Amat$, $\Bmat$, $\Cmat$, $D$ and $\Delta$, as a learned layer needs. Inputs are time first.
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
 * Elementwise discretisation of a diagonal system (S4D, Mamba), with $a$, $b$ and $\Delta$ broadcasting together. With
 * `zoh`, $\bar{a} = e^{\Delta a}$ and $\bar{b} = (e^{\Delta a} - 1) b / a$ (computed as
 * $\Delta\,\mathrm{expm1}(\Delta a)/(\Delta a)\, b$, which is $\Delta b$ where $\Delta a = 0$); with `bilinear`,
 * $\bar{a} = (1 + \Delta a/2)/(1 - \Delta a/2)$ and $\bar{b} = \Delta b/(1 - \Delta a/2)$; with `euler`,
 * $\bar{a} = 1 + \Delta a$ and $\bar{b} = \Delta b$. (Mamba's `selectiveScan` uses $\bar{a} = e^{\Delta a}$ with the
 * Euler $\bar{b} = \Delta b$.) Differentiable in every argument.
 *
 * @param a The diagonal of the continuous state matrix $\Amat$, one entry per state (negative for a stable system).
 * @param b The continuous input map $\Bmat$, entrywise with `a`.
 * @param step The step $\Delta$ (positive): a number, or a tensor broadcasting with `a` and `b` for a step per state.
 * @param method The discretisation rule: zero-order hold `zoh`, `bilinear` (Tustin) or forward `euler`.
 * @returns The discrete diagonal $\bar{a}$ as `A` and input map $\bar{b}$ as `B`, of the broadcast shape.
 *
 * @example Zero-order hold of two decaying states
 * const { A, B } = discretiseDiagonal(tensor([-1, -2]), tensor([1, 1]), 0.1)
 * print('A =', A)
 * print('B =', B)
 *
 * @example The three rules side by side, with a state at a = 0
 * const a = tensor([-1, 0])
 * const b = tensor([1, 1])
 * print('zoh:', discretiseDiagonal(a, b, 0.1, 'zoh'))
 * print('bilinear:', discretiseDiagonal(a, b, 0.1, 'bilinear'))
 * print('euler:', discretiseDiagonal(a, b, 0.1, 'euler'))
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
 * The HiPPO-LegS matrices of order $N$ (Gu et al., 2020, Theorem 2; S4's `transition('legs')`):
 * $A_{nk} = -\sqrt{2n+1}\sqrt{2k+1}$ for $n > k$, $-(n + 1)$ on the diagonal and 0 above; $B_n = \sqrt{2n+1}$. With
 * $\xvec' = \Amat\xvec/t + \Bmat u/t$ the state holds the coefficients of the best Legendre approximation of $u$ on
 * $[0, t]$; S4 uses $\Amat$ and $\Bmat$ with a learned step.
 *
 * @param order The number of states $N$ (the number of Legendre coefficients kept).
 * @returns `A`, the lower-triangular $N \times N$ state matrix, and `B`, the $N \times 1$ input column.
 *
 * @example Order 3
 * const { A, B } = hippoLegS(3)
 * print('A =', A)
 * print('B =', B)
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
 * The convolution kernel
 * $\bar{\Kmat} = (\Cmat\bar{\Bmat}, \Cmat\bar{\Amat}\bar{\Bmat}, \dots, \Cmat\bar{\Amat}^{L-1}\bar{\Bmat})$ of
 * length $L$ of a single-input discrete system. The causal convolution $\bar{\Kmat} * u$ equals the recurrence run from
 * a zero state (S4, §2.4). Computed by $L$ matrix–vector products, so it differentiates in $\bar{\Amat}$,
 * $\bar{\Bmat}$ and $\Cmat$; S4's fast Cauchy-kernel algorithm gives the same values.
 *
 * @param A The discrete state matrix $\bar{\Amat}$, $N \times N$.
 * @param B The discrete input map $\bar{\Bmat}$, $N \times 1$ or a vector of $N$.
 * @param C The output map $\Cmat$, $P \times N$ for $P$ outputs, or a vector of $N$ for one.
 * @param length The number of taps $L$, normally the length of the input it will be convolved with.
 * @returns The kernel, $L \times P$ (a vector of $L$ when `C` is a vector): row $k$ is
 *   $\Cmat\bar{\Amat}^k\bar{\Bmat}$.
 *
 * @example The kernel of a two-state diagonal system, and the convolution it defines
 * const A = tensor([[0.5, 0], [0, 0.25]])
 * const B = tensor([1, 1])
 * const C = tensor([1, 2])
 * const K = ssmKernel(A, B, C, 4)
 * print('K =', K)
 * print('y =', causalConvolution(K, tensor([1, 2, 0, -1])))
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
 * The causal convolution $y_t = \sum_{k \le t} K_k u_{t-k}$ of inputs $u$ with a kernel $K$: a lower-triangular
 * Toeplitz matrix gathered from $K$, times $u$. Differentiable in $K$ and $u$. This is the convolutional mode of a
 * state-space layer. The kernel may have any number of taps $n$: taps past $u$'s length $L$ are unused, and a shorter
 * kernel is read as zero past its last tap ($K_k = 0$ for $k \ge n$).
 *
 * @param kernel The taps $K_0, \dots, K_{n-1}$: a vector of $n$, or $[n, \dots]$ with one kernel per channel.
 * @param u The input, time first: a vector of $L$ for a vector kernel, or $[L, \dots]$ with the kernel's trailing shape
 *   (each channel convolved with its own kernel).
 * @returns The output $y$, with the shape of `u`.
 *
 * @example An impulse returns the kernel; a step returns its running sum
 * const K = tensor([1, 0.5, 0.25])
 * print('impulse:', causalConvolution(K, tensor([1, 0, 0])))
 * print('step:', causalConvolution(K, tensor([1, 1, 1])))
 *
 * @example Two channels, each with its own kernel
 * const K = tensor([[1, 2], [0.5, 1], [0.25, 0]])
 * print('y =', causalConvolution(K, tensor([[1, 1], [1, 0], [0, 0]])))
 */
export function causalConvolution(kernel: Value, u: Value): Value {
  const L = shapeOfValue(u)[0]
  const ks = shapeOfValue(kernel)
  const n = ks[0]
  const rest = ks.slice(1)
  const w = rest.reduce((p, q) => p * q, 1)
  // T[t, s] = K[t − s] for 0 ≤ t − s < n, else the padded zero row (index n, after the kernel's own taps).
  const padded = concat([kernel, zeros([1, ...rest])], 0)
  const ids = new Int32Array(L * L * w)
  for (let t = 0; t < L; t++)
    for (let s = 0; s < L; s++)
      for (let j = 0; j < w; j++) ids[(t * L + s) * w + j] = (s <= t && t - s < n ? t - s : n) * w + j
  const toeplitz = gather(reshape(padded, [-1]), ids, [L, L, ...rest])
  if (rest.length === 0) return squeeze(matmul(toeplitz, expandDims(u, 1)), 1)
  // Per channel: y[t, …] = Σ_s T[t, s, …] u[s, …].
  return sum(mul(toeplitz, expandDims(u, 0)), 1)
}

/** Options of `linearRecurrence` and `matrixRecurrence`. */
export type RecurrenceOptions = {
  /**
   * The state $\hvec_0$ before the first step, shaped as one step of the state (default zeros, so $\hvec_1 = \bvec_1$).
   */
  initial?: Value
}

/**
 * The linear recurrence $\hvec_t = \avec_t \odot \hvec_{t-1} + \bvec_t$ (elementwise), every $\hvec_t$ at once, by
 * the associative scan of the pairs $(\avec_t, \bvec_t)$ under
 * $(\avec, \bvec) \circ (\avec', \bvec') = (\avec\avec', \avec'\bvec + \bvec')$ (Blelloch, 1990, §1.4; Martin
 * and Cundy, 2018). Diagonal state-space layers, linear attention with decay and Mamba's selective scan are this
 * recurrence. Differentiable in `a`, `b` and the initial state.
 *
 * @param a The decays $\avec_t$: anything broadcasting to the shape of `b` (a number for one decay at every step and
 *   entry, or $[T, \dots]$).
 * @param b The inputs $\bvec_t$, time first: $[T, \dots]$. Its shape is the result's.
 * @param options The initial state $\hvec_0$.
 * @returns Every state $\hvec_1, \dots, \hvec_T$, with the shape of `b`.
 *
 * @example A running average with decay one half
 * print('h =', linearRecurrence(0.5, tensor([1, 1, 1])))
 * print('from h0 = 4:', linearRecurrence(0.5, tensor([1, 1, 1]), { initial: tensor(4) }))
 *
 * @example A decay per step, shared by two channels
 * print('h =', linearRecurrence(tensor([[1], [2], [3]]), tensor([[1, 1], [1, 1], [1, 1]])))
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
 * The matrix recurrence $\hvec_t = \Amat_t\hvec_{t-1} + \bvec_t$, every $\hvec_t$ at once, by the associative scan
 * of $(\Amat_t, \bvec_t)$ under $(\Amat, \bvec) \circ (\Amat', \bvec') = (\Amat'\Amat, \Amat'\bvec + \bvec')$.
 * The recurrent mode of a dense state-space layer, whose convolutional mode is `ssmKernel` with `causalConvolution`.
 * Differentiable in `A`, `b` and the initial state.
 *
 * @param A The transition matrices: $T \times N \times N$ for one per step, or one $N \times N$ for every step.
 * @param b The inputs $\bvec_t$, $T \times N$.
 * @param options The initial state $\hvec_0$, a vector of $N$.
 * @returns Every state $\hvec_1, \dots, \hvec_T$, $T \times N$.
 *
 * @example A quarter turn per step, from one kick
 * const A = tensor([[0, -1], [1, 0]])
 * print('h =', matrixRecurrence(A, tensor([[1, 0], [0, 0], [0, 0], [0, 0]])))
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
 * Run a discrete single-input, single-output system over inputs $u$ in recurrent mode, from a zero state:
 * $\xvec_k = \bar{\Amat}\xvec_{k-1} + \bar{\Bmat} u_k$ (by `matrixRecurrence`), $y_k = \Cmat\xvec_k + D u_k$.
 * Equal to `causalConvolution(ssmKernel(A, B, C, L), u)` plus $D u$. Differentiable in every argument.
 *
 * @param system The discrete system: `A` the $N \times N$ matrix $\bar{\Amat}$ and `B` the input map $\bar{\Bmat}$
 *   ($N \times 1$ or $N$ values), as `discretiseSsm` of `aifn-compute/systems` returns them.
 * @param C The output map $\Cmat$: $N$ values ($1 \times N$ or a vector).
 * @param u The scalar inputs $u_1, \dots, u_L$, a vector of $L$.
 * @param D The direct feedthrough $D$ (default 0).
 * @returns The `outputs` $y$, a vector of $L$, and the `states` $\xvec_k$, $L \times N$.
 *
 * @example The recurrence and the convolution agree
 * const A = tensor([[0.5, 0], [0, 0.25]])
 * const B = tensor([1, 1])
 * const C = tensor([1, 2])
 * const u = tensor([1, 2, 0, -1])
 * const { outputs, states } = ssmRecurrent({ A, B }, C, u)
 * print('recurrent:', outputs)
 * print('convolution:', causalConvolution(ssmKernel(A, B, C, 4), u))
 * print('states =', states)
 */
export function ssmRecurrent(system: DiscreteSsm, C: Value, u: Value, D: Value = 0): { outputs: Value; states: Value } {
  const b = reshape(system.B, [-1])
  const states = matrixRecurrence(system.A, mul(expandDims(u, 1), b))
  const outputs = add(squeeze(matmul(states, reshape(C, [-1, 1])), 1), mul(D, u))
  return { outputs, states }
}
