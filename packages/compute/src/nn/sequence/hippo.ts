/**
 * Online polynomial projection (HiPPO). The state $\hvec(t)$ of the linear system
 * $\hvec' = \Amat\hvec + \Bmat u$ (LegT) or $\hvec' = (\Amat\hvec + \Bmat u)/t$ (LegS) holds the coefficients of the
 * best approximation of the input history by the first $N$ normalised Legendre polynomials, under the uniform measure
 * on a sliding window $[t - \theta, t]$ (LegT) or on the whole history $[0, t]$ (LegS). This file supplies the LegT
 * matrices in the same orthonormal basis as `hippoLegS`, an online run of either system over a sampled input by the
 * bilinear (Tustin) rule, the offline projection it approximates, and the reconstruction of the input from a state.
 *
 * Source: Gu, Dao, Ermon, Rudra and Ré (2020), "HiPPO: Recurrent Memory with Optimal Polynomial Projections",
 * Theorem 1 (LegT) and Theorem 2 (LegS), Appendix D.1 and D.3 for the derivations, and the generalised bilinear
 * transform of Appendix B.3. The paper writes LegT in the unnormalised Legendre basis with the sign convention
 * $\hvec' = -\Amat\hvec + \Bmat u$; here LegT uses the orthonormal basis $g_n = \sqrt{2n+1}\,P_n$ and the convention
 * $\hvec' = \Amat\hvec + \Bmat u$ of `hippoLegS`, so both measures reconstruct with the same basis. The LegS step at
 * sample $k$ uses $\delta = 1/k$, as the HiPPO reference code does. The online run and the projection loop over plain
 * arrays (a figure runs $T = 1000$ steps of $N \le 64$ states on every slider move); results are tensors.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import {
  add,
  arange,
  dense,
  div,
  fromData,
  mul,
  sqrt,
  sub,
  toFlat,
  type Tensor,
  type Value,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import { legendre } from 'aifn-compute/numerics/polynomial'
import { discretiseSsm } from 'aifn-compute/systems'
import { hippoLegS } from './ssm'

/** Which history a HiPPO state summarises: the whole past (`legs`) or a sliding window (`legt`). */
export type HippoMeasure = 'legs' | 'legt'

/** Options of `hippoOnline` and `hippoProject`. */
export type HippoOptions = {
  /** Which history the state summarises: the whole past (`legs`) or a window of length $\theta$ (`legt`). */
  measure: HippoMeasure
  /** The number of coefficients $N$. */
  order: Size
  /** The window length $\theta$ of LegT, in the input's time units (default 1; ignored by LegS). */
  theta?: number
}

/**
 * The HiPPO-LegT matrices of order $N$ and window $\theta$ in the orthonormal basis, for
 * $\hvec' = \Amat\hvec + \Bmat u$: $A_{nk} = -\sqrt{2n+1}\sqrt{2k+1}/\theta$ times 1 for $k < n$ and $(-1)^{n-k}$ for
 * $k \ge n$; $B_n = \sqrt{2n+1}/\theta$.
 *
 * @param order The number of states $N$.
 * @param theta The window length $\theta$.
 * @returns `A`, $N \times N$, and `B`, the $N \times 1$ input column (the shapes of `hippoLegS`).
 *
 * @example Order 3 over a window of 2
 * const { A, B } = hippoLegT(3, 2)
 * print('A =', A)
 * print('B =', B)
 */
export function hippoLegT(order: Size, theta: number): { A: Tensor; B: Tensor } {
  const A = new Float64Array(order * order)
  const B = new Float64Array(order)
  for (let n = 0; n < order; n++) {
    const rn = Math.sqrt(2 * n + 1)
    B[n] = rn / theta
    for (let k = 0; k < order; k++) {
      const rk = Math.sqrt(2 * k + 1)
      A[n * order + k] = (-(rn * rk) / theta) * (k < n ? 1 : (n - k) % 2 === 0 ? 1 : -1)
    }
  }
  return { A: fromData(A, [order, order]), B: fromData(B, [order, 1]) }
}

/**
 * Run the HiPPO system over samples $u_1, \dots, u_T$ taken every `dt` ($u_k$ at time $k\,\Delta t$), from
 * $\hvec_0 = 0$, and return every state $\hvec_1, \dots, \hvec_T$. LegT is time-invariant and is discretised once by
 * `discretiseSsm` (bilinear); LegS is time-varying, $\hvec' = (\Amat\hvec + \Bmat u)/t$, and takes the bilinear step
 * with $\delta = 1/k$ at sample $k$, solved by forward substitution because $\Amat$ is lower triangular.
 * $O(T N^2)$, on plain arrays (not differentiable).
 *
 * @param u The samples $u_1, \dots, u_T$: an array of numbers or a vector.
 * @param dt The sampling interval $\Delta t$ (used by LegT; LegS is invariant to the time scale).
 * @param options The measure, the order $N$ and LegT's window $\theta$.
 * @returns The states, $T \times N$: row $k - 1$ is $\hvec_k$.
 *
 * @example The LegS memory of a unit step at t = 1, read at t = 2
 * const dt = 0.01
 * const u = Array.from({ length: 200 }, (_, k) => ((k + 1) * dt > 1 ? 1 : 0))
 * const states = hippoOnline(u, dt, { measure: 'legs', order: 4 })
 * print('h(2) =', slice(states, 199))
 * print('exact (1/2, sqrt 3/4, 0, -sqrt 7/16) =', [0.5, Math.sqrt(3) / 4, 0, -Math.sqrt(7) / 16])
 */
export function hippoOnline(u: VectorLike | Value, dt: number, options: HippoOptions): Tensor {
  const N = options.order
  const x = dense.toF64(u as VectorLike, 'hippoOnline')
  const T = x.length
  const out = new Float64Array(T * N)
  if (options.measure === 'legt') {
    const { A, B } = hippoLegT(N, options.theta ?? 1)
    const d = discretiseSsm(A, B, dt, 'bilinear')
    const Ab = dense.toMatrixF64(d.A as Tensor, 'hippoOnline').data
    const Bb = toFlat(d.B as Tensor)
    let h = new Float64Array(N)
    for (let k = 0; k < T; k++) {
      const next = out.subarray(k * N, (k + 1) * N)
      for (let n = 0; n < N; n++) {
        let s = Bb[n] * x[k]
        for (let j = 0; j < N; j++) s += Ab[n * N + j] * h[j]
        next[n] = s
      }
      h = next
    }
    return fromData(out, [T, N])
  }
  const { A: At, B: Bt } = hippoLegS(N)
  const A = toFlat(At)
  const B = toFlat(Bt)
  let h = new Float64Array(N)
  const rhs = new Float64Array(N)
  for (let k = 1; k <= T; k++) {
    const delta = 1 / k
    // Right side: (I + δA/2) h + δ B u_k.
    for (let n = 0; n < N; n++) {
      let s = h[n] + delta * B[n] * x[k - 1]
      for (let j = 0; j <= n; j++) s += 0.5 * delta * A[n * N + j] * h[j]
      rhs[n] = s
    }
    // Left side: (I − δA/2) h_k = rhs, lower triangular.
    const next = out.subarray((k - 1) * N, k * N)
    for (let n = 0; n < N; n++) {
      let s = rhs[n]
      for (let j = 0; j < n; j++) s += 0.5 * delta * A[n * N + j] * next[j]
      next[n] = s / (1 - 0.5 * delta * A[n * N + n])
    }
    h = next
  }
  return fromData(out, [T, N])
}

/**
 * The window $[a, b]$ that a state at time $t$ summarises: $[0, t]$ for LegS, $[t - \theta, t]$ for LegT.
 *
 * @param measure The measure.
 * @param t The time of the state.
 * @param theta LegT's window length $\theta$ (default 1).
 * @returns The ends $a$ and $b$.
 *
 * @example The two windows at t = 3
 * print('LegS:', hippoWindow('legs', 3))
 * print('LegT, theta = 0.5:', hippoWindow('legt', 3, 0.5))
 */
export function hippoWindow(measure: HippoMeasure, t: number, theta = 1): [number, number] {
  return measure === 'legs' ? [0, t] : [t - theta, t]
}

/**
 * The orthonormal basis $g_n(x) = \sqrt{2n+1}\,P_n(z)$ on the window at time $t$, with $z = 2(x - a)/(b - a) - 1$
 * the window $[a, b]$ mapped to $[-1, 1]$. Broadcasts over `x` and differentiates in it.
 *
 * @param measure The measure, which fixes the window.
 * @param order The number of basis functions $N$.
 * @param t The time of the state.
 * @param theta LegT's window length $\theta$ (ignored by LegS).
 * @param x The points: a number or a tensor of any shape.
 * @returns $g_0(x), \dots, g_{N-1}(x)$ on a last axis of $N$, shape `[...shape(x), N]`.
 *
 * @example The basis at the newest point of the window is sqrt(2n + 1)
 * print('g(t) =', hippoBasis('legs', 4, 2, 1, 2))
 */
export function hippoBasis(measure: HippoMeasure, order: Size, t: number, theta: number, x: Value): Value {
  const [a, b] = hippoWindow(measure, t, theta)
  const z = sub(div(mul(2, sub(x, a)), b - a), 1)
  return mul(legendre(order, z), sqrt(add(mul(2, arange(order)), 1)))
}

/**
 * The input reconstructed from a state at time $t$, $u(x) \approx \sum_n h_n g_n(x)$, at each point of `xs`.
 *
 * @param state The coefficients $\hvec$, $N$ values (a row of `hippoOnline`'s result, or an array).
 * @param measure The measure the state was built under.
 * @param t The time of the state.
 * @param theta LegT's window length $\theta$ (ignored by LegS).
 * @param xs The points to evaluate at, normally inside the window.
 * @returns The reconstruction, one value per point of `xs`.
 *
 * @example A step reconstructed from its four-coefficient LegS memory
 * const h = [0.5, Math.sqrt(3) / 4, 0, -Math.sqrt(7) / 16]
 * print('u(2), u(0) =', hippoReconstruct(h, 'legs', 2, 1, [2, 0]))
 */
export function hippoReconstruct(
  state: VectorLike | Value,
  measure: HippoMeasure,
  t: number,
  theta: number,
  xs: VectorLike | Value,
): Tensor {
  const h = dense.toF64(state as VectorLike, 'hippoReconstruct')
  const N = h.length
  const points = dense.toF64(xs as VectorLike, 'hippoReconstruct')
  const g = toFlat(hippoBasis(measure, N, t, theta, fromData(points, [points.length])) as Tensor)
  const out = new Float64Array(points.length)
  for (let i = 0; i < points.length; i++) {
    let s = 0
    for (let n = 0; n < N; n++) s += h[n] * g[i * N + n]
    out[i] = s
  }
  return fromData(out, [points.length])
}

/**
 * The exact projection the online state approximates: $h_n = \frac{1}{\lvert W \rvert} \int_W u(x)\, g_n(x)\,dx$
 * over the window $W$ at time $t$, for samples $u_k$ at times $k\,\Delta t$ read as piecewise constant
 * ($u(x) = u_k$ on $((k-1)\Delta t, k\,\Delta t]$, as the online run sees them), by the midpoint rule with `refine`
 * points per sample. The part of a LegT window before time 0 contributes nothing.
 *
 * @param u The samples $u_1, \dots, u_T$.
 * @param dt The sampling interval $\Delta t$.
 * @param t The time of the projection.
 * @param options The measure, the order $N$ and LegT's window $\theta$.
 * @param refine The midpoints per sampling interval.
 * @returns The coefficients $h_0, \dots, h_{N-1}$.
 *
 * @example A unit step at t = 1, projected on the LegS basis at t = 2
 * const dt = 0.01
 * const u = Array.from({ length: 200 }, (_, k) => ((k + 1) * dt > 1 ? 1 : 0))
 * print('h =', hippoProject(u, dt, 2, { measure: 'legs', order: 4 }))
 */
export function hippoProject(u: VectorLike | Value, dt: number, t: number, options: HippoOptions, refine = 8): Tensor {
  const N = options.order
  const x = dense.toF64(u as VectorLike, 'hippoProject')
  const theta = options.theta ?? 1
  const [a, b] = hippoWindow(options.measure, t, theta)
  const coeff = new Float64Array(N)
  const h = dt / refine
  const start = Math.max(a, 0)
  const steps = Math.max(0, Math.round((b - start) / h))
  const points = new Float64Array(steps)
  for (let i = 0; i < steps; i++) points[i] = start + (i + 0.5) * h
  const g = toFlat(hippoBasis(options.measure, N, t, theta, fromData(points, [steps])) as Tensor)
  for (let i = 0; i < steps; i++) {
    const k = Math.min(x.length - 1, Math.max(0, Math.ceil(points[i] / dt) - 1))
    for (let n = 0; n < N; n++) coeff[n] += x[k] * g[i * N + n] * h
  }
  for (let n = 0; n < N; n++) coeff[n] /= b - a
  return fromData(coeff, [N])
}
