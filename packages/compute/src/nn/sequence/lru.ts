/**
 * The linear recurrent unit (LRU) of Orvieto et al. (2023): a complex diagonal linear recurrence
 * $\hvec_k = \lambdab \odot \hvec_{k-1} + \gammab \odot (\Bmat u_k)$, read out as
 * $y_k = \operatorname{Re}(\Cmat\hvec_k) + D u_k$.
 *
 * The eigenvalues use the stable exponential parameterisation $\lambda_j = \exp(-e^{\nu_j} + i e^{\theta_j})$, so
 * $\lvert \lambda_j \rvert < 1$ for any real $(\nu_j, \theta_j)$ (the parameters are stored as $\log\nu_j$ and
 * $\log\theta_j$). They are initialised uniformly by area on the ring $r_{\min} \le \lvert\lambda\rvert \le r_{\max}$
 * with phases in $[0, \theta_{\max}]$, and the input is normalised by $\gamma_j = \sqrt{1 - \lvert\lambda_j\rvert^2}$,
 * so a white-noise input gives every state unit variance at initialisation. The recurrence runs as one complex
 * associative scan (`linearRecurrence`); the same output is the causal convolution of the input with the kernel
 * $K_k = \operatorname{Re}(\sum_j C_j \gamma_j \lambda_j^k B_j) + D\,\delta_k$, which is how S4 computes it.
 *
 * Source: Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu and De (2023), "Resurrecting Recurrent Neural Networks for
 * Long Sequences", eq. (7) (the recurrence), §3.3 and Lemma 2 (the ring initialisation), §3.4 (the normalisation
 * $\gamma$), §3.2.1 (the real part of the output). Single input and single output (the figure's case): `B` and `C`
 * hold one complex weight per state. Every function is a composition of primitives, so it differentiates in the
 * parameters, `B`, `C`, `D` and the input.
 */

import { uniform, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import {
  add,
  arange,
  complex,
  cos,
  exp,
  expandDims,
  expj,
  fromData,
  mul,
  neg,
  realPart,
  shapeOfValue,
  sin,
  sum,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { causalConvolution, linearRecurrence } from './ssm'

/** The ring on which `lruRingInit` draws the eigenvalues. */
export type LruRing = {
  /** The smallest eigenvalue modulus $r_{\min}$, in $[0, 1)$. */
  rMin: number
  /** The largest eigenvalue modulus $r_{\max}$, in $(r_{\min}, 1)$. */
  rMax: number
  /** Phases are drawn uniformly from $[0, \theta_{\max}]$ ($2\pi$ for the full ring, $\pi/10$ for Path-X). */
  maxPhase: number
}

/**
 * The LRU's eigenvalue parameters, $n$ values each (float64 vectors):
 * $\lambda_j = \exp(-e^{\log\nu_j} + i e^{\log\theta_j})$ and the input scale $\gamma_j = e^{\log\gamma_j}$, with
 * `nuLog`, `thetaLog` and `gammaLog` holding $\log\nu_j$, $\log\theta_j$ and $\log\gamma_j$.
 */
export type LruParams = {
  /** $\log \nu_j$: the modulus is $\lvert\lambda_j\rvert = \exp(-e^{\log\nu_j})$. */
  nuLog: Value
  /** $\log \theta_j$: the phase is $\theta_j = e^{\log\theta_j}$. */
  thetaLog: Value
  /** $\log \gamma_j$, the log of the input normalisation. */
  gammaLog: Value
}

/**
 * Draw $n$ eigenvalues uniformly by area on the ring $r_{\min} \le \lvert\lambda\rvert \le r_{\max}$ with phase in
 * $[0, \theta_{\max}]$ (Orvieto et al., 2023, Lemma 2):
 * $\nu = -\tfrac12 \log(u_1 (r_{\max}^2 - r_{\min}^2) + r_{\min}^2)$, $\theta = \theta_{\max} u_2$, and
 * $\gamma = \sqrt{1 - \lvert\lambda\rvert^2}$. The two uniforms of each eigenvalue are drawn one after the other
 * from `s`, eigenvalue by eigenvalue (a phase of 0 is raised to $10^{-12}$ so its log is finite).
 *
 * @param n The number of states $n$.
 * @param ring The moduli $r_{\min}$, $r_{\max}$ and the largest phase.
 * @param s The stream the $2n$ uniforms are drawn from, in order; it advances.
 * @returns The parameters `nuLog`, `thetaLog` and `gammaLog`, $n$ values each.
 *
 * @example Four eigenvalues near the unit circle with small phases
 * const p = lruRingInit(4, { rMin: 0.9, rMax: 0.999, maxPhase: Math.PI / 10 }, stream(0))
 * const { modulus } = lruEigenvalues(p)
 * print('|lambda| =', modulus)
 * print('phase =', exp(p.thetaLog))
 */
export function lruRingInit(n: Size, ring: LruRing, s: Stream): { nuLog: Tensor; thetaLog: Tensor; gammaLog: Tensor } {
  const nuLog = new Float64Array(n)
  const thetaLog = new Float64Array(n)
  const gammaLog = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    const u1 = uniform(s)
    const u2 = uniform(s)
    const nu = -0.5 * Math.log(u1 * (ring.rMax ** 2 - ring.rMin ** 2) + ring.rMin ** 2)
    const theta = Math.max(ring.maxPhase * u2, 1e-12)
    nuLog[j] = Math.log(nu)
    thetaLog[j] = Math.log(theta)
    gammaLog[j] = 0.5 * Math.log(1 - Math.exp(-2 * nu))
  }
  return { nuLog: fromData(nuLog, [n]), thetaLog: fromData(thetaLog, [n]), gammaLog: fromData(gammaLog, [n]) }
}

/**
 * The eigenvalues $\lambda_j$ as real and imaginary parts, with their moduli $\lvert\lambda_j\rvert$.
 *
 * @param p The eigenvalue parameters.
 * @returns `re`, `im` and `modulus`, $n$ values each.
 *
 * @example An eigenvalue of modulus 0.9 at phase pi/4
 * const nuLog = tensor([Math.log(-Math.log(0.9))])
 * const p = { nuLog, thetaLog: tensor([Math.log(Math.PI / 4)]), gammaLog: tensor([0]) }
 * print('lambda =', lruEigenvalues(p))
 */
export function lruEigenvalues(p: LruParams): { re: Value; im: Value; modulus: Value } {
  const modulus = exp(neg(exp(p.nuLog)))
  const phase = exp(p.thetaLog)
  return { re: mul(modulus, cos(phase)), im: mul(modulus, sin(phase)), modulus }
}

/**
 * The eigenvalues $\lambda_j = \lvert\lambda_j\rvert e^{i\theta_j}$ as a complex128 vector of $n$ entries.
 *
 * @param p The eigenvalue parameters.
 * @returns $\lambdab$, complex128, $n$ entries.
 *
 * @example The eigenvalue of modulus 0.9 at phase pi/4
 * const nuLog = tensor([Math.log(-Math.log(0.9))])
 * const p = { nuLog, thetaLog: tensor([Math.log(Math.PI / 4)]), gammaLog: tensor([0]) }
 * print('lambda =', lruLambda(p))
 */
export function lruLambda(p: LruParams): Value {
  return mul(exp(neg(exp(p.nuLog))), expj(exp(p.thetaLog)))
}

/** The single-input, single-output read-in and read-out of an LRU. */
export type LruReadout = {
  /** The input map $\Bmat$: one weight per state, $n$ entries (complex128, e.g. `complex(re, im)`, or real). */
  B: Value
  /** The output map $\Cmat$: one weight per state, $n$ entries (complex128 or real). */
  C: Value
  /** The skip weight $D$ (default 0). */
  D?: number
  /**
   * Scale the input by $\gamma_j = e^{\texttt{gammaLog}_j}$ (default true); false runs the unnormalised recurrence
   * $\hvec_k = \lambdab \odot \hvec_{k-1} + \Bmat u_k$.
   */
  normalise?: boolean
}

/**
 * The per-state input weights $\gamma_j B_j$ (just $B_j$ when not normalising).
 *
 * @param p The eigenvalue parameters (`gammaLog` is read).
 * @param readout The read-in `B` and the `normalise` flag.
 * @returns $\gamma \odot \Bmat$, $n$ entries.
 */
function inputWeights(p: LruParams, readout: LruReadout): Value {
  return readout.normalise === false ? readout.B : mul(exp(p.gammaLog), readout.B)
}

/**
 * Run a single-input, single-output LRU over $u$ (length $T$) by the complex associative scan (`linearRecurrence`).
 * Differentiable in the parameters, the readout and `u`.
 *
 * @param u The scalar inputs $u_1, \dots, u_T$, a vector of $T$.
 * @param p The eigenvalue parameters.
 * @param readout The maps $\Bmat$, $\Cmat$, the skip $D$ and whether to normalise.
 * @returns The output `y` (a float64 vector of $T$) and the `states` $\hvec_k$ ($T \times n$, complex128).
 *
 * @example The impulse response of one state is a damped cosine
 * const p = {
 *   nuLog: tensor([Math.log(-Math.log(0.9))]),
 *   thetaLog: tensor([Math.log(Math.PI / 4)]),
 *   gammaLog: tensor([0.5 * Math.log(1 - 0.81)]),
 * }
 * const { y } = lruScan(tensor([1, 0, 0, 0, 0]), p, { B: tensor([1]), C: tensor([1]) })
 * print('y =', y)
 * const k = tensor([0, 1, 2, 3, 4])
 * print('gamma 0.9^k cos(k pi/4) =', mul(Math.sqrt(0.19), mul(pow(0.9, k), cos(mul(k, Math.PI / 4)))))
 */
export function lruScan(u: Value, p: LruParams, readout: LruReadout): { y: Value; states: Value } {
  const lambda = lruLambda(p)
  // Inputs γ_j B_j u_k, T × n complex.
  const b = mul(expandDims(u, 1), expandDims(inputWeights(p, readout), 0))
  const states = linearRecurrence(lambda, b)
  const y = add(realPart(sum(mul(states, expandDims(readout.C, 0)), -1)), mul(readout.D ?? 0, u))
  return { y, states }
}

/**
 * The convolution kernel $K_k = \operatorname{Re}(\sum_j C_j \gamma_j B_j \lambda_j^k)$, $k = 0, \dots, L - 1$, plus
 * $D$ at $k = 0$, from powers of $\lambda$: $\lambda^k = \exp(k \log\lambda) = \exp(k(-e^{\nu} + i e^{\theta}))$.
 *
 * @param p The eigenvalue parameters.
 * @param readout The maps $\Bmat$, $\Cmat$, the skip $D$ and whether to normalise.
 * @param length The number of taps $L$.
 * @returns The kernel, a float64 vector of $L$.
 *
 * @example The white-noise gain of a normalised state is 1
 * const r = 0.99
 * const p = {
 *   nuLog: tensor([Math.log(-Math.log(r))]),
 *   thetaLog: tensor([Math.log(0.1)]),
 *   gammaLog: tensor([0.5 * Math.log(1 - r * r)]),
 * }
 * const re = lruKernel(p, { B: tensor([1]), C: tensor([1]) }, 3000)
 * const im = lruKernel(p, { B: tensor([1]), C: complex(tensor([0]), tensor([-1])) }, 3000)
 * print('sum |gamma lambda^k|^2 =', add(sum(mul(re, re)), sum(mul(im, im))))
 */
export function lruKernel(p: LruParams, readout: LruReadout, length: Size): Value {
  const logLambda = complex(neg(exp(p.nuLog)), exp(p.thetaLog))
  const ks = arange(length)
  const powers = exp(mul(expandDims(ks, 1), expandDims(logLambda, 0)))
  const weights = mul(readout.C, inputWeights(p, readout))
  const K = realPart(sum(mul(powers, expandDims(weights, 0)), -1))
  if (!readout.D) return K
  const delta = new Float64Array(length)
  delta[0] = readout.D
  return add(K, fromData(delta, [length]))
}

/**
 * The LRU output by causal convolution with its kernel (`causalConvolution` of `lruKernel`), equal to `lruScan`'s.
 *
 * @param u The scalar inputs, a vector of $T$.
 * @param p The eigenvalue parameters.
 * @param readout The maps $\Bmat$, $\Cmat$, the skip $D$ and whether to normalise.
 * @returns The output $y$, a float64 vector of $T$.
 *
 * @example Scan and convolution agree
 * const p = lruRingInit(8, { rMin: 0.5, rMax: 0.99, maxPhase: 2 * Math.PI }, stream(3))
 * const B = complex(ones([8]), full([8], 0.5))
 * const C = complex(full([8], 0.25), full([8], -0.25))
 * const readout = { B, C, D: 0.3 }
 * const u = tensor([1, -1, 0.5, 2, 0, -0.5])
 * print('scan =', lruScan(u, p, readout).y)
 * print('convolution =', lruConvolve(u, p, readout))
 */
export function lruConvolve(u: Value, p: LruParams, readout: LruReadout): Value {
  return causalConvolution(lruKernel(p, readout, shapeOfValue(u)[0]), u)
}
