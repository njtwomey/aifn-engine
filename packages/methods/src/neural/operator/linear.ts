/**
 * Closed-form least-squares fits of linear operators from input–output function pairs, the baselines that show what
 * structure buys in operator learning.
 *
 * - `fitDenseOperator` fits an unrestricted linear map on grid values, $\hat{\uvec} = \avec\Gmat$ with $\Gmat$ an
 *   $n \times n$ matrix (ridge-regularised normal equations), the linear case of a network that reads and writes grid
 *   values.
 * - `fitFourierMultiplier` fits a translation-equivariant linear operator, one complex multiplier per Fourier mode of
 *   the periodic grid, which is a one-channel spectral convolution with all modes kept (Li et al., 2021, "Fourier
 *   Neural Operator for Parametric Partial Differential Equations", §4; `SpectralConv1d` of `aifn-compute/nn/layers`).
 *   Its estimate decouples by mode:
 *   $m_k = \sum_i \overline{A_{ik}} B_{ik} / (\sum_i \lvert A_{ik} \rvert^2 + \text{ridge})$.
 *   `applyFourierMultiplier` applies it on a grid of any length.
 * - `fitPhysicsInformedMultiplier` fits a space-time multiplier $m_k(t_j)$, $j = 1, \dots, J$, for the periodic heat
 *   equation from solved pairs observed at the final time and from the equation's residual on unsolved inputs:
 *   backward Euler in time, $(m_k(t_j) - m_k(t_{j-1}))/\Delta t + D(2\pi k)^2 m_k(t_j)$, weighted by each input's
 *   energy in mode $k$, with $m_k(0) = 1$ (the physics-informed neural operator loss of Li et al., 2024,
 *   "Physics-Informed Neural Operator for Learning Partial Differential Equations", §3, for a linear one-layer
 *   model). Each mode is a $J \times J$ least-squares problem; `applyModeMultiplier` applies the result to a series.
 *   It is specific to the heat equation (real multipliers); a general version would take the equation's symbol.
 *
 * Every fit takes milliseconds. Their error is measured by `relativeL2Error` of `aifn-compute/learning/metrics`.
 */

import { fromData, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import { irfft, rfft } from 'aifn-compute/foundation/fourier'
import { solveDense } from 'aifn-compute/numerics/linalg'
import type { FourierSeries1d } from 'aifn-methods/dynamics/pde'

/**
 * A tensor's values as a Float64Array (read only).
 *
 * @param t A float64 tensor.
 * @returns Its data, row-major.
 */
function data(t: Tensor): Float64Array {
  return readonlyData(t) as Float64Array
}

/**
 * The least-squares linear map on grid values: $\Gmat$ minimising
 * $\sum_i \lVert \avec_i\Gmat - \uvec_i \rVert^2 + \text{ridge}\,\lVert \Gmat \rVert_F^2$, by the normal
 * equations and `solveDense`. A singular system (possible only with `ridge` 0) gives the zero matrix.
 *
 * @param inputs The input functions on the grid, $N \times n$.
 * @param outputs The output functions on the same grid, $N \times n$.
 * @param ridge The ridge penalty (default $10^{-8}$); with $N < n$ it selects the minimum-norm solution.
 * @returns $\Gmat$, $n \times n$; predict with `matmul(inputs, G)`.
 *
 * @example Two pairs on three points pin down only part of the map
 * const A = tensor([[1, 0, 0], [0, 1, 0]])
 * const G = fitDenseOperator(A, tensor([[0, 1, 0], [0, 0, 1]]))
 * print('G =', G)
 */
export function fitDenseOperator(inputs: Tensor, outputs: Tensor, ridge = 1e-8): Tensor {
  const [N, n] = inputs.shape
  const A = data(inputs)
  const B = data(outputs)
  const gram = new Float64Array(n * n)
  const cross = new Float64Array(n * n)
  for (let i = 0; i < N; i++)
    for (let r = 0; r < n; r++) {
      const a = A[i * n + r]
      for (let c = 0; c < n; c++) {
        gram[r * n + c] += a * A[i * n + c]
        cross[r * n + c] += a * B[i * n + c]
      }
    }
  for (let r = 0; r < n; r++) gram[r * n + r] += ridge
  const { x } = solveDense(gram, cross, n)
  return fromData(x ?? new Float64Array(n * n), [n, n])
}

/** A complex multiplier per Fourier mode $k = 0, \dots, H - 1$ of a periodic grid. */
export type FourierMultiplier = {
  /** The real parts, $H$ values. */
  re: Float64Array
  /** The imaginary parts, $H$ values. */
  im: Float64Array
}

/**
 * The least-squares Fourier multiplier: per mode $k$, $m_k$ minimising
 * $\sum_i \lvert m_k A_{ik} - B_{ik} \rvert^2 + \text{ridge}\,\lvert m_k \rvert^2$, with $\Amat$ and $\Bmat$ the
 * real FFTs of the inputs and outputs along the grid.
 *
 * @param inputs The inputs, $N \times n$, on the periodic grid $x_j = j/n$.
 * @param outputs The outputs, $N \times n$, on the same grid.
 * @param ridge The ridge penalty (default $10^{-12}$).
 * @returns The multiplier for the $\lfloor n/2 \rfloor + 1$ non-negative frequencies.
 *
 * @example One pair of the heat equation recovers its decay at every mode it excites
 * import { periodicGrid } from 'aifn-methods/dynamics/pde'
 * const x = periodicGrid(16)
 * const a = Float64Array.from(x, (v) => Math.sin(2 * Math.PI * v) + Math.cos(6 * Math.PI * v))
 * const u = Float64Array.from(x, (v) => 0.5 * Math.sin(2 * Math.PI * v) + 0.1 * Math.cos(6 * Math.PI * v))
 * const m = fitFourierMultiplier(fromData(a, [1, 16]), fromData(u, [1, 16]))
 * print('m_1, m_3 =', m.re[1], m.re[3])
 */
export function fitFourierMultiplier(inputs: Tensor, outputs: Tensor, ridge = 1e-12): FourierMultiplier {
  const [N, n] = inputs.shape
  const H = Math.floor(n / 2) + 1
  const A = data(rfft(inputs, { axis: 1 }))
  const B = data(rfft(outputs, { axis: 1 }))
  const re = new Float64Array(H)
  const im = new Float64Array(H)
  for (let k = 0; k < H; k++) {
    let num_re = 0
    let num_im = 0
    let den = ridge
    for (let i = 0; i < N; i++) {
      const j = 2 * (i * H + k)
      const [ar, ai, br, bi] = [A[j], A[j + 1], B[j], B[j + 1]]
      num_re += ar * br + ai * bi
      num_im += ar * bi - ai * br
      den += ar * ar + ai * ai
    }
    re[k] = num_re / den
    im[k] = num_im / den
  }
  return { re, im }
}

/**
 * A Fourier multiplier applied to functions on a periodic grid of any length $n$: modes the multiplier does not cover
 * are set to zero. The multiplier is dimensionless, so a fit on one grid transfers to another.
 *
 * @param m The multiplier, from `fitFourierMultiplier`.
 * @param inputs The functions, $N \times n$, on the periodic grid $x_j = j/n$.
 * @returns The outputs, $N \times n$.
 *
 * @example Halve the first mode and remove the rest
 * const m = { re: Float64Array.of(1, 0.5), im: Float64Array.of(0, 0) }
 * const wave = (k, j) => Math.cos((2 * Math.PI * k * j) / 8)
 * const a = Float64Array.from({ length: 8 }, (_, j) => 1 + wave(1, j) + wave(3, j))
 * print('out =', applyFourierMultiplier(m, fromData(a, [1, 8])))
 */
export function applyFourierMultiplier(m: FourierMultiplier, inputs: Tensor): Tensor {
  const [N, n] = inputs.shape
  const H = Math.floor(n / 2) + 1
  const A = data(rfft(inputs, { axis: 1 }))
  const out = new Float64Array(2 * N * H)
  // Spectra scale with n under rfft's convention; the multiplier is dimensionless, so it applies unchanged.
  for (let i = 0; i < N; i++)
    for (let k = 0; k < Math.min(H, m.re.length); k++) {
      const j = 2 * (i * H + k)
      out[j] = m.re[k] * A[j] - m.im[k] * A[j + 1]
      out[j + 1] = m.re[k] * A[j + 1] + m.im[k] * A[j]
    }
  return irfft(fromData(out, [N, H], 'complex128'), { axis: 1, n })
}

/** Options of `fitPhysicsInformedMultiplier`. */
export type PhysicsInformedMultiplierOptions = {
  /** The solved pairs: inputs and their observed outputs at the final time. */
  train: { input: FourierSeries1d; output: FourierSeries1d }[]
  /** The unsolved inputs on which the residual is evaluated. */
  virtual: FourierSeries1d[]
  /** The diffusivity $D$ of $u_t = D u_{xx}$. */
  diffusivity: number
  /** The final time $T$. */
  time: number
  /** The number of time levels $J$ of the space-time output ($\Delta t = T/J$). */
  levels: number
  /** The weight $\lambda$ of the residual term (0: data only). */
  weight: number
  /** The ridge penalty pulling unidentified multipliers to 0 (default $10^{-10}$). */
  ridge?: number
}

/**
 * The space-time multiplier $m_k(t_j)$ minimising, mode by mode,
 *
 * $$
 * \sum_i \lvert m_k(T) A_{ik} - B_{ik} \rvert^2
 * + \lambda \sum_v \lvert A_{vk} \rvert^2 \sum_j \Delta t \Big(\frac{m_k(t_j) - m_k(t_{j-1})}{\Delta t}
 * + D(2\pi k)^2 m_k(t_j)\Big)^2 + \text{ridge} \sum_j m_k(t_j)^2
 * $$
 *
 * with $m_k(t_0) = 1$, for real multipliers (the heat equation's are real). The residual term needs no solutions,
 * only inputs. A mode whose normal equations are singular gets multipliers 0.
 *
 * @param options The solved and unsolved inputs, the equation, the time levels and the weights.
 * @returns For each mode $k = 1, \dots, K$ ($K$ the series' modes) the multipliers at $t_1, \dots, t_J$, row-major
 *   $K \times J$ (`trajectory`), and the final-time multiplier $m_k(T)$ (`final`).
 *
 * @example From the residual alone the fit is backward Euler, (1 + D (2 pi)^2 dt)^(-J) at mode 1
 * const f = { mean: 0, cos: Float64Array.of(1), sin: Float64Array.of(0) }
 * const options = { train: [], virtual: [f], diffusivity: 0.1, time: 0.1, levels: 10, weight: 1 }
 * const { final } = fitPhysicsInformedMultiplier(options)
 * print('m_1(T) =', final[0])
 * print('backward Euler =', (1 + 0.1 * (2 * Math.PI) ** 2 * 0.01) ** -10)
 * print('exact =', Math.exp(-0.1 * (2 * Math.PI) ** 2 * 0.1))
 */
export function fitPhysicsInformedMultiplier(options: PhysicsInformedMultiplierOptions): {
  trajectory: Float64Array
  final: Float64Array
} {
  const { train, virtual, diffusivity, time, levels: J, weight, ridge = 1e-10 } = options
  const K = (train[0]?.input ?? virtual[0]).cos.length
  const dt = time / J
  const trajectory = new Float64Array(K * J)
  const final = new Float64Array(K)
  for (let k = 0; k < K; k++) {
    // Energies and cross terms of mode k+1.
    let dataEnergy = 0
    let dataCross = 0
    for (const { input, output } of train) {
      dataEnergy += input.cos[k] ** 2 + input.sin[k] ** 2
      dataCross += input.cos[k] * output.cos[k] + input.sin[k] * output.sin[k]
    }
    let virtualEnergy = 0
    for (const v of virtual) virtualEnergy += v.cos[k] ** 2 + v.sin[k] ** 2
    const c = 1 + diffusivity * (2 * Math.PI * (k + 1)) ** 2 * dt // residual of level j: (c m_j − m_{j−1})/Δt
    const w = (weight * virtualEnergy) / dt // λ E Δt (1/Δt)²
    // Normal equations of the J unknowns m_1..m_J (tridiagonal, assembled densely: J is small).
    const M = new Float64Array(J * J)
    const rhs = new Float64Array(J)
    // Residual j (0-based) is c m_j − m_{j−1}: c² on (j, j); for j ≥ 1 also 1 on (j−1, j−1) and −c off the diagonal.
    for (let j = 0; j < J; j++) {
      M[j * J + j] += w * c * c + ridge
      if (j >= 1) {
        M[(j - 1) * J + (j - 1)] += w
        M[j * J + (j - 1)] -= w * c
        M[(j - 1) * J + j] -= w * c
      }
    }
    rhs[0] += w * c // the first residual (c m_1 − 1)/Δt
    M[(J - 1) * J + (J - 1)] += dataEnergy
    rhs[J - 1] += dataCross
    const { x } = solveDense(M, rhs, J)
    if (x) trajectory.set(x, k * J)
    final[k] = x ? x[J - 1] : 0
  }
  return { trajectory, final }
}

/**
 * The output series of a final-time multiplier applied to an input series (the mean is carried unchanged).
 *
 * @param final The multipliers $m_k(T)$ for $k = 1, \dots, K$, as `fitPhysicsInformedMultiplier` returns them.
 * @param input The input series.
 * @returns The output series.
 *
 * @example Scale two modes
 * const f = { mean: 1, cos: Float64Array.of(1, 1), sin: Float64Array.of(0, 2) }
 * const out = applyModeMultiplier(Float64Array.of(0.5, 0.1), f)
 * print('cos =', out.cos, 'sin =', out.sin)
 */
export function applyModeMultiplier(final: Float64Array, input: FourierSeries1d): FourierSeries1d {
  return { mean: input.mean, cos: input.cos.map((c, k) => c * final[k]), sin: input.sin.map((s, k) => s * final[k]) }
}
