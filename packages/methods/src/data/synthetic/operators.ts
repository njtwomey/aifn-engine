/**
 * Training pairs for operator learning: input functions drawn from a Gaussian prior and the solutions of a PDE or an
 * integral operator applied to them.
 *
 * - `periodicGaussianField` draws periodic functions on $[0, 1)$ as Fourier series from the prior
 *   $\Gauss(0, \sigma^2(-\Delta + \tau^2\Imat)^{-\alpha})$ of Li et al. (2021, "Fourier Neural Operator for Parametric
 *   Partial Differential Equations", §5.1), whose Karhunen–Loève coefficients have variance
 *   $\sigma^2((2\pi k)^2 + \tau^2)^{-\alpha}$ (`priorVariance`).
 * - `heatOperatorData` maps each draw to the exact solution of the periodic heat equation $u_t = D u_{xx}$ at time $t$
 *   (`heatSolution` of `aifn-methods/dynamics/pde`).
 * - `burgersOperatorData` maps each draw to the solution of the viscous Burgers equation $u_t + (u^2/2)_x = \nu u_{xx}$
 *   at time $t$ by `solveBurgersPeriodic`, solved on a finer grid and subsampled.
 * - `antiderivativeData` draws functions on $[0, 1]$ from a Gaussian process with an RBF kernel and returns them with
 *   their antiderivatives $G(u)(y) = \int_0^y u(s)\,ds$ by the cumulative trapezoid rule, the first example of Lu et
 *   al. (2021, "Learning nonlinear operators via DeepONet", §4.1).
 *
 * The grids, viscosity and times are small enough to generate in a browser; Li et al. solve Burgers on 8192 points with
 * $\nu = 0.1$ and $t = 1$. Every generator is a plain function of a stream and options, so a figure can run it in the
 * compute worker.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import { rbf } from 'aifn-compute/learning/kernels'
import {
  evaluateSeries,
  heatSolution,
  periodicGrid,
  solveBurgersPeriodic,
  type FourierSeries1d,
} from 'aifn-methods/dynamics/pde'
import { samplePrior } from 'aifn-methods/learning/gaussian-processes'

/** The prior $\Gauss(0, \sigma^2(-\Delta + \tau^2\Imat)^{-\alpha})$ on periodic functions of $[0, 1)$. */
export type PeriodicPrior = {
  /** The inverse length scale $\tau > 0$ (default 5). */
  tau?: number
  /** The smoothness exponent $\alpha > 1/2$ (default 2). */
  alpha?: number
  /** The scale $\sigma$ (default $\tau^\alpha$, so that the mean has variance 1). */
  sigma?: number
  /** The number of Fourier modes $K$ kept (default 32). */
  modes?: number
  /** Draw the constant mode too (default false: zero-mean draws). */
  withMean?: boolean
}

/**
 * The variance of the coefficient of $\cos 2\pi kx$ (and of $\sin 2\pi kx$) under the prior:
 * $2\sigma^2((2\pi k)^2 + \tau^2)^{-\alpha}$ for $k \ge 1$, $\sigma^2\tau^{-2\alpha}$ for the mean.
 *
 * @param k The mode $k$ (0 for the mean).
 * @param prior The prior's $\tau$, $\alpha$ and $\sigma$.
 * @returns The variance.
 *
 * @example The variance falls off as k^(-4) for alpha = 2
 * for (const k of [0, 1, 2, 4]) print(k, priorVariance(k))
 */
export function priorVariance(k: number, prior: PeriodicPrior = {}): number {
  const tau = prior.tau ?? 5
  const alpha = prior.alpha ?? 2
  const sigma = prior.sigma ?? tau ** alpha
  const lambda = sigma ** 2 * ((2 * Math.PI * k) ** 2 + tau ** 2) ** -alpha
  return k === 0 ? lambda : 2 * lambda
}

/**
 * `count` functions drawn from the periodic prior, as Fourier series: one block of $\text{count} \times (2K + 1)$
 * standard normals from `s` (per function the mean, then the $K$ cosines, then the $K$ sines), scaled by the prior's
 * standard deviations.
 *
 * @param s The stream the coefficients are drawn from.
 * @param count The number of functions.
 * @param prior The prior's $\tau$, $\alpha$, $\sigma$, number of modes and whether the mean is drawn.
 * @returns One series per function.
 *
 * @example Two smooth periodic functions on eight points
 * import { evaluateSeries, periodicGrid } from 'aifn-methods/dynamics/pde'
 * const [f, g] = periodicGaussianField(stream(1), 2, { modes: 8 })
 * print('f =', evaluateSeries(f, periodicGrid(8)))
 * print('g =', evaluateSeries(g, periodicGrid(8)))
 */
export function periodicGaussianField(s: Stream, count: number, prior: PeriodicPrior = {}): FourierSeries1d[] {
  const K = prior.modes ?? 32
  const z = readonlyData(normals(s, [count, 2 * K + 1])) as Float64Array
  const sd = Array.from({ length: K + 1 }, (_, k) => Math.sqrt(priorVariance(k, prior)))
  return Array.from({ length: count }, (_, i) => {
    const row = z.subarray(i * (2 * K + 1), (i + 1) * (2 * K + 1))
    return {
      mean: prior.withMean ? sd[0] * row[0] : 0,
      cos: Float64Array.from({ length: K }, (_, k) => sd[k + 1] * row[1 + k]),
      sin: Float64Array.from({ length: K }, (_, k) => sd[k + 1] * row[1 + K + k]),
    }
  })
}

/** Input and output functions on a common grid, one row per instance. */
export type OperatorPairs = {
  /** The grid, $n$ points. */
  x: Float64Array
  /** The input functions at the grid, $[\text{count}, n]$. */
  inputs: Tensor
  /** The output functions at the grid, $[\text{count}, n]$. */
  outputs: Tensor
}

/** Options of `heatOperatorData`. */
export type HeatOperatorOptions = {
  /** The number of pairs. */
  count: number
  /** The number of grid points $n$ on $[0, 1)$. */
  grid: number
  /** The diffusivity $D > 0$. */
  diffusivity: number
  /** The output time $t > 0$. */
  time: number
  /** The prior of the inputs (default `PeriodicPrior`'s defaults). */
  prior?: PeriodicPrior
}

/**
 * Pairs $(u_0, u(\cdot, t))$ for the periodic heat equation $u_t = D u_{xx}$ on $[0, 1)$, with $u_0$ from the periodic
 * prior and the exact solution, sampled on a uniform grid. Also returns the series, so either can be evaluated on
 * another grid.
 *
 * @param s The stream the inputs are drawn from.
 * @param options The number of pairs, the grid, the diffusivity, the time and the prior.
 * @param options.count The number of pairs.
 * @param options.grid The number of grid points $n$ on $[0, 1)$.
 * @param options.diffusivity The diffusivity $D > 0$.
 * @param options.time The output time $t > 0$.
 * @param options.prior The prior of the inputs (`PeriodicPrior`'s defaults when left out).
 * @returns The grid, inputs and outputs, and each pair's `series`.
 *
 * @example Three pairs on sixteen points: the outputs are smoother
 * const d = heatOperatorData(stream(2), { count: 3, grid: 16, diffusivity: 0.1, time: 0.1 })
 * print('input 0 =', slice(d.inputs, 0))
 * print('output 0 =', slice(d.outputs, 0))
 */
export function heatOperatorData(
  s: Stream,
  { count, grid, diffusivity, time, prior = {} }: HeatOperatorOptions,
): OperatorPairs & { series: { input: FourierSeries1d; output: FourierSeries1d }[] } {
  const x = periodicGrid(grid)
  const series = periodicGaussianField(s, count, prior).map((input) => ({
    input,
    output: heatSolution(input, diffusivity, time),
  }))
  const inputs = new Float64Array(count * grid)
  const outputs = new Float64Array(count * grid)
  series.forEach(({ input, output }, i) => {
    inputs.set(evaluateSeries(input, x), i * grid)
    outputs.set(evaluateSeries(output, x), i * grid)
  })
  return { x, inputs: fromData(inputs, [count, grid]), outputs: fromData(outputs, [count, grid]), series }
}

/** Options of `burgersOperatorData`. */
export type BurgersOperatorOptions = {
  /** The number of pairs. */
  count: number
  /** The number of output grid points $n$ on $[0, 1)$; it must divide `solverGrid`. */
  grid: number
  /** The viscosity $\nu > 0$. */
  viscosity: number
  /** The output time $t > 0$. */
  time: number
  /** The number of points of the solver's grid, even (default 128). */
  solverGrid?: number
  /**
   * The solver's time steps (default: enough that $\Delta t \le 0.4\,\Delta x / \max\lvert u_0 \rvert$, at least
   * 50).
   */
  steps?: number
  /** The prior of the inputs (default `PeriodicPrior`'s defaults). */
  prior?: PeriodicPrior
}

/**
 * Pairs $(u_0, u(\cdot, t))$ for the viscous Burgers equation $u_t + (u^2/2)_x = \nu u_{xx}$ on the periodic domain
 * $[0, 1)$, $u_0$ from the periodic prior. The solver (`solveBurgersPeriodic`) is Fourier pseudo-spectral with 2/3
 * dealiasing and integrating-factor RK4, run on `solverGrid` points for all instances at once and subsampled to `grid`
 * points. Throws `DomainError` when `grid` does not divide `solverGrid`.
 *
 * @param s The stream the inputs are drawn from.
 * @param options The number of pairs, the grids, the viscosity, the time, the steps and the prior.
 * @param options.count The number of pairs.
 * @param options.grid The number of output grid points $n$ on $[0, 1)$; it must divide `solverGrid`.
 * @param options.viscosity The viscosity $\nu > 0$.
 * @param options.time The output time $t > 0$.
 * @param options.solverGrid The number of points of the solver's grid, even.
 * @param options.steps The solver's time steps; left out, enough that $\Delta t \le 0.4\,\Delta x / \max\lvert u_0 \rvert$, at least 50.
 * @param options.prior The prior of the inputs (`PeriodicPrior`'s defaults when left out).
 * @returns The subsampled grid, inputs and outputs, with the solver's grid `solverX` and its `fine` inputs and outputs.
 *
 * @example Four pairs on 32 points from a 64-point solve
 * const d = burgersOperatorData(stream(3), { count: 4, grid: 32, viscosity: 0.05, time: 0.2, solverGrid: 64 })
 * print('shapes:', shapeOf(d.inputs), shapeOf(d.outputs), shapeOf(d.fine.outputs))
 */
export function burgersOperatorData(
  s: Stream,
  { count, grid, viscosity, time, solverGrid = 128, steps, prior = {} }: BurgersOperatorOptions,
): OperatorPairs & { solverX: Float64Array; fine: { inputs: Tensor; outputs: Tensor } } {
  const M = solverGrid
  if (M % grid !== 0) throw new DomainError('burgersOperatorData', 'burgersOperatorData: grid must divide solverGrid')
  const xs = periodicGrid(M)
  const u0 = new Float64Array(count * M)
  periodicGaussianField(s, count, prior).forEach((f, i) => u0.set(evaluateSeries(f, xs), i * M))
  const inputs = fromData(u0, [count, M])
  const outputs = solveBurgersPeriodic(inputs, { viscosity, time, steps })
  const stride = M / grid
  const sub = (u: Float64Array) => {
    const out = new Float64Array(count * grid)
    for (let i = 0; i < count; i++) for (let j = 0; j < grid; j++) out[i * grid + j] = u[i * M + j * stride]
    return fromData(out, [count, grid])
  }
  return {
    x: periodicGrid(grid),
    inputs: sub(u0),
    outputs: sub(readonlyData(outputs) as Float64Array),
    solverX: xs,
    fine: { inputs, outputs },
  }
}

/** Options of `antiderivativeData`. */
export type AntiderivativeOptions = {
  /** The number of functions. */
  count: number
  /** The number of sensor points $m$ on $[0, 1]$, ends included. */
  sensors: number
  /** The fine points per sensor interval (default 4), for the antiderivative and for plotting. */
  refine?: number
  /** The RBF kernel's length scale $\ell$ (default 0.2). */
  lengthScale?: number
}

/**
 * Functions $u$ on $[0, 1]$ drawn from a zero-mean Gaussian process with RBF kernel
 * $\exp(-(x - x')^2/(2\ell^2))$ (`samplePrior` of `aifn-methods/learning/gaussian-processes`, from the child stream
 * `draws`) and their antiderivatives $G(u)(y) = \int_0^y u(s)\,ds$, by the cumulative trapezoid rule on a fine grid
 * that contains the sensors.
 *
 * @param s The stream; only its child `draws` is drawn from.
 * @param options The number of functions, the sensors, the refinement and the length scale.
 * @param options.count The number of functions.
 * @param options.sensors The number of sensor points $m$ on $[0, 1]$, ends included.
 * @param options.refine The fine points per sensor interval, for the antiderivative and for plotting.
 * @param options.lengthScale The RBF kernel's length scale $\ell$.
 * @returns The fine grid, the sensors' indices in it, the sensor values (the branch network's input), and $u$ and
 *   $G(u)$ on the fine grid.
 *
 * @example Two functions read at five sensors
 * const d = antiderivativeData(stream(4), { count: 2, sensors: 5 })
 * print('sensor values =', d.sensorValues)
 * print('G(u)(1) =', slice(d.G, [0, 2], -1))
 */
export function antiderivativeData(
  s: Stream,
  { count, sensors, refine = 4, lengthScale = 0.2 }: AntiderivativeOptions,
): {
  /** The fine grid, $F = (m - 1)\,\text{refine} + 1$ points on $[0, 1]$. */
  x: Float64Array
  /** The index of each sensor in the fine grid. */
  sensorIndex: number[]
  /** $u$ at the sensors, $[\text{count}, m]$: the branch network's input. */
  sensorValues: Tensor
  /** $u$ on the fine grid, $[\text{count}, F]$. */
  u: Tensor
  /** $G(u)$ on the fine grid, $[\text{count}, F]$. */
  G: Tensor
} {
  const F = (sensors - 1) * refine + 1
  const x = Float64Array.from({ length: F }, (_, i) => i / (F - 1))
  const { draws } = samplePrior(
    child(s, 'draws'),
    rbf({ lengthscale: lengthScale, variance: 1 }),
    fromData(x, [F, 1]),
    count,
  )
  const u = Float64Array.from(readonlyData(draws) as Float64Array)
  const G = new Float64Array(count * F)
  const dx = 1 / (F - 1)
  for (let i = 0; i < count; i++)
    for (let j = 1; j < F; j++) G[i * F + j] = G[i * F + j - 1] + (dx * (u[i * F + j - 1] + u[i * F + j])) / 2
  const sensorIndex = Array.from({ length: sensors }, (_, k) => k * refine)
  const sv = new Float64Array(count * sensors)
  for (let i = 0; i < count; i++) sensorIndex.forEach((j, k) => (sv[i * sensors + k] = u[i * F + j]))
  return {
    x,
    sensorIndex,
    sensorValues: fromData(sv, [count, sensors]),
    u: fromData(u, [count, F]),
    G: fromData(G, [count, F]),
  }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')
const OPERATORS = ['operator-learning', 'fourier-neural-operator', 'physics-informed-neural-operator']

fn(
  {
    key: 'priorVariance',
    name: 'Periodic prior variance',
    tex: '2\\sigma^2((2\\pi k)^2 + \\tau^2)^{-\\alpha}',
    role: 'property',
    notes: OPERATORS,
    cite: ['li2021fno'],
  },
  priorVariance,
)
fn(
  {
    key: 'periodicGaussianField',
    name: 'Periodic Gaussian random field',
    summary: 'Periodic functions on [0, 1) drawn as Fourier series from the prior N(0, σ²(−Δ + τ²I)^(−α)).',
    role: 'simulation',
    random: true,
    notes: OPERATORS,
    cite: ['li2021fno'],
  },
  periodicGaussianField,
)
fn(
  {
    key: 'heatOperatorData',
    name: 'Heat-equation operator pairs',
    summary: 'Random periodic initial profiles and their exact heat-equation solutions at a time t, on a grid.',
    role: 'simulation',
    random: true,
    notes: ['operator-learning', 'physics-informed-neural-operator'],
    cite: ['li2021fno'],
  },
  heatOperatorData,
)
fn(
  {
    key: 'burgersOperatorData',
    name: 'Burgers operator pairs',
    summary: 'Random periodic initial profiles and their viscous Burgers solutions at a time t, by a spectral solver.',
    role: 'simulation',
    random: true,
    notes: ['fourier-neural-operator', 'operator-learning'],
    cite: ['li2021fno'],
  },
  burgersOperatorData,
)
fn(
  {
    key: 'antiderivativeData',
    name: 'Antiderivative operator pairs',
    summary: 'Gaussian-process functions on [0, 1] read at sensors, with their antiderivatives by the trapezoid rule.',
    role: 'simulation',
    random: true,
    notes: ['deeponet', 'operator-learning'],
    cite: ['lu2021deeponet'],
  },
  antiderivativeData,
)
