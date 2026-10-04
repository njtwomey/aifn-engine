/**
 * `aifn-methods/dynamics/maps`: discrete-time dynamical systems x_{n+1} = f(x_n): iteration as a traceable `Algorithm`,
 * orbits, cobweb paths, bifurcation diagrams and Lyapunov exponents, with the standard families (logistic, tent, sine,
 * Hénon, Chirikov's standard map). References: May (1976), "Simple mathematical models with very complicated dynamics",
 * Nature 261; Strogatz (2015), "Nonlinear Dynamics and Chaos", 2nd ed., §10; Benettin et al. (1980), Meccanica 15, for
 * the QR method for Lyapunov spectra; Hénon (1976); Chirikov (1979).
 *
 * One-dimensional maps work on numbers; maps of ℝᵈ on rank-1 tensors.
 */

import { grad, jacobian } from 'aifn-compute/foundation/autodiff'
import { qr } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Matrix, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'

type F64 = Float64Array<ArrayBuffer>

/** A map of the real line with its derivative (from `aifn-compute/foundation/autodiff` when not given). */
export type Map1 = { name?: string; f: (x: number) => number; derivative?: (x: number) => number }

/** A map of ℝᵈ with its Jacobian (d × d, from `aifn-compute/foundation/autodiff` when not given). */
export type MapN = {
  name?: string
  f: (x: Tensor) => Tensor | ArrayLike<number>
  jacobian?: (x: Tensor) => Matrix | readonly ArrayLike<number>[]
}

/** Either kind of map. */
export type AnyMap = Map1 | MapN

const vec = (a: ArrayLike<number>): Vector => fromData(Float64Array.from(a as ArrayLike<number>), [a.length])
const flat = (v: Tensor | ArrayLike<number>): F64 =>
  Float64Array.from((v as Tensor).shape !== undefined ? toFlat(v as Tensor) : Array.from(v as ArrayLike<number>))

/** The derivative of a one-dimensional map, by autodiff when not given. */
export function derivativeOf(map: Map1): (x: number) => number {
  if (map.derivative) return map.derivative
  const g = grad((x: Value) => map.f(x as number) as Value)
  return (x) => g(x) as number
}

/** The Jacobian of a map of ℝᵈ at x (d × d), by autodiff when not given. */
export function jacobianOf(map: MapN, x: Tensor): F64 {
  if (map.jacobian) {
    const J = map.jacobian(x)
    return (J as Tensor).shape !== undefined
      ? Float64Array.from(toFlat(J as Tensor))
      : Float64Array.from((J as readonly ArrayLike<number>[]).flatMap((r) => Array.from(r)))
  }
  return Float64Array.from(toFlat(jacobian((y: Value) => map.f(y as Tensor) as Value)(x) as Tensor))
}

// ---------------------------------------------------------------------------------------------------------------------
// Families

/** The logistic map x ↦ r x (1 − x) on [0, 1] (r ∈ [0, 4]); chaos at r = 4, with Lyapunov exponent ln 2. */
export function logisticMap(r: number): Map1 {
  return { name: `logistic r=${r}`, f: (x) => r * x * (1 - x), derivative: (x) => r * (1 - 2 * x) }
}

/** The tent map x ↦ μ min(x, 1 − x) (μ ∈ [0, 2]); its Lyapunov exponent is ln μ wherever the orbit avoids x = ½. */
export function tentMap(mu: number): Map1 {
  return { name: `tent μ=${mu}`, f: (x) => mu * Math.min(x, 1 - x), derivative: (x) => (x < 0.5 ? mu : -mu) }
}

/** The sine map x ↦ r sin(πx) on [0, 1] (r ∈ [0, 1]): unimodal like the logistic map, with the same Feigenbaum δ. */
export function sineMap(r: number): Map1 {
  return {
    name: `sine r=${r}`,
    f: (x) => r * Math.sin(Math.PI * x),
    derivative: (x) => r * Math.PI * Math.cos(Math.PI * x),
  }
}

/** The Hénon map (x, y) ↦ (1 − a x² + y, b x); a = 1.4, b = 0.3 gives the Hénon attractor. Area contracts by |b|. */
export function henonMap(a = 1.4, b = 0.3): MapN {
  return {
    name: `hénon a=${a} b=${b}`,
    f: (p) => {
      const [x, y] = toFlat(p)
      return Float64Array.of(1 - a * x * x + y, b * x)
    },
    jacobian: (p) => {
      const [x] = toFlat(p)
      return [
        [-2 * a * x, 1],
        [b, 0],
      ]
    },
  }
}

/**
 * Chirikov's standard map on the torus [0, 2π)²: p ↦ p + K sin θ, θ ↦ θ + p (the new p), both mod 2π. Area
 * preserving; invariant circles break up as K grows past about 0.97.
 */
export function standardMap(K: number): MapN {
  const TAU = 2 * Math.PI
  const mod = (v: number) => ((v % TAU) + TAU) % TAU
  return {
    name: `standard K=${K}`,
    f: (s) => {
      const [theta, p] = toFlat(s)
      const p1 = mod(p + K * Math.sin(theta))
      return Float64Array.of(mod(theta + p1), p1)
    },
    jacobian: (s) => {
      const [theta] = toFlat(s)
      const c = K * Math.cos(theta)
      return [
        [1 + c, 1],
        [c, 1],
      ]
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Iteration

/** The state of `mapIteration`. */
export type MapState = {
  /** The iteration count n. */
  t: number
  /** x_n: a number for a one-dimensional map, a vector otherwise. */
  x: number | Vector
  diverged: boolean
}

function applyMap(map: AnyMap, x: number | Vector): number | Vector {
  if (typeof x === 'number') return (map as Map1).f(x)
  return vec(flat((map as MapN).f(x)))
}

const isFinite = (x: number | Vector) => (typeof x === 'number' ? Number.isFinite(x) : toFlat(x).every(Number.isFinite))

/**
 * Iteration of a map as a traceable algorithm: `init` takes `{ x0 }` (a number for a map of the line, an array or
 * vector for a map of ℝᵈ) and each step applies the map once. A non-finite iterate sets `diverged`.
 */
export function mapIteration(map: AnyMap): Algorithm<{ x0: number | ArrayLike<number> | Tensor }, MapState> {
  return {
    name: map.name ?? 'map',
    init: ({ x0 }) => {
      const x: number | Vector = typeof x0 === 'number' ? x0 : vec(flat(x0))
      return { t: 0, x, diverged: !isFinite(x) }
    },
    step: (s) => {
      const x = applyMap(map, s.x)
      return { t: s.t + 1, x, diverged: !isFinite(x) }
    },
  }
}

/**
 * The orbit x_k, x_{k+1}, …, x_{k+n−1} after discarding the first k = `discard` iterates: a vector of length n for a
 * map of the line, an n × d matrix otherwise. Iteration stops early (and the orbit is shorter) if it diverges.
 */
export function orbit(
  map: AnyMap,
  x0: number | ArrayLike<number> | Tensor,
  n: number,
  { discard = 0 }: { discard?: number } = {},
): Vector | Matrix {
  let x: number | Vector = typeof x0 === 'number' ? x0 : vec(flat(x0))
  for (let k = 0; k < discard && isFinite(x); k++) x = applyMap(map, x)
  const out: number[] = []
  let d = 1
  for (let k = 0; k < n && isFinite(x); k++) {
    if (typeof x === 'number') out.push(x)
    else {
      const v = toFlat(x)
      d = v.length
      out.push(...v)
    }
    x = applyMap(map, x)
  }
  if (typeof x0 === 'number') return fromData(Float64Array.from(out), [out.length])
  return fromData(Float64Array.from(out), [out.length / d, d])
}

/**
 * The cobweb (staircase) path of a map of the line from x₀ over n iterations: from (x₀, 0) up to the graph at
 * (x₀, f(x₀)), across to the diagonal at (f(x₀), f(x₀)), up or down to the graph again, and so on. Returns the path's
 * vertices as `x` and `y` (length 2n + 1).
 */
export function cobweb(map: Map1, x0: number, n: number): { x: Vector; y: Vector } {
  const xs = [x0]
  const ys = [0]
  let x = x0
  for (let k = 0; k < n; k++) {
    const y = map.f(x)
    if (!Number.isFinite(y)) break
    xs.push(x, y)
    ys.push(y, y)
    x = y
  }
  return { x: vec(xs), y: vec(ys) }
}

/** Options for `bifurcationDiagram` and `lyapunovCurve`. */
export type SweepOptions = {
  /** The starting point for every parameter value. Default 0.5 (or the map's choice). */
  x0?: number
  /** Iterates discarded so the orbit settles on its attractor. Default 500. */
  transient?: number
  /** Iterates kept (plotted, or averaged). Default 100 (bifurcation) or 1000 (Lyapunov). */
  keep?: number
}

/**
 * The bifurcation diagram of a one-parameter family of maps of the line: for each parameter r, the orbit from x₀ after
 * `transient` iterates, `keep` points long. Returns flat `r` and `x` (length rs × keep, fewer if an orbit diverges),
 * ready for a scatter plot of the attractor against the parameter.
 */
export function bifurcationDiagram(
  family: (r: number) => Map1,
  rs: ArrayLike<number> | Tensor,
  { x0 = 0.5, transient = 500, keep = 100 }: SweepOptions = {},
): { r: Vector; x: Vector } {
  const params = flat(rs)
  const outR: number[] = []
  const outX: number[] = []
  for (const r of params) {
    const { f } = family(r)
    let x = x0
    for (let k = 0; k < transient && Number.isFinite(x); k++) x = f(x)
    for (let k = 0; k < keep && Number.isFinite(x); k++) {
      outR.push(r)
      outX.push(x)
      x = f(x)
    }
  }
  return { r: vec(outR), x: vec(outX) }
}

/** A Lyapunov exponent estimate with its running average, which shows convergence. */
export type LyapunovEstimate = {
  /** λ ≈ (1/n) Σ ln|f′(x_k)| over the kept iterates. */
  exponent: number
  /** The running estimate after each kept iterate (length n). */
  running: Vector
  /** True when the orbit hit a point with f′ = 0 (the exponent is then −∞) or diverged. */
  degenerate: boolean
}

/**
 * The Lyapunov exponent λ = lim (1/n) Σ ln|f′(x_k)| of a map of the line along the orbit from x₀, after `transient`
 * iterates: the mean exponential rate at which nearby orbits separate (λ > 0 is chaos). For the logistic map at r = 4
 * it is ln 2. Rounding makes long orbits of the tent map at μ = 2 collapse to 0, so use μ slightly below 2 there.
 */
export function lyapunovExponent(
  map: Map1,
  x0: number,
  { transient = 500, keep = 1000 }: Omit<SweepOptions, 'x0'> = {},
): LyapunovEstimate {
  const df = derivativeOf(map)
  let x = x0
  for (let k = 0; k < transient; k++) x = map.f(x)
  const running = new Float64Array(keep)
  let sum = 0
  let degenerate = false
  for (let k = 0; k < keep; k++) {
    const d = Math.abs(df(x))
    if (!(d > 0) || !Number.isFinite(x)) degenerate = true
    sum += Math.log(d)
    running[k] = sum / (k + 1)
    x = map.f(x)
  }
  return { exponent: sum / keep, running: fromData(running, [keep]), degenerate }
}

/**
 * The Lyapunov exponent at each parameter of a family (see `lyapunovExponent`), e.g. to plot under a bifurcation
 * diagram.
 */
export function lyapunovCurve(
  family: (r: number) => Map1,
  rs: ArrayLike<number> | Tensor,
  { x0 = 0.5, transient = 500, keep = 1000 }: SweepOptions = {},
): Vector {
  return vec(Array.from(flat(rs), (r) => lyapunovExponent(family(r), x0, { transient, keep }).exponent))
}

/**
 * The Lyapunov spectrum λ₁ ≥ … ≥ λ_d of a map of ℝᵈ along the orbit from x₀ by the QR method of Benettin et al.
 * (1980): propagate an orthonormal frame by the Jacobian, re-orthonormalise with QR each step, and average ln|R_ii|.
 * Their sum is the mean of ln|det J| (ln|b| for the Hénon map; 0 for the area-preserving standard map).
 */
export function lyapunovSpectrum(
  map: MapN,
  x0: ArrayLike<number> | Tensor,
  { transient = 500, keep = 5000 }: Omit<SweepOptions, 'x0'> = {},
): { exponents: Vector; running: Matrix } {
  let x = vec(flat(x0))
  const d = x.shape[0]
  for (let k = 0; k < transient; k++) x = vec(flat(map.f(x)))
  let Q = new Float64Array(d * d)
  for (let i = 0; i < d; i++) Q[i * d + i] = 1
  const sums = new Float64Array(d)
  const running = new Float64Array(keep * d)
  for (let k = 0; k < keep; k++) {
    const J = jacobianOf(map, x)
    const JQ = new Float64Array(d * d)
    for (let i = 0; i < d; i++)
      for (let l = 0; l < d; l++) for (let j = 0; j < d; j++) JQ[i * d + j] += J[i * d + l] * Q[l * d + j]
    const f = qr(fromData(JQ, [d, d]))
    const R = toFlat(f.R)
    const Qn = Float64Array.from(toFlat(f.Q))
    for (let i = 0; i < d; i++) {
      sums[i] += Math.log(Math.abs(R[i * d + i]))
      running[k * d + i] = sums[i] / (k + 1)
    }
    Q = Qn
    x = vec(flat(map.f(x)))
  }
  const exponents = Array.from(sums, (s) => s / keep).sort((a, b) => b - a)
  return { exponents: vec(exponents), running: fromData(running, [keep, d]) }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/maps')
const fn = definer<FunctionInfo>('function', 'dynamics/maps')

algorithm(
  {
    key: 'mapIteration',
    name: 'Map iteration',
    summary: 'x_{n+1} = f(x_n) for a map of the line or of ℝᵈ, one iterate per step.',
    problem: 'map',
    state: { iterate: 'x', flags: ['diverged'] },
    cite: ['strogatz2015'],
  },
  mapIteration,
)
fn(
  {
    key: 'logisticMap',
    name: 'Logistic map',
    tex: 'x \\mapsto r x (1 - x)',
    role: 'construction',
    cite: ['strogatz2015'],
  },
  logisticMap,
)
fn({ key: 'tentMap', name: 'Tent map', role: 'construction' }, tentMap)
fn({ key: 'sineMap', name: 'Sine map', role: 'construction' }, sineMap)
fn({ key: 'henonMap', name: 'Hénon map', role: 'construction' }, henonMap)
fn({ key: 'standardMap', name: 'Chirikov standard map', role: 'construction' }, standardMap)
fn({ key: 'orbit', name: 'Orbit', role: 'simulation' }, orbit)
fn({ key: 'cobweb', name: 'Cobweb diagram', role: 'construction' }, cobweb)
fn(
  { key: 'bifurcationDiagram', name: 'Bifurcation diagram', role: 'simulation', cite: ['strogatz2015'] },
  bifurcationDiagram,
)
fn(
  { key: 'lyapunovExponent', name: 'Lyapunov exponent of a map', role: 'estimator', cite: ['strogatz2015'] },
  lyapunovExponent,
)
fn({ key: 'lyapunovCurve', name: 'Lyapunov exponent across a family', role: 'estimator' }, lyapunovCurve)
fn({ key: 'lyapunovSpectrum', name: 'Lyapunov spectrum (QR method)', role: 'estimator' }, lyapunovSpectrum)

/** The algorithms of the module, keyed by factory name. */
export const mapsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', { mapIteration }) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const mapsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', {
    logisticMap,
    tentMap,
    sineMap,
    henonMap,
    standardMap,
    orbit,
    cobweb,
    bifurcationDiagram,
    lyapunovExponent,
    lyapunovCurve,
    lyapunovSpectrum,
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
