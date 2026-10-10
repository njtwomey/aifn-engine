/**
 * `aifn-methods/dynamics/maps`: discrete-time dynamical systems $x_{n+1} = f(x_n)$, their orbits, bifurcation
 * diagrams and Lyapunov exponents.
 *
 * - The standard families: `logisticMap`, `tentMap` and `sineMap` (maps of the line, `Map1`), and `henonMap` and
 *   Chirikov's `standardMap` (maps of the plane, `MapN`).
 * - Iterating: `mapIteration` as a traceable `Algorithm` (one iterate per step), `orbit` for the iterates after a
 *   transient, `cobweb` for the staircase path of a map of the line.
 * - Sweeping a parameter: `bifurcationDiagram` (the attractor against the parameter) and `lyapunovCurve` (the
 *   exponent against the parameter).
 * - Chaos: `lyapunovExponent` of a map of the line, and `lyapunovSpectrum` of a map of $\reals^d$ by the QR method.
 * - Derivatives: `derivativeOf` and `jacobianOf` use the map's own, or `aifn-compute/foundation/autodiff` when it
 *   gives none.
 * - `mapsAlgorithms` and `mapsFunctions`: the registry entries.
 *
 * One-dimensional maps work on numbers; maps of $\reals^d$ on rank-1 tensors. Divergence (a non-finite iterate) ends
 * an orbit early or sets `diverged`, rather than throwing. References: May (1976), "Simple mathematical models with
 * very complicated dynamics", Nature 261; Strogatz (2015), "Nonlinear Dynamics and Chaos", 2nd ed., §10; Benettin et
 * al. (1980), Meccanica 15, for the QR method for Lyapunov spectra; Hénon (1976); Chirikov (1979).
 */

import { grad, jacobian } from 'aifn-compute/foundation/autodiff'
import { qr } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Matrix, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'

type F64 = Float64Array<ArrayBuffer>

/**
 * A map of the real line: an optional `name` (for labels), the map `f`, and optionally its `derivative` $f'$ (from
 * `aifn-compute/foundation/autodiff` when not given, which needs `f` written with aifn's operations).
 */
export type Map1 = { name?: string; f: (x: number) => number; derivative?: (x: number) => number }

/** A map of $\reals^d$ with its Jacobian ($d \times d$, from `aifn-compute/foundation/autodiff` when not given). */
export type MapN = {
  /** A label for the map. */
  name?: string
  /** The map, from a vector of $d$ values to $d$ values. */
  f: (x: Tensor) => Tensor | ArrayLike<number>
  /**
   * Its Jacobian at $\xvec$, as a matrix or as $d$ rows. When it is left out, `f` must be written with aifn's
   * operations and return a tensor, for autodiff.
   */
  jacobian?: (x: Tensor) => Matrix | readonly ArrayLike<number>[]
}

/** Either kind of map. */
export type AnyMap = Map1 | MapN

/**
 * A float64 vector from numbers.
 *
 * @param a The values.
 * @returns The vector (a copy).
 */
const vec = (a: ArrayLike<number>): Vector => fromData(Float64Array.from(a as ArrayLike<number>), [a.length])
/**
 * A tensor's or array's values as a new float64 array.
 *
 * @param v A tensor (flattened row-major) or an array of numbers.
 * @returns The values.
 */
const flat = (v: Tensor | ArrayLike<number>): F64 =>
  Float64Array.from((v as Tensor).shape !== undefined ? toFlat(v as Tensor) : Array.from(v as ArrayLike<number>))

/**
 * The derivative $f'$ of a one-dimensional map: its own `derivative`, or $f$'s gradient by autodiff when not given.
 *
 * @param map The map.
 * @returns The derivative, as a function of $x$.
 *
 * @example The map's own derivative, or autodiff's
 * print("logistic r = 3: f'(0.25) =", derivativeOf(logisticMap(3))(0.25))
 * const cube = { f: (x) => mul(x, mul(x, x)) }
 * print("x^3: f'(2) =", derivativeOf(cube)(2))
 */
export function derivativeOf(map: Map1): (x: number) => number {
  if (map.derivative) return map.derivative
  const g = grad((x: Value) => map.f(x as number) as Value)
  return (x) => g(x) as number
}

/**
 * The Jacobian of a map of $\reals^d$ at $\xvec$: its own `jacobian`, or by autodiff when not given.
 *
 * @param map The map.
 * @param x The point $\xvec$ ($d$ values).
 * @returns The $d \times d$ Jacobian, row-major ($d^2$ values; row $i$ holds the derivatives of the $i$-th output).
 *
 * @example The Hénon map's Jacobian, and one by autodiff
 * print('Hénon at (0.5, 0.2):', jacobianOf(henonMap(), tensor([0.5, 0.2])))
 * const squares = { f: (x) => mul(x, x) }
 * print('x -> x^2 at (1, 3):', jacobianOf(squares, tensor([1, 3])))
 */
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

/**
 * The logistic map $x \mapsto r x (1 - x)$ on $[0, 1]$ ($r \in [0, 4]$), with its derivative $r(1 - 2x)$. A stable
 * fixed point for $r < 3$, period doubling from $r = 3$, and chaos at $r = 4$, with Lyapunov exponent $\ln 2$.
 *
 * @param r The growth parameter $r$.
 * @returns The map.
 *
 * @example At r = 3.2 the orbit settles into a 2-cycle
 * print('first iterates:', orbit(logisticMap(3.2), 0.2, 6))
 * print('after 200:     ', orbit(logisticMap(3.2), 0.2, 6, { discard: 200 }))
 *
 * @example A fixed point below r = 3, chaos at r = 3.9
 * print('r = 2.5:', orbit(logisticMap(2.5), 0.2, 4, { discard: 200 }))
 * print('r = 3.9:', orbit(logisticMap(3.9), 0.2, 4, { discard: 200 }))
 */
export function logisticMap(r: number): Map1 {
  return { name: `logistic r=${r}`, f: (x) => r * x * (1 - x), derivative: (x) => r * (1 - 2 * x) }
}

/**
 * The tent map $x \mapsto \mu \min(x, 1 - x)$ ($\mu \in [0, 2]$); its Lyapunov exponent is $\ln \mu$ wherever the
 * orbit avoids $x = 1/2$, where the derivative $\pm\mu$ jumps.
 *
 * @param mu The slope $\mu$.
 * @returns The map.
 *
 * @example Its Lyapunov exponent is the log of its slope
 * const r = lyapunovExponent(tentMap(1.5), 0.2)
 * print('estimate =', r.exponent, ' ln 1.5 =', Math.log(1.5))
 */
export function tentMap(mu: number): Map1 {
  return { name: `tent μ=${mu}`, f: (x) => mu * Math.min(x, 1 - x), derivative: (x) => (x < 0.5 ? mu : -mu) }
}

/**
 * The sine map $x \mapsto r \sin(\pi x)$ on $[0, 1]$ ($r \in [0, 1]$): unimodal like the logistic map, with the
 * same Feigenbaum $\delta$.
 *
 * @param r The amplitude $r$.
 * @returns The map, with its derivative $r\pi\cos(\pi x)$.
 *
 * @example A fixed point, then a 2-cycle, as r grows
 * print('r = 0.6: ', orbit(sineMap(0.6), 0.2, 4, { discard: 500 }))
 * print('r = 0.8: ', orbit(sineMap(0.8), 0.2, 4, { discard: 500 }))
 */
export function sineMap(r: number): Map1 {
  return {
    name: `sine r=${r}`,
    f: (x) => r * Math.sin(Math.PI * x),
    derivative: (x) => r * Math.PI * Math.cos(Math.PI * x),
  }
}

/**
 * The Hénon map $(x, y) \mapsto (1 - a x^2 + y, b x)$, with its Jacobian; $a = 1.4$, $b = 0.3$ gives the Hénon
 * attractor. Area contracts by $\lvert b \rvert$ each step ($\det \Jmat = -b$).
 *
 * @param a The parameter $a$ (default 1.4).
 * @param b The parameter $b$ (default 0.3).
 * @returns The map of the plane, on the state $(x, y)$.
 *
 * @example The first points of an orbit, and points on the attractor
 * print('from (0, 0):', orbit(henonMap(), [0, 0], 4))
 * print('on the attractor:', orbit(henonMap(), [0, 0], 3, { discard: 1000 }))
 */
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
 * Chirikov's standard map on the torus $[0, 2\pi)^2$: $p \mapsto p + K \sin\theta$, $\theta \mapsto \theta + p$
 * (the new $p$), both modulo $2\pi$, with its Jacobian. Area preserving; invariant circles break up as $K$ grows
 * past about $0.97$.
 *
 * @param K The kick strength $K$.
 * @returns The map of the plane, on the state $(\theta, p)$.
 *
 * @example Area preserving: the Lyapunov exponents sum to 0
 * const { exponents } = lyapunovSpectrum(standardMap(5), [1, 1], { transient: 100, keep: 1000 })
 * print('exponents =', exponents, ' sum =', exponents.data[0] + exponents.data[1])
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
  /** $x_n$: a number for a one-dimensional map, a vector otherwise. */
  x: number | Vector
  /** True when $x_n$ is not finite. */
  diverged: boolean
}

/**
 * One application of a map.
 *
 * @param map The map: of the line when `x` is a number, of $\reals^d$ otherwise.
 * @param x The point.
 * @returns The image: a number, or a new vector.
 */
function applyMap(map: AnyMap, x: number | Vector): number | Vector {
  if (typeof x === 'number') return (map as Map1).f(x)
  return vec(flat((map as MapN).f(x)))
}

/**
 * Whether a point is finite in every coordinate.
 *
 * @param x A number or a vector.
 * @returns True when no entry is NaN or infinite.
 */
const isFinite = (x: number | Vector) => (typeof x === 'number' ? Number.isFinite(x) : toFlat(x).every(Number.isFinite))

/**
 * Iteration of a map as a traceable algorithm: `init` takes `{ x0 }` (a number for a map of the line, an array or
 * vector for a map of $\reals^d$) and each step applies the map once. A non-finite iterate sets `diverged`.
 *
 * @param map The map.
 * @returns The algorithm, named after the map.
 *
 * @example Run the logistic map, and trace it
 * const it = mapIteration(logisticMap(3.2))
 * print('x after 100 steps:', run(it, { x0: 0.2 }, 100).x)
 * print('first steps:', trace(it, { x0: 0.2 }, 5, { record: { x: (s) => s.x } }).series.x)
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
 * The orbit $x_k, x_{k+1}, \dots, x_{k+n-1}$ after discarding the first $k$ = `discard` iterates: a vector of length
 * $n$ for a map of the line, an $n \times d$ matrix otherwise. Iteration stops early (and the orbit is shorter) if it
 * diverges.
 *
 * @param map The map.
 * @param x0 The starting point $x_0$: a number for a map of the line, $d$ values otherwise.
 * @param n The number of iterates kept, $n$.
 * @param options The transient.
 * @param options.discard The number of iterates $k$ dropped before the first kept one (default 0, which keeps $x_0$).
 * @returns The kept iterates: a vector, or one row per iterate.
 *
 * @example An orbit of the line and one of the plane
 * print('logistic r = 2.8:', orbit(logisticMap(2.8), 0.1, 5))
 * print('Hénon:', orbit(henonMap(), [0, 0], 3))
 *
 * @example A diverging orbit is cut short
 * print(orbit({ f: (x) => x * x }, 10, 20))
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
 * The cobweb (staircase) path of a map of the line from $x_0$ over $n$ iterations: from $(x_0, 0)$ up to the graph at
 * $(x_0, f(x_0))$, across to the diagonal at $(f(x_0), f(x_0))$, up or down to the graph again, and so on. The path
 * stops early at a non-finite value.
 *
 * @param map The map.
 * @param x0 The starting point $x_0$.
 * @param n The number of iterations.
 * @returns The path's vertices as `x` and `y` (length $2n + 1$).
 *
 * @example Two steps towards the fixed point of the logistic map at r = 2.5
 * const { x, y } = cobweb(logisticMap(2.5), 0.1, 2)
 * print('x =', x)
 * print('y =', y)
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
  /** The starting point for every parameter value. Default 0.5. */
  x0?: number
  /** Iterates discarded so the orbit settles on its attractor. Default 500. */
  transient?: number
  /**
   * Iterates kept (plotted, or averaged). Default 100 (bifurcation), 1000 (Lyapunov exponent) or 5000 (Lyapunov
   * spectrum).
   */
  keep?: number
}

/**
 * The bifurcation diagram of a one-parameter family of maps of the line: for each parameter $r$, the orbit from $x_0$
 * after `transient` iterates, `keep` points long.
 *
 * @param family The family: the map for each parameter value.
 * @param rs The parameter values.
 * @param options The start, transient and number of points kept.
 * @param options.x0 The starting point $x_0$ for every parameter value (default 0.5).
 * @param options.transient Iterates discarded first, so the orbit settles on its attractor (default 500).
 * @param options.keep Points kept per parameter value (default 100).
 * @returns Flat `r` and `x` (length `rs.length` times `keep`, fewer if an orbit diverges), ready for a scatter plot of
 *   the attractor against the parameter.
 *
 * @example One, two and four points on the attractor
 * const { r, x } = bifurcationDiagram(logisticMap, [2.8, 3.2, 3.5], { keep: 4, x0: 0.2 })
 * print('r =', r)
 * print('x =', x)
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
  /** $\lambda \approx (1/n) \sum_k \ln\lvert f'(x_k) \rvert$ over the $n$ kept iterates. */
  exponent: number
  /** The running estimate after each kept iterate (length $n$). */
  running: Vector
  /** True when the orbit hit a point with $f' = 0$ (the exponent is then $-\infty$) or diverged. */
  degenerate: boolean
}

/**
 * The Lyapunov exponent $\lambda = \lim_{n \to \infty} (1/n) \sum_k \ln\lvert f'(x_k) \rvert$ of a map of the
 * line along the orbit from $x_0$, after `transient` iterates: the mean exponential rate at which nearby orbits
 * separate ($\lambda > 0$ is chaos). For the logistic map at $r = 4$ it is $\ln 2$ (start off $x_0 = 0.5$, which
 * maps to 1 and then to the fixed point 0). Rounding makes long orbits of the tent map at $\mu = 2$ collapse to 0, so
 * use $\mu$ slightly below 2 there.
 *
 * @param map The map, with its derivative (by autodiff when not given).
 * @param x0 The starting point $x_0$.
 * @param options The transient and the number of iterates averaged.
 * @param options.transient Iterates discarded first (default 500).
 * @param options.keep Iterates averaged, $n$ (default 1000).
 * @returns The estimate, its running average and whether it is degenerate.
 *
 * @example Chaos at r = 4, a stable 2-cycle at r = 3.2
 * print('r = 4:  ', lyapunovExponent(logisticMap(4), 0.3).exponent, ' ln 2 =', Math.log(2))
 * print('r = 3.2:', lyapunovExponent(logisticMap(3.2), 0.3).exponent)
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
 *
 * @param family The family: the map for each parameter value.
 * @param rs The parameter values.
 * @param options The start, transient and number of iterates averaged.
 * @param options.x0 The starting point $x_0$ for every parameter value (default 0.5).
 * @param options.transient Iterates discarded first (default 500).
 * @param options.keep Iterates averaged (default 1000).
 * @returns The exponent at each parameter value.
 *
 * @example Negative in the periodic windows, positive in chaos
 * const rs = [2.8, 3.2, 3.5, 3.83, 3.9]
 * print('r =     ', rs)
 * print('lambda =', lyapunovCurve(logisticMap, rs, { x0: 0.3 }))
 */
export function lyapunovCurve(
  family: (r: number) => Map1,
  rs: ArrayLike<number> | Tensor,
  { x0 = 0.5, transient = 500, keep = 1000 }: SweepOptions = {},
): Vector {
  return vec(Array.from(flat(rs), (r) => lyapunovExponent(family(r), x0, { transient, keep }).exponent))
}

/**
 * The Lyapunov spectrum $\lambda_1 \ge \dots \ge \lambda_d$ of a map of $\reals^d$ along the orbit from $\xvec_0$
 * by the QR method of Benettin et al. (1980): propagate an orthonormal frame by the Jacobian, re-orthonormalise with
 * QR each step, and average $\ln\lvert R_{ii} \rvert$. Their sum is the mean of $\ln\lvert \det \Jmat \rvert$
 * ($\ln\lvert b \rvert$ for the Hénon map; 0 for the area-preserving standard map).
 *
 * @param map The map, with its Jacobian (by autodiff when not given).
 * @param x0 The starting point $\xvec_0$ ($d$ values).
 * @param options The transient and the number of steps averaged.
 * @param options.transient Iterates discarded first (default 500).
 * @param options.keep Steps averaged (default 5000).
 * @returns `exponents`, sorted in descending order, and `running`, the running averages after each step
 *   ($\text{keep} \times d$, columns in the QR order, unsorted).
 *
 * @example The Hénon attractor: one positive exponent, and a sum of ln 0.3
 * const { exponents } = lyapunovSpectrum(henonMap(), [0, 0], { keep: 2000 })
 * print('exponents =', exponents)
 * print('sum =', exponents.data[0] + exponents.data[1], ' ln 0.3 =', Math.log(0.3))
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
