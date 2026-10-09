/**
 * Flows of autonomous vector fields $\xvec' = \fvec(\xvec)$: flow maps, trajectories and streamlines by classical RK4,
 * and the transport of densities and samples along the flow by the method of characteristics (Arnold, 1992, "Ordinary
 * Differential Equations", §1–2; Villani, 2003, "Topics in Optimal Transportation", §8.1, for the continuity
 * equation).
 *
 * Every integration here takes a fixed number of RK4 steps (`rungeKutta` of `aifn-compute/dynamics/ode`), so the cost
 * is known in advance and the result is a smooth function of the start; for error control use the ode solvers on
 * `autonomous(f)`. The transport functions integrate the divergence $\nabla \cdot \fvec$ alongside each path, which
 * gives the change of density (or of log-volume) along it.
 */

import { rungeKutta } from 'aifn-compute/dynamics/ode'
import { type Rhs } from 'aifn-compute/dynamics/ode'
import { stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { divergence, type VectorField } from './calculus'

const { allFinite, toF64, toMatrixF64 } = dense
type F64 = dense.F64

/**
 * The right-hand side $(t, \xvec) \mapsto \fvec(\xvec)$ of the autonomous system $\xvec' = \fvec(\xvec)$, for the ode
 * solvers.
 *
 * @param f The vector field.
 * @returns The ode right-hand side, which ignores the time.
 *
 * @example The same slope at every time
 * const rhs = autonomous((x) => stack([neg(get(x, 1)), get(x, 0)]))
 * print('at t = 0:', rhs(0, tensor([1, 2])))
 * print('at t = 5:', rhs(5, tensor([1, 2])))
 */
export const autonomous =
  (f: VectorField): Rhs =>
  (_t, x) =>
    f(x)

/** Options for integrating a flow with fixed RK4 steps. */
export type FlowOptions = {
  /** The number of RK4 steps over the time span. Default 100. */
  steps?: Size
}

/**
 * The flow map $\varphi_t(\xvec_0)$: where the trajectory from $\xvec_0$ is after time $t$ (negative $t$ runs
 * backwards), by RK4 in `steps` equal steps.
 *
 * @param f The vector field.
 * @param x0 The starting point $\xvec_0$ ($n$ values).
 * @param t The time to flow for; negative flows backwards, and 0 returns $\xvec_0$ unchanged.
 * @param options The number of RK4 steps.
 * @param options.steps The number of equal RK4 steps over $[0, t]$ (default 100).
 * @returns The point $\varphi_t(\xvec_0)$, a vector of $n$.
 *
 * @example A quarter turn, and back again
 * // The rotation x′ = −y, y′ = x carries (1, 0) to (0, 1) in time π/2.
 * const f = (x) => stack([neg(get(x, 1)), get(x, 0)])
 * const there = flowMap(f, [1, 0], Math.PI / 2)
 * print('φ(π/2) =', there)
 * print('and back:', flowMap(f, there, -Math.PI / 2))
 */
export function flowMap(f: VectorField, x0: VectorLike, t: Scalar, { steps = 100 }: FlowOptions = {}): Vector {
  if (t === 0) return dense.vec(toF64(x0, 'flowMap'))
  return run(rungeKutta(autonomous(f), 'rk4', { stepSize: t / steps, tEnd: t }), { x0 }, steps + 1).x
}

/**
 * The trajectory from $\xvec_0$ over $[0, t]$ by RK4 in `steps` equal steps. The run continues through non-finite
 * states and reports them in `diverged`.
 *
 * @param f The vector field.
 * @param x0 The starting point $\xvec_0$ ($n$ values).
 * @param t The final time (negative runs backwards).
 * @param options The number of RK4 steps.
 * @param options.steps The number of equal RK4 steps over $[0, t]$ (default 100).
 * @returns `time`, the $\text{steps} + 1$ times; `x`, the states, one per row ($(\text{steps} + 1) \times n$); and
 *   `diverged`, whether the run met a non-finite state.
 *
 * @example Exponential decay in four steps
 * // x′ = −x from 1: compare with e^{−t}.
 * const { time, x } = trajectory((x) => neg(x), [1], 1, { steps: 4 })
 * print('t =', time)
 * print('x =', x)
 * print('e^{-t} =', toFlat(time).map((t) => Math.exp(-t)))
 */
export function trajectory(
  f: VectorField,
  x0: VectorLike,
  t: Scalar,
  { steps = 100 }: FlowOptions = {},
): { time: Vector; x: Matrix; diverged: boolean } {
  const tr = trace(rungeKutta(autonomous(f), 'rk4', { stepSize: t / steps, tEnd: t }), { x0 }, steps + 1, {
    stopOnNonFinite: false,
  })
  const n = tr.steps[0].x.shape[0]
  const X = new Float64Array(tr.steps.length * n)
  tr.steps.forEach((s, k) => X.set(toFlat(s.x), k * n))
  return {
    time: fromData(
      Float64Array.from(tr.steps, (s) => s.time),
      [tr.steps.length],
    ),
    x: fromData(X, [tr.steps.length, n]),
    diverged: tr.meta.stopped === 'diverged',
  }
}

/** A box: one interval $[\mathit{lo}_i, \mathit{hi}_i]$ per coordinate, as `[lo, hi]` pairs. */
export type Box = readonly (readonly [Scalar, Scalar])[]

/** Options for `streamline`. */
export type StreamlineOptions = {
  /** The time to follow the flow in each direction. Default 10. */
  t?: Scalar
  /** RK4 steps per direction. Default 200. */
  steps?: Size
  /** `'forward'`, `'backward'` or `'both'` (default): both joins the backward part, reversed, to the forward part. */
  direction?: 'forward' | 'backward' | 'both'
  /** Stop when the curve leaves this box (the last point is the first one outside). */
  bounds?: Box
  /**
   * Stop when the speed $\lVert \fvec \rVert$ is at or below this (the curve has reached a fixed point). Default
   * $10^{-9}$.
   */
  minSpeed?: Scalar
}

/**
 * Follows the flow from `x0` with RK4 steps of `h`, collecting the points.
 *
 * @param f The vector field.
 * @param x0 The starting point; it is the first point returned.
 * @param h The step size: positive follows the flow forwards, negative backwards.
 * @param steps The most steps to take.
 * @param bounds The box to stay in: the curve stops at its first point outside (which is kept). None when undefined.
 * @param minSpeed Stop before a step when $\lVert \fvec \rVert$ is at or below this (or not a number).
 * @returns The points in the order visited, starting with `x0`; a non-finite point stops the curve and is dropped.
 */
function follow(f: VectorField, x0: F64, h: Scalar, steps: Size, bounds: Box | undefined, minSpeed: Scalar): F64[] {
  const pts: F64[] = [x0]
  const rk = rungeKutta(autonomous(f), 'rk4', { stepSize: h })
  // RK4 draws nothing, so one stream serves every step.
  const root = stream(0)
  let s = rk.init({ x0 }, root)
  const inside = (x: F64) => !bounds || bounds.every(([lo, hi], i) => x[i] >= lo && x[i] <= hi)
  for (let k = 0; k < steps; k++) {
    const speed = Math.hypot(...toF64(f(s.x), 'streamline'))
    if (!(speed > minSpeed)) break
    s = rk.step(s, { t: s.t, stream: root })
    const x = dense.data(s.x)
    if (!allFinite(x)) break
    pts.push(x)
    if (!inside(x)) break
  }
  return pts
}

/**
 * A streamline (trajectory curve) of the flow through `seed`, followed by RK4 forwards, backwards or both, stopping on
 * leaving `bounds`, on reaching a fixed point (speed at or below `minSpeed`) or on a non-finite state.
 *
 * @param f The vector field.
 * @param seed The point the streamline passes through ($n$ values).
 * @param options How long and which way to follow it, and when to stop.
 * @returns The points in time order, one per row ($m \times n$); `seed` is the last row for `'backward'`, the first
 *   for `'forward'`, and in between for `'both'`.
 *
 * @example A circle of the rotation field
 * // From (1, 0) for time π in 4 steps forwards: points on the unit circle.
 * const f = (x) => stack([neg(get(x, 1)), get(x, 0)])
 * print(streamline(f, [1, 0], { t: Math.PI, steps: 4, direction: 'forward' }))
 *
 * @example It stops on leaving the box
 * // x′ = x from (0.5, 0) runs out of [−1, 1]²; backwards it slows towards the fixed point at the origin.
 * const curve = streamline((x) => x, [0.5, 0], { t: 5, steps: 50, bounds: [[-1, 1], [-1, 1]] })
 * print('points:', curve.shape[0], ' first:', toRows(curve)[0], ' last:', toRows(curve).at(-1))
 */
export function streamline(f: VectorField, seed: VectorLike, options: StreamlineOptions = {}): Matrix {
  const { t = 10, steps = 200, direction = 'both', bounds, minSpeed = 1e-9 } = options
  const x0 = toF64(seed, 'streamline')
  const h = t / steps
  const forward = direction === 'backward' ? [x0] : follow(f, x0, h, steps, bounds, minSpeed)
  const backward = direction === 'forward' ? [] : follow(f, x0, -h, steps, bounds, minSpeed).slice(1).reverse()
  const pts = [...backward, ...forward]
  const n = x0.length
  const out = new Float64Array(pts.length * n)
  pts.forEach((p, k) => out.set(p, k * n))
  return fromData(out, [pts.length, n])
}

/**
 * Streamlines through every row of `seeds`, each as `streamline` gives it.
 *
 * @param f The vector field.
 * @param seeds The points the streamlines pass through, one per row ($k \times n$).
 * @param options How long and which way to follow each, and when to stop, as for `streamline`.
 * @returns The $k$ streamlines, in the order of the seeds.
 *
 * @example Three circles of the rotation field
 * const f = (x) => stack([neg(get(x, 1)), get(x, 0)])
 * const lines = streamlines(f, [[0.5, 0], [1, 0], [2, 0]], { t: Math.PI / 2, steps: 10, direction: 'forward' })
 * for (const c of lines) print('from', toRows(c)[0], 'to', toRows(c).at(-1))
 */
export function streamlines(f: VectorField, seeds: MatrixLike, options: StreamlineOptions = {}): Matrix[] {
  const { data, m, n } = toMatrixF64(seeds, 'streamlines')
  return Array.from({ length: m }, (_, i) => streamline(f, data.subarray(i * n, (i + 1) * n), options))
}

/** Options for `transportDensity`. */
export type TransportOptions = {
  /** RK4 steps along each characteristic. Default 50. */
  steps?: Size
  /** $\nabla \cdot \fvec$ in closed form, if known (otherwise by autodiff, which is much slower). */
  divergence?: (x: Tensor) => Scalar
}

/**
 * The density $\rho(\xvec, t)$ at the given points of an initial density $\rho_0$ transported by the flow
 * $\xvec' = \fvec(\xvec)$: the continuity (Liouville) equation
 * $\partial\rho/\partial t + \nabla \cdot (\rho\fvec) = 0$. Along a trajectory
 * $d\rho/dt = -\rho \, \nabla \cdot \fvec$, so
 * $\rho(\xvec, t) = \rho_0(\varphi_{-t}(\xvec)) \exp(-\int_0^t \nabla \cdot \fvec(\varphi_{-s}(\xvec)) \, ds)$: each
 * point is followed backwards to its origin while the divergence is accumulated (the method of characteristics).
 *
 * @param f The vector field; differentiable unless `divergence` is given.
 * @param rho0 The initial density $\rho_0$, called with a point as a rank-1 tensor and returning a number.
 * @param points The points at which to evaluate $\rho$, one per row ($k \times n$).
 * @param t The time at which to evaluate it.
 * @param options The RK4 steps along each characteristic and the divergence in closed form.
 * @param options.steps RK4 steps along each characteristic over $[0, t]$ (default 50).
 * @param options.divergence $\nabla \cdot \fvec$ in closed form; without it, by autodiff (much slower).
 * @returns $\rho(\xvec, t)$ at every point, a vector of $k$.
 *
 * @example A contracting flow concentrates a Gaussian
 * // x′ = −x has divergence −1, so ρ(x, t) = ρ₀(x eᵗ) eᵗ.
 * const rho0 = (x) => Math.exp(-0.5 * toFlat(x)[0] ** 2) / Math.sqrt(2 * Math.PI)
 * const rho = transportDensity((x) => neg(x), rho0, [[0], [0.5]], 1, { divergence: () => -1 })
 * print('ρ(x, 1) =', rho)
 * print('exact =', [0, 0.5].map((x) => rho0(tensor([x * Math.E])) * Math.E))
 */
export function transportDensity(
  f: VectorField,
  rho0: (x: Tensor) => Scalar,
  points: MatrixLike,
  t: Scalar,
  { steps = 50, divergence: div }: TransportOptions = {},
): Vector {
  const m = toMatrixF64(points, 'transportDensity').m
  const frames = transportDensityFrames(f, rho0, points, t, { frames: 1, steps, divergence: div })
  return fromData(Float64Array.from(dense.data(frames).subarray(m)), [m])
}

/**
 * The flow with its divergence integrated alongside, $\zvec = (\xvec, L)$: $\xvec' = \sigma\fvec(\xvec)$,
 * $L' = \nabla \cdot \fvec(\xvec)$, with $\sigma = \pm 1$ the direction. $L$ accumulates the divergence along the path
 * whichever way it runs.
 *
 * @param f The vector field.
 * @param n The dimension $n$ of the field; the state $\zvec$ has $n + 1$ entries.
 * @param sign The direction $\sigma$: 1 follows the flow forwards, $-1$ backwards.
 * @param divAt The divergence $\nabla \cdot \fvec$ at a point.
 * @param where The caller's name, used in error messages.
 * @returns The ode right-hand side on $\zvec$.
 */
function withDivergence(f: VectorField, n: Size, sign: 1 | -1, divAt: (x: Tensor) => Scalar, where: string): Rhs {
  return (_t, z) => {
    const d = toFlat(z)
    const x = fromData(Float64Array.from(d.slice(0, n)), [n])
    const out = new Float64Array(n + 1)
    const fx = toF64(f(x), where)
    for (let i = 0; i < n; i++) out[i] = sign * fx[i]
    out[n] = divAt(x)
    return fromData(out, [n + 1])
  }
}

/**
 * The states of `rhs` from $\zvec_0$ at `frames` + 1 equally spaced times over $[0, t]$, `steps` RK4 steps per frame.
 *
 * @param rhs The ode right-hand side.
 * @param z0 The starting state $\zvec_0$; returned (not copied) for every frame when $t = 0$.
 * @param t The final time.
 * @param frames The number of equal intervals of $[0, t]$.
 * @param steps RK4 steps per interval.
 * @returns The `frames` + 1 states, at times $j t / \text{frames}$; if the run stopped early, its last state repeats.
 */
function framesOf(rhs: Rhs, z0: F64, t: Scalar, frames: Size, steps: Size): F64[] {
  if (t === 0) return Array.from({ length: frames + 1 }, () => z0)
  const kept = trace(
    rungeKutta(rhs, 'rk4', { stepSize: t / (frames * steps), tEnd: t }),
    { x0: z0 },
    frames * steps + 1,
    {
      every: steps,
      stopOnNonFinite: false,
    },
  ).steps
  return Array.from({ length: frames + 1 }, (_, k) => dense.data(kept[Math.min(k, kept.length - 1)].x))
}

/** Options for `transportDensityFrames` and `pushForwardDensityFrames`. */
export type TransportFrameOptions = TransportOptions & {
  /** The number of equal time intervals over $[0, t]$; `frames` + 1 times are returned. Default 40. */
  frames?: Size
  /** RK4 steps per frame. Default 1. */
  steps?: Size
}

/**
 * `transportDensity` at `frames` + 1 equally spaced times over $[0, t]$ (for playing the transport): each point's
 * characteristic is followed backwards once, over the whole span, and $\rho$ is read off at every frame.
 *
 * @param f The vector field; differentiable unless `divergence` is given.
 * @param rho0 The initial density $\rho_0$, called with a point as a rank-1 tensor and returning a number.
 * @param points The points at which to evaluate $\rho$, one per row ($k \times n$).
 * @param t The final time.
 * @param options The frames, the steps and the divergence in closed form.
 * @param options.frames The number of equal intervals of $[0, t]$ (default 40).
 * @param options.steps RK4 steps per frame (default 1).
 * @param options.divergence $\nabla \cdot \fvec$ in closed form; without it, by autodiff (much slower).
 * @returns A $(\text{frames} + 1) \times k$ matrix, row $j$ the density at time $j t / \text{frames}$ (row 0 is
 *   $\rho_0$).
 *
 * @example The Gaussian concentrating, frame by frame
 * // x′ = −x: at the origin ρ(0, t) = ρ₀(0) eᵗ.
 * const rho0 = (x) => Math.exp(-0.5 * toFlat(x)[0] ** 2) / Math.sqrt(2 * Math.PI)
 * const frames = transportDensityFrames((x) => neg(x), rho0, [[0], [1]], 1, { frames: 2, steps: 20 })
 * print('rows t = 0, 0.5, 1; columns x = 0, 1:', frames)
 * print('ρ₀(0) eᵗ =', [0, 0.5, 1].map((t) => rho0(tensor([0])) * Math.exp(t)))
 */
export function transportDensityFrames(
  f: VectorField,
  rho0: (x: Tensor) => Scalar,
  points: MatrixLike,
  t: Scalar,
  { frames = 40, steps = 1, divergence: div }: TransportFrameOptions = {},
): Matrix {
  const { data, m, n } = toMatrixF64(points, 'transportDensityFrames')
  const back = withDivergence(f, n, -1, div ?? ((x: Tensor) => divergence(f, x)), 'transportDensityFrames')
  const out = new Float64Array((frames + 1) * m)
  for (let k = 0; k < m; k++) {
    const z0 = new Float64Array(n + 1)
    z0.set(data.subarray(k * n, (k + 1) * n))
    framesOf(back, z0, t, frames, steps).forEach((z, j) => {
      out[j * m + k] = rho0(fromData(Float64Array.from(z.subarray(0, n)), [n])) * Math.exp(-z[n])
    })
  }
  return fromData(out, [frames + 1, m])
}

/**
 * Samples moved along the flow, at `frames` + 1 equally spaced times over $[0, t]$, with the log-volume change
 * $\int_0^t \nabla \cdot \fvec \, ds$ accumulated along each path (the log of the local volume ratio, so the density
 * carried by a sample falls by its exponential along the path).
 *
 * @param f The vector field; differentiable unless `divergence` is given.
 * @param samples The samples, one per row ($k \times n$).
 * @param t The final time.
 * @param options The frames, the steps and the divergence in closed form.
 * @param options.frames The number of equal intervals of $[0, t]$ (default 40).
 * @param options.steps RK4 steps per frame (default 1).
 * @param options.divergence $\nabla \cdot \fvec$ in closed form; without it, by autodiff (much slower).
 * @returns `x`, the moved samples ($(\text{frames} + 1) \times k \times n$), and `logVolume`, the accumulated
 *   divergence ($(\text{frames} + 1) \times k$); frame 0 is the samples themselves with zero log-volume.
 *
 * @example Samples contracting towards the origin
 * // x′ = −x: x(t) = x₀ e^{−t}, and the log-volume falls as −t.
 * const { x, logVolume } = pushForwardDensityFrames((x) => neg(x), [[1], [2]], 1, { frames: 2, steps: 20 })
 * print('x =', x)
 * print('log-volume =', logVolume)
 */
export function pushForwardDensityFrames(
  f: VectorField,
  samples: MatrixLike,
  t: Scalar,
  { frames = 40, steps = 1, divergence: div }: TransportFrameOptions = {},
): { x: Tensor; logVolume: Matrix } {
  const { data, m, n } = toMatrixF64(samples, 'pushForwardDensityFrames')
  const fwd = withDivergence(f, n, 1, div ?? ((x: Tensor) => divergence(f, x)), 'pushForwardDensityFrames')
  const x = new Float64Array((frames + 1) * m * n)
  const logVolume = new Float64Array((frames + 1) * m)
  for (let k = 0; k < m; k++) {
    const z0 = new Float64Array(n + 1)
    z0.set(data.subarray(k * n, (k + 1) * n))
    framesOf(fwd, z0, t, frames, steps).forEach((z, j) => {
      x.set(z.subarray(0, n), (j * m + k) * n)
      logVolume[j * m + k] = z[n]
    })
  }
  return { x: fromData(x, [frames + 1, m, n]), logVolume: fromData(logVolume, [frames + 1, m]) }
}

/**
 * Moves every sample along the flow for time $t$ by `flowMap`: the push-forward of an empirical density.
 *
 * @param f The vector field.
 * @param samples The samples, one per row ($k \times n$).
 * @param t The time to flow for (negative flows backwards).
 * @param options The number of RK4 steps per sample (default 100).
 * @returns The moved samples, one per row ($k \times n$).
 *
 * @example A quarter turn of three samples
 * const f = (x) => stack([neg(get(x, 1)), get(x, 0)])
 * print(pushForwardDensity(f, [[1, 0], [0, 1], [2, 2]], Math.PI / 2))
 */
export function pushForwardDensity(f: VectorField, samples: MatrixLike, t: Scalar, options: FlowOptions = {}): Matrix {
  const { data, m, n } = toMatrixF64(samples, 'pushForwardDensity')
  const out = new Float64Array(m * n)
  for (let k = 0; k < m; k++) out.set(toFlat(flowMap(f, data.subarray(k * n, (k + 1) * n), t, options)), k * n)
  return fromData(out, [m, n])
}
