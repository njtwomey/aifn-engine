/**
 * Flows of autonomous vector fields x′ = f(x): flow maps, trajectories and streamlines by classical RK4, and the
 * transport of densities and samples along the flow by the method of characteristics (Arnold, 1992, "Ordinary
 * Differential Equations", §1–2; Villani, 2003, "Topics in Optimal Transportation", §8.1, for the continuity
 * equation).
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

/** The right-hand side (t, x) ↦ f(x) of the autonomous system x′ = f(x), for the ode solvers. */
export const autonomous =
  (f: VectorField): Rhs =>
  (_t, x) =>
    f(x)

/** Options for integrating a flow with fixed RK4 steps. */
export type FlowOptions = {
  /** The number of RK4 steps over the time span. Default 100. */
  steps?: Size
}

/** The flow map φ_t(x₀): where the trajectory from x₀ is after time t (negative t runs backwards), by RK4. */
export function flowMap(f: VectorField, x0: VectorLike, t: Scalar, { steps = 100 }: FlowOptions = {}): Vector {
  if (t === 0) return dense.vec(toF64(x0, 'flowMap'))
  return run(rungeKutta(autonomous(f), 'rk4', { stepSize: t / steps, tEnd: t }), { x0 }, steps + 1).x
}

/** The trajectory from x₀ over [0, t] by RK4: `time` (steps + 1 times) and `x` ((steps + 1) × n states). */
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

/** A box [lo_i, hi_i] per coordinate. */
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
  /** Stop when the speed ‖f‖ falls below this (the curve has reached a fixed point). Default 1e-9. */
  minSpeed?: Scalar
}

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
 * leaving `bounds`, on reaching a fixed point or on a non-finite state. Returns the points (m × n) in time order.
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

/** Streamlines through every row of `seeds` (k × n). */
export function streamlines(f: VectorField, seeds: MatrixLike, options: StreamlineOptions = {}): Matrix[] {
  const { data, m, n } = toMatrixF64(seeds, 'streamlines')
  return Array.from({ length: m }, (_, i) => streamline(f, data.subarray(i * n, (i + 1) * n), options))
}

/** Options for `transportDensity`. */
export type TransportOptions = {
  /** RK4 steps along each characteristic. Default 50. */
  steps?: Size
  /** ∇·f in closed form, if known (otherwise by autodiff, which is much slower). */
  divergence?: (x: Tensor) => Scalar
}

/**
 * The density ρ(x, t) at the given points (k × n) of an initial density ρ₀ transported by the flow x′ = f(x):
 * the continuity (Liouville) equation ∂ρ/∂t + ∇·(ρf) = 0. Along a trajectory dρ/dt = −ρ ∇·f, so
 * ρ(x, t) = ρ₀(φ₋ₜ(x)) · exp(−∫₀ᵗ ∇·f(φ₋ₛ(x)) ds): each point is followed backwards to its origin while the divergence
 * is accumulated (the method of characteristics). Returns ρ at every point (length k).
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

/** The flow with its divergence integrated alongside, z = (x, L): x′ = sign·f(x), L′ = ∇·f(x). */
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

/** The states of `rhs` from z₀ at `frames` + 1 equally spaced times over [0, t], `steps` RK4 steps per frame. */
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
  /** The number of equal time intervals over [0, t]; frames + 1 times are returned. Default 40. */
  frames?: Size
  /** RK4 steps per frame. Default 1. */
  steps?: Size
}

/**
 * `transportDensity` at `frames` + 1 equally spaced times over [0, t] (for playing the transport): each point's
 * characteristic is followed backwards once, over the whole span, and ρ is read off at every frame. Returns a
 * (frames + 1) × k matrix, row j the density at time j·t/frames.
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
 * Samples (rows of a k × n matrix) moved along the flow, at `frames` + 1 equally spaced times over [0, t], with the
 * log-volume change ∫₀ᵗ ∇·f ds accumulated along each path (the log of the local area ratio, so ρ falls by its
 * exponential along the path). Returns `x` ((frames + 1) × k × n) and `logVolume` ((frames + 1) × k).
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

/** Moves every sample (row of a k × n matrix) along the flow for time t: the push-forward of an empirical density. */
export function pushForwardDensity(f: VectorField, samples: MatrixLike, t: Scalar, options: FlowOptions = {}): Matrix {
  const { data, m, n } = toMatrixF64(samples, 'pushForwardDensity')
  const out = new Float64Array(m * n)
  for (let k = 0; k < m; k++) out.set(toFlat(flowMap(f, data.subarray(k * n, (k + 1) * n), t, options)), k * n)
  return fromData(out, [m, n])
}
