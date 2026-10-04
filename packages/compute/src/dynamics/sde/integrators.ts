/**
 * The integrators of `aifn-compute/dynamics/sde` (Euler–Maruyama, Milstein, stochastic Runge–Kutta), the `Sde` types and
 * `paths` (Kloeden & Platen, 1992, "Numerical Solution of Stochastic Differential Equations", §9–11). See the
 * module's index for the conventions.
 */

import { grad } from 'aifn-compute/foundation/autodiff'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  fromData,
  isTensor,
  isTraced,
  mul,
  shapeOfValue,
  sum,
  toFlat,
  unwrap,
  type Matrix,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm, Index, Scalar, Shape, Size, Status, Trace } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64

/**
 * An SDE with diagonal noise, evaluated on a whole cloud of states: `drift` and `diffusion` take the time and the
 * states x (any shape, e.g. [N] for N paths of a scalar SDE or [N, d]) and return a value of the same shape (or a
 * number, broadcast). Written with `aifn-compute/foundation/tensor` primitives they apply elementwise and can be differentiated.
 */
export type Sde = {
  /** The drift a(t, x). */
  drift: (t: Scalar, x: Tensor) => Value
  /** The diffusion b(t, x) (diagonal noise). */
  diffusion: (t: Scalar, x: Tensor) => Value
  /** ∂b/∂x elementwise (for Milstein); by autodiff of Σ b when omitted, which is right for diagonal noise. */
  diffusionDerivative?: (t: Scalar, x: Tensor) => Value
}

/** The initial cloud: `x0` (a number, or an array/tensor of one state), repeated for `paths` paths, at time `t0`. */
export type SdeInitial = { x0: Scalar | ArrayLike<number> | Tensor; paths?: Size; t0?: Scalar }

/** The state of an SDE scheme. `t` counts steps; the time is `time`. */
export interface SdeState extends Status {
  /** Steps taken. */
  t: Size
  /** The current time. */
  time: Scalar
  /** The states of all paths: [paths] for a scalar SDE, [paths, d] otherwise. */
  x: Tensor
  /** The size of the last step (0 at t₀). */
  stepSize: Scalar
  /** The Brownian increments ΔW of the last step (same shape as x; zeros at t₀). */
  dW: Tensor
  /** Evaluations of the drift and diffusion (each over the whole cloud) so far. */
  evaluations: Size
  /** True when every path has stopped being finite. */
  diverged: boolean
  /** The number of paths whose state is not finite. */
  nonFinite: Size
}

/** Options common to the schemes. */
export type SdeOptions = {
  /** The step size h (positive). */
  stepSize: Scalar
  /** Stop on reaching this time (the last step is shortened). */
  tEnd?: Scalar
}

function cloud(value: Value, shape: Shape, where: string): F64 {
  const size = shape.reduce((a, b) => a * b, 1)
  if (typeof value === 'number') return new Float64Array(size).fill(value)
  const flat = isTensor(value)
    ? Float64Array.from(toFlat(value))
    : Array.isArray(value) || ArrayBuffer.isView(value)
      ? Float64Array.from(value as ArrayLike<number>)
      : null
  if (!flat) throw new DomainError(where, `${where}: expected a number, tensor or array`)
  if (flat.length === 1) return new Float64Array(size).fill(flat[0])
  if (flat.length !== size) throw new ShapeError(where, `${where}: returned ${flat.length} values for ${size} states`)
  return flat
}

function initial({ x0, paths = 1, t0 = 0 }: SdeInitial): SdeState {
  const one = typeof x0 === 'number' ? [x0] : isTensor(x0) ? toFlat(x0) : Array.from(x0)
  const scalar = typeof x0 === 'number' || one.length === 1
  const d = one.length
  const shape = scalar ? [paths] : [paths, d]
  const x = new Float64Array(paths * d)
  for (let i = 0; i < paths; i++) x.set(one, i * d)
  return {
    t: 0,
    time: t0,
    x: fromData(x, shape),
    stepSize: 0,
    dW: fromData(new Float64Array(paths * d), shape),
    evaluations: 0,
    diverged: !one.every(Number.isFinite),
    nonFinite: one.every(Number.isFinite) ? 0 : paths,
  }
}

/**
 * Brownian increments ΔW ~ N(0, |h|), one per element of `shape`, drawn in row-major order from `s` (a scheme passes
 * the step's stream, `child(root, 'step', t)`), so element i depends only on the stream and i.
 */
export function increments(s: Stream, shape: Shape, stepSize: Scalar): Tensor {
  return fromData(Float64Array.from(toFlat(normals(s, [...shape], 0, Math.sqrt(Math.abs(stepSize))))), [...shape])
}

/** One step: the next states as raw values, or as a value (possibly traced) for a scheme written with primitives. */
type Update = (sde: Sde, t: Scalar, x: Tensor, h: Scalar, dW: Tensor) => { next: F64 | Value; evaluations: Size }

/** A scheme from its one-step update (shared with the exact solutions in `processes.ts`; internal). */
export function scheme(
  name: string,
  sde: Sde,
  { stepSize: h, tEnd }: SdeOptions,
  update: Update,
): Algorithm<SdeInitial, SdeState> {
  if (!(h > 0)) throw new DomainError(name, `${name}: the step size must be positive`)
  return {
    name,
    init: (opts) => initial(opts),
    step: (s, ctx) => {
      const hk = tEnd !== undefined && tEnd - s.time < h ? tEnd - s.time : h
      // A traced state (under `unrolled`) has no `shape` field of its own.
      const shape = shapeOfValue(s.x)
      const dW = increments(ctx.stream, shape, hk)
      const { next, evaluations } = update(sde, s.time, s.x, hk, dW)
      const x = next instanceof Float64Array ? fromData(next, shape) : (next as Tensor)
      const values = next instanceof Float64Array ? next : toFlat(unwrap(x) as Tensor)
      if (values.length !== dense.data(dW).length)
        throw new ShapeError(
          name,
          `${name}: the update gave ${values.length} values for ${dense.data(dW).length} states`,
        )
      let nonFinite = 0
      for (const v of values) if (!Number.isFinite(v)) nonFinite++
      return {
        ...s,
        t: s.t + 1,
        time: s.time + hk,
        x,
        stepSize: hk,
        dW,
        evaluations: s.evaluations + evaluations,
        nonFinite,
        diverged: nonFinite === values.length,
      }
    },
    done: (s) => tEnd !== undefined && s.time >= tEnd - 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}

/**
 * The Euler–Maruyama scheme X_{n+1} = X_n + a(t_n, X_n) h + b(t_n, X_n) ΔW_n with ΔW_n ~ N(0, h): strong order ½
 * (pathwise error O(h^½)) and weak order 1 (error in expectations O(h)). `init` takes `{ x0, paths, t0 }`; step t
 * draws its increments from the runner's step stream. A path that becomes non-finite is counted in `nonFinite`; the run is `diverged` only when all have.
 * The step is written with primitives: with a drift and diffusion written with primitives too, `unrolled` differentiates
 * the paths with respect to the parameters they close over (the pathwise, reparameterised gradient: the increments are
 * fixed by the stream, so each path is a smooth function of the parameters).
 */
export function eulerMaruyama(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  return scheme('euler-maruyama', sde, options, (p, t, x, h, dW) => ({
    next: add(x, add(mul(asValue(p.drift(t, x), x, 'drift'), h), mul(asValue(p.diffusion(t, x), x, 'diffusion'), dW))),
    evaluations: 1,
  }))
}

/** A drift or diffusion as a value that broadcasts against the states x (a plain array becomes a tensor of x's shape). */
function asValue(v: Value, x: Tensor, where: string): Value {
  if (typeof v === 'number' || isTensor(v) || isTraced(v)) return v
  const shape = shapeOfValue(x)
  return fromData(cloud(v, shape, where), shape)
}

/**
 * The Milstein scheme: Euler–Maruyama plus the Itô correction ½ b b′ (ΔW² − h), which raises the strong order to 1
 * for diagonal noise. The derivative b′ = ∂b/∂x comes from `diffusionDerivative` or by autodiff of Σ b(t, x).
 */
export function milstein(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  const db =
    sde.diffusionDerivative ??
    ((t: Scalar, x: Tensor) => grad((y: Value) => sum(sde.diffusion(t, y as Tensor) as Tensor) as Value)(x) as Value)
  return scheme('milstein', sde, options, (p, t, x, h, increments) => {
    const dW = dense.data(increments)
    const a = cloud(p.drift(t, x), x.shape, 'drift')
    const b = cloud(p.diffusion(t, x), x.shape, 'diffusion')
    const bp = cloud(db(t, x), x.shape, 'diffusionDerivative')
    const xs = dense.data(x)
    return {
      next: Float64Array.from(xs, (v, i) => v + a[i] * h + b[i] * dW[i] + 0.5 * b[i] * bp[i] * (dW[i] * dW[i] - h)),
      evaluations: 2,
    }
  })
}

/**
 * Platen's derivative-free explicit scheme of strong order 1 (Kloeden & Platen, 1992, §11.1, eq. 11.1.3): with the
 * supporting value X̂ = X + a h + b √h, X_{n+1} = X + a h + b ΔW + (b(X̂) − b(X))(ΔW² − h)/(2√h). It replaces
 * Milstein's b′ by a finite difference, so the diffusion need not be differentiable.
 */
export function stochasticRungeKutta(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  return scheme('stochastic-runge-kutta', sde, options, (p, t, x, h, increments) => {
    const dW = dense.data(increments)
    const a = cloud(p.drift(t, x), x.shape, 'drift')
    const b = cloud(p.diffusion(t, x), x.shape, 'diffusion')
    const xs = dense.data(x)
    const sq = Math.sqrt(h)
    const support = fromData(
      Float64Array.from(xs, (v, i) => v + a[i] * h + b[i] * sq),
      x.shape,
    )
    const bs = cloud(p.diffusion(t, support), x.shape, 'diffusion')
    return {
      next: Float64Array.from(
        xs,
        (v, i) => v + a[i] * h + b[i] * dW[i] + ((bs[i] - b[i]) * (dW[i] * dW[i] - h)) / (2 * sq),
      ),
      evaluations: 3,
    }
  })
}

/**
 * The path matrix of a traced SDE run: entry [k, i] is path i at kept step k (for a d-dimensional SDE, its
 * `component`). Rows align with `trace.index` and `times`.
 */
export function paths(tr: Trace<SdeState>, component: Index = 0): { times: Tensor; values: Matrix } {
  const first = tr.steps[0].x
  const n = first.shape[0]
  const d = first.shape.length > 1 ? first.shape[1] : 1
  const out = new Float64Array(tr.steps.length * n)
  tr.steps.forEach((s, k) => {
    const x = dense.data(s.x)
    for (let i = 0; i < n; i++) out[k * n + i] = x[i * d + component]
  })
  return {
    times: fromData(
      Float64Array.from(tr.steps, (s) => s.time),
      [tr.steps.length],
    ),
    values: fromData(out, [tr.steps.length, n]),
  }
}
