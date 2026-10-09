/**
 * The integrators of `aifn-compute/dynamics/sde` (Euler–Maruyama, Milstein, stochastic Runge–Kutta), the `Sde` types
 * and `paths` (Kloeden & Platen, 1992, "Numerical Solution of Stochastic Differential Equations", §9–11).
 *
 * Every scheme is built by `scheme` from its one-step update and steps a whole cloud of paths at once. The increments
 * $\Delta W$ of step $k$ are drawn by `increments` from the runner's step stream in row-major order, so each path's
 * noise depends only on the root key, the step and the path's index (see the module's index for the consequences).
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
 * An SDE $dX = a(t, X) \, dt + b(t, X) \, dW$ with diagonal noise, evaluated on a whole cloud of states: `drift` and
 * `diffusion` take the time and the states $\xvec$ (any shape, such as $[N]$ for $N$ paths of a scalar SDE or
 * $[N, d]$) and return a value of the same shape (or a number, broadcast). Written with
 * `aifn-compute/foundation/tensor` primitives they apply elementwise and can be differentiated.
 */
export type Sde = {
  /** The drift $a(t, \xvec)$. */
  drift: (t: Scalar, x: Tensor) => Value
  /** The diffusion $b(t, \xvec)$ (diagonal noise: each state is driven by its own Brownian motion). */
  diffusion: (t: Scalar, x: Tensor) => Value
  /**
   * $\partial b / \partial x$ elementwise (for Milstein); by autodiff of $\sum b$ when omitted, which is right for
   * diagonal noise.
   */
  diffusionDerivative?: (t: Scalar, x: Tensor) => Value
}

/**
 * The initial cloud: `x0` (a number, or an array or tensor of one state; one value counts as a scalar SDE), repeated
 * for `paths` paths (default 1), at time `t0` (default 0).
 */
export type SdeInitial = { x0: Scalar | ArrayLike<number> | Tensor; paths?: Size; t0?: Scalar }

/** The state of an SDE scheme. `t` counts steps; the time is `time`. */
export interface SdeState extends Status {
  /** Steps taken. */
  t: Size
  /** The current time. */
  time: Scalar
  /** The states of all paths: shape $[\text{paths}]$ for a scalar SDE, $[\text{paths}, d]$ otherwise. */
  x: Tensor
  /** The size of the last step (0 at $t_0$). */
  stepSize: Scalar
  /** The Brownian increments $\Delta W$ of the last step (the shape of `x`; zeros at $t_0$). */
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
  /** The step size $h$ (positive; otherwise the scheme throws `DomainError`). */
  stepSize: Scalar
  /** Stop on reaching this time (the last step is shortened). */
  tEnd?: Scalar
}

/**
 * A drift's or diffusion's value as one number per state: a number or a one-entry value is broadcast, anything else
 * must have exactly one value per state (else `ShapeError`; a value of another kind throws `DomainError`).
 *
 * @param value What the drift, diffusion or its derivative returned: a number, a tensor or an array.
 * @param shape The shape of the states.
 * @param where The caller's name for the value, used in error messages.
 * @returns A new array with one value per state, row-major.
 */
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

/**
 * The state at the start of a run: `x0` repeated for every path, with no step taken.
 *
 * @param options The initial cloud.
 * @param options.x0 The starting state of every path: a number, or an array or tensor of $d$ values (one value is a
 *   scalar SDE).
 * @param options.paths The number of paths (default 1).
 * @param options.t0 The starting time (default 0).
 * @returns The initial state; `diverged` (with every path counted in `nonFinite`) when `x0` is not finite.
 */
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
 * Brownian increments $\Delta W \sim \Gauss(0, \lvert h \rvert)$, one per element of `shape`, drawn in row-major order
 * from `s` (a scheme passes the step's stream, `child(root, 'step', t)`), so element $i$ depends only on the stream
 * and $i$.
 *
 * @param s The stream to draw from.
 * @param shape The shape of the increments, that of the states.
 * @param stepSize The step $h$; the increments have variance $\lvert h \rvert$.
 * @returns The increments, a float64 tensor of shape `shape`.
 *
 * @example More paths keep the existing ones
 * // Element i depends only on the stream and i: the first three of five draws are the three draws.
 * print('5 increments:', increments(stream(1), [5], 0.01))
 * print('3 increments:', increments(stream(1), [3], 0.01))
 *
 * @example Their variance is the step
 * print('variance:', variance(increments(stream(2), [10000], 0.01)))
 */
export function increments(s: Stream, shape: Shape, stepSize: Scalar): Tensor {
  return fromData(Float64Array.from(toFlat(normals(s, [...shape], 0, Math.sqrt(Math.abs(stepSize))))), [...shape])
}

/** One step: the next states as raw values, or as a value (possibly traced) for a scheme written with primitives. */
type Update = (sde: Sde, t: Scalar, x: Tensor, h: Scalar, dW: Tensor) => { next: F64 | Value; evaluations: Size }

/**
 * A scheme from its one-step update (shared with the exact solutions in `processes.ts`; internal). Each step draws
 * the increments for the whole cloud from the step's stream, applies `update`, and counts the paths that are no longer
 * finite. With `tEnd` the last step is shortened to land on it and the run is done there. A step size that is not
 * positive throws `DomainError`; an update that returns the wrong number of values throws `ShapeError`.
 *
 * @param name The scheme's name, used for the algorithm and in error messages.
 * @param sde The SDE passed to `update`.
 * @param options The step size and end time.
 * @param options.stepSize The step size $h$ (positive).
 * @param options.tEnd The time at which to stop, if any (the last step is shortened to reach it).
 * @param update The one-step update: given the SDE, the time, the states, the step and its increments, the next states
 *   (a raw array, or a possibly traced value) and the number of evaluations of the whole cloud it made.
 * @returns The algorithm: `init` takes `{ x0, paths, t0 }` and each `step` advances every path by one step.
 */
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
 * The Euler–Maruyama scheme $X_{n+1} = X_n + a(t_n, X_n) \, h + b(t_n, X_n) \, \Delta W_n$ with
 * $\Delta W_n \sim \Gauss(0, h)$: strong order $\tfrac{1}{2}$ (pathwise error $O(h^{1/2})$) and weak order 1 (error in
 * expectations $O(h)$). `init` takes `{ x0, paths, t0 }`; step $t$ draws its increments from the runner's step
 * stream. A path that becomes non-finite is counted in `nonFinite`; the run is `diverged` only when all have. The step
 * is written with primitives: with a drift and diffusion written with primitives too, `unrolled` differentiates the
 * paths with respect to the parameters they close over (the pathwise, reparameterised gradient: the increments are
 * fixed by the stream, so each path is a smooth function of the parameters).
 *
 * @param sde The SDE: its drift and diffusion (the derivative is not used).
 * @param options The step size and the end time.
 * @returns The algorithm: `init` takes `{ x0, paths, t0 }`, and each `step` advances every path by one step.
 *
 * @example The moments of an Ornstein–Uhlenbeck process
 * // dX = −X dt + dW from 1: at t = 1 the mean is e^{−1} and the variance (1 − e^{−2}) / 2.
 * const sde = { drift: (t, x) => neg(x), diffusion: () => 1 }
 * const end = run(eulerMaruyama(sde, { stepSize: 0.01, tEnd: 1 }), { x0: 1, paths: 2000 }, 200, { stream: stream(0) })
 * print('t =', end.time, 'after', end.t, 'steps')
 * print('mean =', mean(end.x), ' exact', Math.exp(-1))
 * print('variance =', variance(end.x), ' exact', (1 - Math.exp(-2)) / 2)
 */
export function eulerMaruyama(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  return scheme('euler-maruyama', sde, options, (p, t, x, h, dW) => ({
    next: add(x, add(mul(asValue(p.drift(t, x), x, 'drift'), h), mul(asValue(p.diffusion(t, x), x, 'diffusion'), dW))),
    evaluations: 1,
  }))
}

/**
 * A drift or diffusion as a value that broadcasts against the states $\xvec$: numbers, tensors and traced values are
 * returned as they are, and a plain array becomes a tensor of $\xvec$'s shape.
 *
 * @param v The value the drift or diffusion returned.
 * @param x The states, whose shape a plain array takes.
 * @param where The caller's name for the value, used in error messages.
 * @returns A value that broadcasts against `x`.
 */
function asValue(v: Value, x: Tensor, where: string): Value {
  if (typeof v === 'number' || isTensor(v) || isTraced(v)) return v
  const shape = shapeOfValue(x)
  return fromData(cloud(v, shape, where), shape)
}

/**
 * The Milstein scheme: Euler–Maruyama plus the Itô correction $\tfrac{1}{2} b b' (\Delta W^2 - h)$, which raises the
 * strong order to 1 for diagonal noise. The derivative $b' = \partial b / \partial x$ comes from
 * `diffusionDerivative` or by autodiff of $\sum b(t, \xvec)$. The step is computed on plain arrays, so unlike
 * `eulerMaruyama` it is not differentiated by `unrolled`.
 *
 * @param sde The SDE: its drift, diffusion and, optionally, the diffusion's derivative.
 * @param options The step size and the end time.
 * @returns The algorithm: `init` takes `{ x0, paths, t0 }`, and each `step` advances every path by one step.
 *
 * @example Pathwise error against the exact solution
 * // Geometric Brownian motion: on the same stream the exact sampler gives the true path each scheme approximates.
 * const gbm = geometricBrownianMotion({ mu: 0.5, sigma: 1 })
 * const at1 = (alg) => run(alg, { x0: 1, paths: 500 }, 100, { stream: stream(3) }).x
 * const exact = at1(gbm.exact({ stepSize: 0.05, tEnd: 1 }))
 * const error = (alg) => mean(abs(sub(at1(alg), exact)))
 * print('Euler–Maruyama error:', error(eulerMaruyama(gbm.sde, { stepSize: 0.05, tEnd: 1 })))
 * print('Milstein error:', error(milstein(gbm.sde, { stepSize: 0.05, tEnd: 1 })))
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
 * supporting value $\hat{X} = X + a h + b \sqrt{h}$,
 * $X_{n+1} = X + a h + b \, \Delta W + (b(\hat{X}) - b(X))(\Delta W^2 - h) / (2\sqrt{h})$. It replaces Milstein's
 * $b'$ by a finite difference, so the diffusion need not be differentiable. With additive noise ($b$ constant) it is
 * Euler–Maruyama.
 *
 * @param sde The SDE: its drift and diffusion (the derivative is not used).
 * @param options The step size and the end time.
 * @returns The algorithm: `init` takes `{ x0, paths, t0 }`, and each `step` advances every path by one step.
 *
 * @example As accurate as Milstein, without the derivative
 * // Geometric Brownian motion, measured against its exact solution on the same stream.
 * const gbm = geometricBrownianMotion({ mu: 0.5, sigma: 1 })
 * const at1 = (alg) => run(alg, { x0: 1, paths: 500 }, 100, { stream: stream(3) }).x
 * const exact = at1(gbm.exact({ stepSize: 0.05, tEnd: 1 }))
 * const error = (alg) => mean(abs(sub(at1(alg), exact)))
 * print('Euler–Maruyama error:', error(eulerMaruyama(gbm.sde, { stepSize: 0.05, tEnd: 1 })))
 * print('stochastic Runge–Kutta error:', error(stochasticRungeKutta(gbm.sde, { stepSize: 0.05, tEnd: 1 })))
 *
 * @example With additive noise it is Euler–Maruyama
 * const sde = { drift: (t, x) => neg(x), diffusion: () => 1 }
 * const options = { stepSize: 0.1, tEnd: 1 }
 * print('SRK:', run(stochasticRungeKutta(sde, options), { x0: 1, paths: 3 }, 10, { stream: stream(6) }).x)
 * print('EM: ', run(eulerMaruyama(sde, options), { x0: 1, paths: 3 }, 10, { stream: stream(6) }).x)
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
 * The path matrix of a traced SDE run: entry $[k, i]$ is path $i$ at kept step $k$ (for a $d$-dimensional SDE, its
 * `component`). Rows align with `trace.index` and `times`.
 *
 * @param tr The trace of a scheme's run, with its states kept.
 * @param component For a $d$-dimensional SDE, which coordinate of the state to take (default 0); ignored for a scalar
 *   one.
 * @returns `times`, the time of each kept step, and `values`, the $(\text{kept steps}) \times (\text{paths})$ matrix.
 *
 * @example Three Brownian paths
 * const tr = trace(eulerMaruyama(brownianMotion().sde, { stepSize: 0.25 }), { x0: 0, paths: 3 }, 4, {
 *   stream: stream(5),
 * })
 * const { times, values } = paths(tr)
 * print('times =', times)
 * print('paths (one per column) =', values)
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
