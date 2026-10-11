/**
 * Streamed training of a physics-informed neural network (Raissi, Perdikaris & Karniadakis, 2019): a network
 * $u_{\thetavec}(x, t)$ fitted to an equation rather than to data, by minimising
 *
 * $$ L(\thetavec) = \frac{1}{N_r} \sum_i \mathcal{N}[u_{\thetavec}](x_i, t_i)^2 + \frac{w}{N_0} \sum_j \big(u_{\thetavec}(x_j, t_0) - u_0(x_j)\big)^2 + \frac{w}{N_b} \sum_k \big(u_{\thetavec}(x_k, t_k) - g(x_k, t_k)\big)^2, $$
 *
 * the mean squared residual at collocation points inside the domain plus the misfit to the initial profile and to the
 * boundary values. The derivatives in the residual are the network's own, by nested forward mode (`partials` in
 * `aifn-compute/dynamics/fields`), and the loss is differentiated again in $\thetavec$ by reverse mode.
 *
 * Training follows the paper's practice: Adam to get near a good basin, then full-batch L-BFGS (`fullBatchTraining`)
 * to converge, the collocation points fixed throughout (a Sobol set, so they cover the domain evenly). Runs are
 * deterministic in their seed, wall times aside.
 */

import { partials } from 'aifn-compute/dynamics/fields'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves, type Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  mean,
  mul,
  square,
  sub,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import type { Activation } from 'aifn-compute/nn/functional'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { fullBatchTraining } from 'aifn-compute/nn/training'
import { sobol } from 'aifn-compute/numerics/quadrature'
import { adamRule, applyUpdates } from 'aifn-compute/optim/first-order'
import type { PinnDerivatives, PinnProblem } from './problems'

/** Options of {@link pinnRun}; plain data, so a worker task can carry them. */
export type PinnRunOptions = {
  /** Widths of the hidden layers. Default three of 20. */
  hidden?: readonly number[]
  /** The hidden activation; it must be smooth, since the residual takes second derivatives. Default `'tanh'`. */
  activation?: Activation
  /** Collocation points inside the domain, where the residual is penalised (a Sobol set). Default 1024. */
  collocation?: number
  /** Points on the initial line $t = t_0$, evenly spaced. Default 100. */
  initialPoints?: number
  /** Points on each end $x = a$ and $x = b$, evenly spaced in time. Default 50. */
  boundaryPoints?: number
  /** The weight $w$ of the initial and boundary terms against the residual's. Default 1. */
  weight?: number
  /** Adam steps before L-BFGS. Default 1000. */
  adamSteps?: number
  /** Adam's step size. Default 0.005. */
  learningRate?: number
  /** L-BFGS steps after Adam (each a line search, perhaps of several evaluations). Default 500. */
  lbfgsSteps?: number
  /** Seed of the network's initialisation. Default 0. */
  seed?: number
  /** Points of the snapshot grid in $x$ and in $t$. Default `[101, 51]`. */
  grid?: readonly [number, number]
  /** Steps between snapshots (each evaluates the network and its residual on the grid). Default 50. */
  every?: number
}

/** A snapshot of a run. */
export type PinnRun = {
  /** The problem's name. */
  problem: string
  /** The phase the last step was taken in. */
  phase: 'adam' | 'lbfgs'
  /** Steps taken so far, Adam's and L-BFGS's together. */
  done: number
  /** The Adam steps asked for. */
  adamSteps: number
  /** The L-BFGS steps asked for. */
  lbfgsSteps: number
  /** True on the last snapshot. */
  finished: boolean
  /** The message of the error that ended the run early, or null. */
  error: string | null
  /** The number of network parameters. */
  parameters: number
  /** Per step: the loss $L$. */
  loss: Float64Array
  /** Per step: the mean squared residual. */
  residualLoss: Float64Array
  /** Per step: the mean squared misfit to the initial profile. */
  initialLoss: Float64Array
  /** Per step: the mean squared misfit to the boundary values. */
  boundaryLoss: Float64Array
  /** The steps at which the error was measured. */
  errorSteps: Float64Array
  /**
   * The relative error $\lVert u_{\thetavec} - u \rVert_2 / \lVert u \rVert_2$ on the grid at those steps (NaN
   * without an exact solution).
   */
  error2: Float64Array
  /** The grid's positions, $n_x$ values. */
  xs: Float64Array
  /** The grid's times, $n_t$ values. */
  ts: Float64Array
  /** The network on the grid, $[n_t \times n_x]$ row-major, time outer. */
  prediction: Float64Array
  /** The residual $\mathcal{N}[u_{\thetavec}]$ on the grid, laid out as `prediction`. */
  residual: Float64Array
  /** The exact solution on the grid, laid out as `prediction`, or null when the problem has none. */
  solution: Float64Array | null
  /** The collocation points, $[N_r \times 2]$ row-major, each $(x, t)$. */
  collocation: Float64Array
}

/**
 * Evenly spaced values from `a` to `b`.
 *
 * @param a The first value.
 * @param b The last value.
 * @param n The number of values (one gives `a`).
 * @returns The values.
 */
const spaced = (a: number, b: number, n: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)))

/**
 * A matrix of points from their rows.
 *
 * @param rows The points, each $(x, t)$.
 * @returns An $n \times 2$ tensor.
 */
const points = (rows: readonly (readonly [number, number])[]): Tensor =>
  fromData(Float64Array.from(rows.flat()), [rows.length, 2])

/**
 * A rank-0 or one-element value as a number.
 *
 * @param v The value, perhaps traced.
 * @returns Its entry.
 */
const scalarOf = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : toFlat(raw)[0]
}

/**
 * Train a physics-informed network on `problem` and yield snapshots: the losses of every step, and every `every` steps
 * the network, its residual and its error against the exact solution on a grid of the domain. Adam runs first, then
 * L-BFGS from where it stopped; L-BFGS ends early when it converges or stalls. An error during training ends the run
 * with a finished snapshot carrying its message. Deterministic in `seed`, the collocation points being a fixed Sobol
 * set.
 *
 * @param problem The equation, its domain, initial profile and boundary values, as `burgers` and `heat` build them.
 * @param options The network, the points, the weight of the data terms, the two phases and the snapshots; see
 *   `PinnRunOptions`.
 * @returns A generator of `PinnRun` snapshots; the last has `finished` set.
 *
 * @example The heat equation, solved with no data inside the domain
 * const options = { hidden: [12, 12], collocation: 128, initialPoints: 24, boundaryPoints: 12, grid: [21, 11] }
 * let run
 * for (const r of pinnRun(heat(), { ...options, adamSteps: 150, lbfgsSteps: 60, every: 50 })) run = r
 * print('loss: start', run.loss[0], ' after Adam', run.loss[149], ' end', run.loss.at(-1))
 * print('relative error by step:', run.errorSteps, run.error2)
 *
 * @example Burgers' equation: the error is concentrated at the shock
 * const options = { hidden: [16, 16], collocation: 256, initialPoints: 40, boundaryPoints: 20, grid: [41, 3] }
 * let run
 * for (const r of pinnRun(burgers(), { ...options, adamSteps: 200, lbfgsSteps: 100, every: 300 })) run = r
 * const late = run.prediction.length - 41
 * const at = (x) => Math.round((x + 1) * 20)
 * for (const x of [-0.5, -0.05, 0.05, 0.5])
 *   print(`u(${x}, 1): network`, run.prediction[late + at(x)].toFixed(3), ' exact', run.solution[late + at(x)].toFixed(3))
 * print('relative error:', run.error2.at(-1))
 */
export function* pinnRun(problem: PinnProblem, options: PinnRunOptions = {}): Generator<PinnRun, PinnRun> {
  const {
    hidden = [20, 20, 20],
    activation = 'tanh',
    collocation: nr = 1024,
    initialPoints = 100,
    boundaryPoints = 50,
    weight = 1,
    adamSteps = 1000,
    learningRate = 0.005,
    lbfgsSteps = 500,
    seed = 0,
    grid: [nx, nt] = [101, 51],
    every = 50,
  } = options
  const [a, b] = problem.x
  const [t0, t1] = problem.t

  // The network sees its inputs scaled to [−1, 1]², so that tanh starts out of saturation whatever the domain.
  const net = Mlp([2, ...hidden, 1], { activation, init: xavierUniform() })
  const scale = fromData(Float64Array.of(2 / (b - a), 2 / (t1 - t0)), [2])
  const shift = fromData(Float64Array.of(-(a + b) / (b - a), -(t0 + t1) / (t1 - t0)), [2])
  const u = (params: Params, X: Value): Value => net.apply(params as never, add(mul(X, scale), shift))

  // The points: a Sobol set inside, an even row at t₀ and even columns at the two ends.
  const unit = toFlat(sobol(nr, 2, { skip: 1 }))
  const colloc = fromData(
    Float64Array.from(unit, (v, i) => (i % 2 === 0 ? a + (b - a) * v : t0 + (t1 - t0) * v)),
    [nr, 2],
  )
  const x0 = spaced(a, b, initialPoints)
  const initialX = points(Array.from(x0, (x) => [x, t0] as const))
  const initialU = fromData(Float64Array.from(x0, problem.initial), [initialPoints, 1])
  const tb = spaced(t0, t1, boundaryPoints)
  const boundaryX = points([...Array.from(tb, (t) => [a, t] as const), ...Array.from(tb, (t) => [b, t] as const)])
  const ends = Array.from(tb, (t) => problem.boundary(t))
  const boundaryU = fromData(Float64Array.from([...ends.map((e) => e[0]), ...ends.map((e) => e[1])]), [
    2 * boundaryPoints,
    1,
  ])

  /** The derivatives of the network at `X`, each taken when the residual first reads it. */
  const derivatives = (params: Params, X: Value): PinnDerivatives => {
    const f = (Y: Value) => u(params, Y)
    const memo = new Map<string, Value>()
    const get = (key: string, axes: number[]) => {
      if (!memo.has(key)) memo.set(key, partials(f, X, [axes])[0])
      return memo.get(key)!
    }
    return {
      get u() {
        return get('u', [])
      },
      get ut() {
        return get('ut', [1])
      },
      get ux() {
        return get('ux', [0])
      },
      get uxx() {
        return get('uxx', [0, 0])
      },
    }
  }

  // The last evaluation's three terms, read from inside the traced loss.
  let parts = { residual: NaN, initial: NaN, boundary: NaN }
  const loss = (params: Params): Value => {
    const r = mean(square(problem.residual(derivatives(params, colloc))))
    const i = mean(square(sub(u(params, initialX), initialU)))
    const bd = mean(square(sub(u(params, boundaryX), boundaryU)))
    parts = { residual: scalarOf(r), initial: scalarOf(i), boundary: scalarOf(bd) }
    return add(r, mul(weight, add(i, bd)))
  }

  // The snapshot grid, time outer, and the exact solution on it.
  const xs = spaced(a, b, nx)
  const ts = spaced(t0, t1, nt)
  const gridX = points(Array.from(ts).flatMap((t) => Array.from(xs, (x) => [x, t] as const)))
  const exact = problem.solution
  const solution = exact
    ? Float64Array.from({ length: nt * nx }, (_, k) => exact(xs[k % nx], ts[Math.floor(k / nx)]))
    : null
  const norm = solution ? Math.hypot(...solution) : NaN

  const losses: number[] = []
  const residualLoss: number[] = []
  const initialLoss: number[] = []
  const boundaryLoss: number[] = []
  const errorSteps: number[] = []
  const error2: number[] = []
  let prediction = new Float64Array(nt * nx)
  let residual = new Float64Array(nt * nx)
  let phase: 'adam' | 'lbfgs' = 'adam'
  let error: string | null = null

  const record = (value: number) => {
    losses.push(value)
    residualLoss.push(parts.residual)
    initialLoss.push(parts.initial)
    boundaryLoss.push(parts.boundary)
  }
  const measure = (params: Params) => {
    prediction = Float64Array.from(toFlat(u(params, gridX) as Tensor))
    residual = Float64Array.from(toFlat(problem.residual(derivatives(params, gridX)) as Tensor))
    errorSteps.push(losses.length)
    if (!solution) return error2.push(NaN)
    let e = 0
    for (let k = 0; k < prediction.length; k++) e += (prediction[k] - solution[k]) ** 2
    error2.push(Math.sqrt(e) / norm)
  }

  let params = net.init(stream(seed)) as Params
  const parameters = treeLeaves(params).reduce((s, l) => s + toFlat(l.value as Tensor).length, 0)
  const collocation = Float64Array.from(toFlat(colloc))
  const snapshot = (finished: boolean): PinnRun => ({
    problem: problem.name,
    phase,
    done: losses.length,
    adamSteps,
    lbfgsSteps,
    finished,
    error,
    parameters,
    loss: Float64Array.from(losses),
    residualLoss: Float64Array.from(residualLoss),
    initialLoss: Float64Array.from(initialLoss),
    boundaryLoss: Float64Array.from(boundaryLoss),
    errorSteps: Float64Array.from(errorSteps),
    error2: Float64Array.from(error2),
    xs,
    ts,
    prediction,
    residual,
    solution,
    collocation,
  })

  try {
    measure(params)
    // Adam.
    const rule = adamRule({ stepSize: learningRate })
    let state = rule.init(params)
    const lossAndGrad = valueAndGrad(loss as (p: unknown) => Value) as (p: Params) => { value: Value; grad: unknown }
    for (let k = 0; k < adamSteps; k++) {
      const { value, grad } = lossAndGrad(params)
      record(scalarOf(value))
      const next = rule.update(grad as Params, state, params)
      params = applyUpdates(params, next.updates)
      state = next.state
      if (losses.length % every === 0) {
        measure(params)
        yield snapshot(false)
      }
    }
    // L-BFGS from Adam's end; each step records the loss at its accepted point.
    phase = 'lbfgs'
    if (lbfgsSteps > 0) {
      const steps = live(fullBatchTraining({ loss: (p: Params) => loss(p), data: null }), { params })
      let item = steps.next()
      for (let k = 0; k < lbfgsSteps; k++) {
        item = steps.next()
        if (item.done) break
        params = item.value.state.params as Params
        record(item.value.state.value as number)
        if (losses.length % every === 0) {
          measure(params)
          yield snapshot(false)
        }
        if (item.value.stopped) break
      }
    }
    measure(params)
  } catch (e) {
    error = e instanceof Error ? e.message : String(e)
  }
  const last = snapshot(true)
  yield last
  return last
}
