/**
 * Streamed training of a physics-informed network with the variants that `pinnRun` lacks: exact (hard) conditions
 * through a lift and a distance function, trainable coefficients of the equation fitted from observations (an inverse
 * problem), loss weights balanced by the global norms of the terms' gradients, collocation points resampled where the
 * residual is large, random Fourier-feature inputs, causal weights in time and a curriculum on a coefficient. The run
 * is a generator of snapshots, as `pinnRun` is: Adam with the chosen variants, then full-batch L-BFGS on the loss as it
 * stands at the end of Adam (weights and points frozen, causal weights off). Problems are `FieldProblem`s, one
 * dimension in space, steady ($u(x)$) or time-dependent ($u(x, t)$), with Dirichlet ends.
 *
 * Sources: Raissi, Perdikaris and Karniadakis (2019), J. Comput. Phys. 378, §2 (the loss, the inverse problem);
 * Lagaris, Likas and Fotiadis (1998), IEEE TNN 9(5), §3, and Sukumar and Srivastava (2022), CMAME 389 (trial functions
 * $u = g + d \cdot N$); Wang, Sankaran, Wang and Perdikaris (2023), "An expert's guide to training PINNs",
 * arXiv:2308.08468, Algorithm 1, eqs. 2.11 to 2.15 (gradient-norm weights with a moving average, $\alpha = 0.9$;
 * causal weights $w_i = \exp(-\epsilon \sum_{k<i} L_r^k)$) and eq. 4.3 (Fourier features
 * $[\cos \Bmat\xvec, \sin \Bmat\xvec]$, $B_{ij} \sim \Gauss(0, \sigma^2)$); Wu, Zhu, Tan, Kartha and Lu (2023),
 * CMAME 403 (residual-based adaptive distribution,
 * $p(\xvec) \propto \lvert r(\xvec) \rvert^k / \expect\lvert r \rvert^k + c$); Krishnapriyan et al. (2021), §4.1
 * (curriculum, here a continuous ramp).
 *
 * Differences from the papers: the gradient-norm weights are updated every `balanceEvery` steps of a short run rather
 * than every 1000 steps of a long one, and are capped at `maxWeight` (a term whose gradient vanishes once it is
 * satisfied would otherwise have its weight grow without bound); the collocation points are a fixed scrambled Sobol
 * set unless resampled (the guide redraws them every step); resampling draws with replacement from a pool ten times
 * the size of the set. The plain path (soft conditions, fixed weights) differs from `pinnRun` in its points
 * (scrambled Sobol) and its L-BFGS (`methodTraining`), so the two do not give identical numbers.
 */

import { partials } from 'aifn-compute/dynamics/fields'
import { stopGradient, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Params } from 'aifn-compute/foundation/pytree'
import { categorical, child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  cos,
  cumsum,
  exp,
  fromData,
  matmul,
  mean,
  mul,
  sin,
  slice,
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
import { methodTraining, type MethodTrainingState } from 'aifn-compute/nn/training'
import { scrambledSobol } from 'aifn-compute/numerics/quadrature'
import { adamRule, applyUpdates, globalNorm } from 'aifn-compute/optim/first-order'
import type { FieldDerivatives, FieldProblem } from './problems'

/** A term of the loss: the equation's `residual`, the `initial` and `boundary` misfits, and the misfit to `data`. */
export type LossTerm = 'residual' | 'initial' | 'boundary' | 'data'

/** Options of `pinnVariantRun`: plain data, every field optional, so a worker task can carry them. */
export type PinnVariantOptions = {
  /** The hidden widths. Default `[20, 20]`. */
  hidden?: readonly number[]
  /** The hidden activation. Default `'tanh'`. */
  activation?: Activation
  /** The number of collocation points. Default 200. */
  collocation?: number
  /** The number of points on the initial line. Default 50. */
  initialPoints?: number
  /** The points on each end (time-dependent problems; a steady problem has one point per end). Default 25. */
  boundaryPoints?: number
  /** The starting weights of the terms; the residual's is 1 unless given. Default 1 each. */
  weights?: Partial<Record<LossTerm, number>>
  /** `'soft'` penalises the conditions; `'hard'` builds them into $u = g + d \cdot N_{\thetavec}$. Default soft. */
  constraint?: 'soft' | 'hard'
  /** `'fixed'` keeps the weights; `'gradient-norm'` balances them by the terms' gradient norms. Default fixed. */
  balancing?: 'fixed' | 'gradient-norm'
  /** The steps between gradient-norm weight updates. Default 100. */
  balanceEvery?: number
  /**
   * The moving-average factor of the weight updates, $\lambda \leftarrow \alpha\lambda + (1 - \alpha)\hat\lambda$.
   * Default 0.9.
   */
  balanceAlpha?: number
  /**
   * The largest weight a gradient-norm update may set (default $10^6$, the largest starting weight the notes offer).
   * A satisfied term's gradient tends to 0, so without a cap its balanced weight grows without bound.
   */
  maxWeight?: number
  /** `'fixed'` keeps the collocation points; `'residual-adaptive'` redraws them by residual size. Default fixed. */
  sampling?: 'fixed' | 'residual-adaptive'
  /** The steps between redraws. Default 250. */
  resampleEvery?: number
  /** The exponent $k$ of the redraw density $\lvert r \rvert^k / \expect\lvert r \rvert^k + c$. Default 1. */
  resampleK?: number
  /** The offset $c$ of the redraw density. Default 1. */
  resampleC?: number
  /** Random Fourier features of the scaled inputs: count $m$ and scale $\sigma$; null for none. Default null. */
  fourier?: { count: number; scale: number } | null
  /** Causal weights in time: $\epsilon$ and the number of time bins; null for none. Default null. */
  causal?: { epsilon: number; bins: number } | null
  /**
   * Curriculum: the named coefficient rises linearly from `from` to its true value over the first `steps` Adam steps,
   * and stays there (Krishnapriyan et al., 2021, §4.1, as a continuous ramp in place of separate stages). Default null.
   */
  curriculum?: { name: string; from: number; steps: number } | null
  /** Coefficients to fit, by name, with their starting guesses (positive; fitted in log space). Default none. */
  trainable?: Readonly<Record<string, number>>
  /** Condition terms to leave out of the loss, e.g. `['initial']` when the initial profile is unknown. Default none. */
  drop?: readonly LossTerm[]
  /** Observations of $u$: positions, times (time-dependent problems) and values. Default none. */
  data?: { x: readonly number[]; t?: readonly number[]; y: readonly number[] } | null
  /** The number of Adam steps. Default 1000. */
  adamSteps?: number
  /** Adam's step size. Default 0.005. */
  learningRate?: number
  /** The number of L-BFGS steps after Adam. Default 300. */
  lbfgsSteps?: number
  /** The seed of the initialisation, the points and the features. Default 0. */
  seed?: number
  /** The snapshot grid's points in $x$ and $t$. Default `[81, 41]` (steady: `[201, 1]`). */
  grid?: readonly [number, number]
  /** The steps between snapshots. Default 50. */
  every?: number
  /** Measure each term's gradient norm at every snapshot. Default false. */
  measureGradients?: boolean
}

/** A snapshot of a run. */
export type PinnVariantSnapshot = {
  /** The problem's name. */
  problem: string
  /** The phase the last step was taken in. */
  phase: 'adam' | 'lbfgs'
  /** The steps taken so far. */
  done: number
  /** True on the last snapshot. */
  finished: boolean
  /** The message of the error that ended the run early, or null. */
  error: string | null
  /** The loss terms present, in order. */
  terms: LossTerm[]
  /** Per step: the weighted loss. */
  loss: number[]
  /** Per step and term: the term's mean squared value, unweighted. */
  termLoss: Record<LossTerm, number[]>
  /** Per step and term: the term's weight. */
  weights: Record<LossTerm, number[]>
  /** Per step and fitted coefficient: its estimate. */
  coefficients: Record<string, number[]>
  /** The steps at which gradient norms were measured. */
  gradientSteps: number[]
  /** Each term's gradient norm $\lVert \nabla_{\thetavec} L_i \rVert$ at those steps. */
  gradientNorms: Record<LossTerm, number[]>
  /** The steps at which the error was measured. */
  errorSteps: number[]
  /** The relative $L_2$ error on the grid at those steps. */
  error2: number[]
  /** The largest misfit to the initial and boundary values, measured on the conditions' own points. */
  conditionError: number
  /** The grid's positions, $n_x$ values. */
  xs: number[]
  /** The grid's times, $n_t$ values (one, 0, for a steady problem). */
  ts: number[]
  /** The network on the grid, $n_t \times n_x$ row-major, time outer. */
  prediction: number[]
  /** The residual on the grid, laid out as `prediction`. */
  residual: number[]
  /** The exact solution on the grid, laid out as `prediction`, or null when the problem has none. */
  solution: number[] | null
  /** The collocation points now, flat $[N_r \times d]$ rows of $(x)$ or $(x, t)$. */
  collocation: number[]
  /** The causal weights of the time bins at the last step, or null. */
  causalWeights: number[] | null
}

/** The trained quantities: the network's parameters and the logs of the fitted coefficients. */
type Model = { net: Params; log: Record<string, number> }

/**
 * A column vector of values.
 *
 * @param v The values.
 * @returns An $n \times 1$ tensor.
 */
function column(v: readonly number[]): Tensor {
  return fromData(Float64Array.from(v), [v.length, 1])
}

/**
 * A matrix of points from their rows.
 *
 * @param rows The points, each of `d` coordinates.
 * @param d The number of coordinates.
 * @returns An $n \times d$ tensor.
 */
function matrix(rows: readonly (readonly number[])[], d: number): Tensor {
  return fromData(Float64Array.from(rows.flat()), [rows.length, d])
}

/**
 * A rank-0 or one-element value as a number.
 *
 * @param v The value, perhaps traced.
 * @returns Its entry.
 */
function number(v: Value): number {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : toFlat(raw as Tensor)[0]
}

/**
 * A value's entries as an array.
 *
 * @param v The value, perhaps traced.
 * @returns Its entries, row-major.
 */
function flat(v: Value): number[] {
  return toFlat(unwrap(v) as Tensor)
}

/** `valueAndGrad` with the model's type erased, which keeps the checker from expanding the pytree types. */
const gradientOf = valueAndGrad as unknown as (f: (p: Model) => Value) => (p: Model) => { value: Value; grad: Params }

/**
 * Train a physics-informed network on `problem` with the chosen variants, yielding a snapshot every `every` steps and
 * a final one with `finished` set. Deterministic in `seed` (every draw is from a child of the stream
 * `pinn-variants/<seed>`). An error during training, or a non-finite loss, ends the run with a finished snapshot
 * carrying its message; asking for hard conditions on a problem without a lift and a distance throws `DomainError`.
 * A plain function of a problem and plain options, so a figure can stream it from the compute worker with the problem
 * built there (`call('neural/pinn/heatProblem', 0.1)`).
 *
 * @param problem The equation, domain, conditions and (for hard conditions) lift and distance.
 * @param options The network, points, weights, variants, observations and phases; see `PinnVariantOptions`.
 * @returns A generator of snapshots; the last has `finished` set.
 *
 * @example Hard conditions hold exactly whatever the weights
 * const options = { constraint: 'hard', hidden: [8], collocation: 32, adamSteps: 20, lbfgsSteps: 0, grid: [11, 5] }
 * let run
 * for (const r of pinnVariantRun(heatProblem(0.1), options)) run = r
 * print('terms:', run.terms, ' condition error:', run.conditionError)
 * print('loss:', run.loss[0], '->', run.loss.at(-1))
 */
export function* pinnVariantRun(
  problem: FieldProblem,
  options: PinnVariantOptions = {},
): Generator<PinnVariantSnapshot, PinnVariantSnapshot> {
  const o = {
    hidden: [20, 20],
    activation: 'tanh' as Activation,
    collocation: 200,
    initialPoints: 50,
    boundaryPoints: 25,
    constraint: 'soft' as const,
    balancing: 'fixed' as const,
    balanceEvery: 100,
    balanceAlpha: 0.9,
    maxWeight: 1e6,
    sampling: 'fixed' as const,
    resampleEvery: 250,
    resampleK: 1,
    resampleC: 1,
    fourier: null,
    causal: null,
    curriculum: null,
    trainable: {},
    data: null,
    drop: [] as readonly LossTerm[],
    adamSteps: 1000,
    learningRate: 0.005,
    lbfgsSteps: 300,
    seed: 0,
    every: 50,
    measureGradients: false,
    ...options,
  }
  const timed = problem.t !== undefined
  const d = timed ? 2 : 1
  const [a, b] = problem.x
  const [t0, t1] = problem.t ?? [0, 0]
  const hard = o.constraint === 'hard'
  if (hard && (!problem.lift || !problem.distance))
    throw new DomainError('pinnVariantRun', `pinnVariantRun: ${problem.name} has no lift and distance`)
  const root = stream(`pinn-variants/${o.seed}`)

  // Inputs scaled to [−1, 1], then optionally lifted to Fourier features.
  const features = o.fourier
    ? (normals(child(root, 'fourier'), [d, o.fourier.count], 0, o.fourier.scale) as Tensor)
    : null
  const width = features ? 2 * o.fourier!.count : d
  const net = Mlp([width, ...o.hidden, 1], { activation: o.activation, init: xavierUniform() })
  const scaled = (X: Value) => {
    const xs = sub(mul(slice(X, null, [0, 1]), 2 / (b - a)), (a + b) / (b - a))
    if (!timed) return xs
    const ts = sub(mul(slice(X, null, [1, 2]), 2 / (t1 - t0)), (t0 + t1) / (t1 - t0))
    return concat([xs, ts], 1)
  }
  const field = (p: Model) => (X: Value) => {
    let z = scaled(X)
    if (features) {
      const zb = matmul(z, features)
      z = concat([cos(zb), sin(zb)], 1)
    }
    const n = net.apply(p.net as never, z)
    return hard ? add(problem.lift!(X), mul(problem.distance!(X), n)) : n
  }
  const ramp = { ...problem.coefficients }
  const rampTo = (k: number) => {
    const cur = o.curriculum
    if (!cur) return
    const target = problem.coefficients[cur.name]
    ramp[cur.name] = cur.from + (target - cur.from) * Math.min(1, k / Math.max(1, cur.steps))
  }
  rampTo(0)
  const coefficientsOf = (p: Model): Record<string, Value> => {
    const c: Record<string, Value> = { ...ramp }
    for (const k of Object.keys(o.trainable)) c[k] = exp(p.log[k] as Value)
    return c
  }
  const orders: number[][] = [[]]
  const names: (keyof FieldDerivatives)[] = ['u']
  for (const need of problem.needs) {
    names.push(need)
    orders.push(need === 'ux' ? [0] : need === 'uxx' ? [0, 0] : [1])
  }
  const residualAt = (p: Model, X: Value) => {
    const vals = partials(field(p), X, orders)
    const got: Record<string, Value> = {}
    names.forEach((k, i) => (got[k] = vals[i]))
    const ds: FieldDerivatives = {
      u: got.u,
      ux: got.ux ?? 0,
      uxx: got.uxx ?? 0,
      ut: got.ut ?? 0,
      x: slice(X, null, [0, 1]),
      t: timed ? slice(X, null, [1, 2]) : null,
    }
    return problem.residual(ds, coefficientsOf(p))
  }

  // Collocation points: a scrambled Sobol set mapped onto the domain.
  const toDomain = (unit: number[], n: number) => {
    const rows: number[][] = []
    for (let i = 0; i < n; i++) {
      const r = [a + (b - a) * unit[i * d]]
      if (timed) r.push(t0 + (t1 - t0) * unit[i * d + 1])
      rows.push(r)
    }
    return rows
  }
  let points = toDomain(toFlat(scrambledSobol(child(root, 'collocation'), o.collocation, d)), o.collocation)
  let X = matrix(points, d)

  // Causal bins: an indicator matrix [M, n] with rows divided by their counts.
  const binsOf = (rows: number[][]) => {
    if (!o.causal || !timed) return null
    const M = o.causal.bins
    const ind = new Float64Array(M * rows.length)
    const counts = new Array(M).fill(0)
    const bin = rows.map((r) => Math.min(M - 1, Math.floor(((r[1] - t0) / (t1 - t0)) * M)))
    bin.forEach((k) => counts[k]++)
    bin.forEach((k, i) => (ind[k * rows.length + i] = 1 / Math.max(1, counts[k])))
    return fromData(ind, [M, rows.length])
  }
  let bins = binsOf(points)
  let causalOn = true

  // Condition points and values.
  const initialRows: number[][] = []
  const initialValues: number[] = []
  const boundaryRows: number[][] = []
  const boundaryValues: number[] = []
  if (timed) {
    for (let j = 0; j < o.initialPoints; j++) {
      const x = a + ((b - a) * j) / (o.initialPoints - 1)
      initialRows.push([x, t0])
      initialValues.push(problem.initial!(x))
    }
    for (let k = 0; k < o.boundaryPoints; k++) {
      const t = t0 + ((t1 - t0) * k) / (o.boundaryPoints - 1)
      const [ga, gb] = problem.boundary(t)
      boundaryRows.push([a, t], [b, t])
      boundaryValues.push(ga, gb)
    }
  } else {
    const [ga, gb] = problem.boundary(0)
    boundaryRows.push([a], [b])
    boundaryValues.push(ga, gb)
  }
  const X0 = initialRows.length ? matrix(initialRows, d) : null
  const Y0 = column(initialValues)
  const Xb = matrix(boundaryRows, d)
  const Yb = column(boundaryValues)
  const data = o.data
  const Xd = data
    ? matrix(
        data.x.map((x, i) => (timed ? [x, data.t![i]] : [x])),
        d,
      )
    : null
  const Yd = data ? column(data.y) : null

  const terms: LossTerm[] = ['residual']
  if (!hard && timed && !o.drop.includes('initial')) terms.push('initial')
  if (!hard && !o.drop.includes('boundary')) terms.push('boundary')
  if (data) terms.push('data')
  const weight: Record<LossTerm, number> = { residual: 1, initial: 1, boundary: 1, data: 1, ...o.weights }

  const termValue = (p: Model, term: LossTerm): Value => {
    if (term === 'residual') {
      const r2 = square(residualAt(p, X))
      if (!bins || !causalOn) return mean(r2)
      const per = matmul(bins, r2)
      const fixed = stopGradient(per)
      const w = exp(mul(-o.causal!.epsilon, sub(cumsum(fixed, 0), fixed)))
      return mean(mul(w, per))
    }
    if (term === 'initial') return mean(square(sub(field(p)(X0!), Y0)))
    if (term === 'boundary') return mean(square(sub(field(p)(Xb), Yb)))
    return mean(square(sub(field(p)(Xd!), Yd!)))
  }
  const totalLoss = (p: Model) => {
    let total: Value = 0
    for (const term of terms) total = add(total, mul(weight[term], termValue(p, term)))
    return total
  }

  // Grid for snapshots.
  const [nx, nt] = o.grid ?? (timed ? [81, 41] : [201, 1])
  const xs = Array.from({ length: nx }, (_, i) => a + ((b - a) * i) / (nx - 1))
  const ts = timed ? Array.from({ length: nt }, (_, i) => t0 + ((t1 - t0) * i) / (nt - 1)) : [0]
  const gridRows: number[][] = []
  for (const t of ts) for (const x of xs) gridRows.push(timed ? [x, t] : [x])
  const G = matrix(gridRows, d)
  const solution = problem.solution ? gridRows.map((r) => problem.solution!(r[0], timed ? r[1] : 0)) : null
  const solutionNorm = solution ? Math.hypot(...solution) : NaN
  const conditionRows = matrix([...initialRows, ...boundaryRows], d)
  const conditionValues = [...initialValues, ...boundaryValues]

  let model: Model = {
    net: net.init(child(root, 'init')) as Params,
    log: Object.fromEntries(Object.entries(o.trainable).map(([k, v]) => [k, Math.log(v)])),
  }
  const history = {
    loss: [] as number[],
    termLoss: Object.fromEntries(terms.map((t) => [t, [] as number[]])) as Record<LossTerm, number[]>,
    weights: Object.fromEntries(terms.map((t) => [t, [] as number[]])) as Record<LossTerm, number[]>,
    coefficients: Object.fromEntries(Object.keys(o.trainable).map((k) => [k, [] as number[]])) as Record<
      string,
      number[]
    >,
    gradientSteps: [] as number[],
    gradientNorms: Object.fromEntries(terms.map((t) => [t, [] as number[]])) as Record<LossTerm, number[]>,
    errorSteps: [] as number[],
    error2: [] as number[],
  }
  const termGradients = (p: Model) =>
    terms.map((term) => globalNorm(gradientOf((q: Model) => termValue(q, term))(p).grad))
  const record = (p: Model, loss: number) => {
    history.loss.push(loss)
    for (const term of terms) {
      history.termLoss[term].push(number(termValue(p, term)))
      history.weights[term].push(weight[term])
    }
    for (const k of Object.keys(o.trainable)) history.coefficients[k].push(Math.exp(p.log[k]))
  }
  let done = 0
  let phase: 'adam' | 'lbfgs' = 'adam'
  const snapshot = (finished: boolean, error: string | null = null): PinnVariantSnapshot => {
    const prediction = flat(field(model)(G))
    const residual = flat(residualAt(model, G))
    if (solution) {
      let s = 0
      for (let i = 0; i < prediction.length; i++) s += (prediction[i] - solution[i]) ** 2
      history.errorSteps.push(done)
      history.error2.push(Math.sqrt(s) / solutionNorm)
    }
    if (o.measureGradients) {
      history.gradientSteps.push(done)
      termGradients(model).forEach((g, i) => history.gradientNorms[terms[i]].push(g))
    }
    const atConditions = flat(field(model)(conditionRows))
    let causalWeights: number[] | null = null
    if (bins) {
      const per = flat(matmul(bins, square(residualAt(model, X))))
      let cum = 0
      causalWeights = per.map((l) => {
        const w = Math.exp(-o.causal!.epsilon * cum)
        cum += l
        return w
      })
    }
    const copy = (r: Record<string, number[]>) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, [...v]]))
    return {
      problem: problem.name,
      phase,
      done,
      finished,
      error,
      terms,
      loss: [...history.loss],
      termLoss: copy(history.termLoss) as Record<LossTerm, number[]>,
      weights: copy(history.weights) as Record<LossTerm, number[]>,
      coefficients: copy(history.coefficients),
      gradientSteps: [...history.gradientSteps],
      gradientNorms: copy(history.gradientNorms) as Record<LossTerm, number[]>,
      errorSteps: [...history.errorSteps],
      error2: [...history.error2],
      conditionError: atConditions.reduce((m, v, i) => Math.max(m, Math.abs(v - conditionValues[i])), 0),
      xs,
      ts,
      prediction,
      residual,
      solution,
      collocation: points.flat(),
      causalWeights,
    }
  }

  try {
    yield snapshot(false)
    // Adam with the variants.
    const rule = adamRule({ stepSize: o.learningRate })
    let ruleState = rule.init(model as unknown as Params)
    const step = gradientOf(totalLoss)
    for (let k = 1; k <= o.adamSteps; k++) {
      rampTo(k)
      if (o.balancing === 'gradient-norm' && (k - 1) % o.balanceEvery === 0) {
        const norms = termGradients(model)
        const total = norms.reduce((s, g) => s + g, 0)
        terms.forEach((term, i) => {
          const target = norms[i] > 0 ? total / norms[i] : weight[term]
          weight[term] = Math.min(o.maxWeight, o.balanceAlpha * weight[term] + (1 - o.balanceAlpha) * target)
        })
      }
      if (o.sampling === 'residual-adaptive' && k > 1 && (k - 1) % o.resampleEvery === 0) {
        const pool = toDomain(toFlat(scrambledSobol(child(root, 'pool', k), 10 * o.collocation, d)), 10 * o.collocation)
        const r = flat(residualAt(model, matrix(pool, d))).map((v) => Math.abs(v) ** o.resampleK)
        const m = r.reduce((s, v) => s + v, 0) / r.length || 1
        const p = r.map((v) => v / m + o.resampleC)
        const s: Stream = child(root, 'draw', k)
        const picks = toFlat(categorical(s, p, { shape: [o.collocation] }) as Tensor)
        points = picks.map((i) => pool[i])
        X = matrix(points, d)
        bins = binsOf(points)
      }
      const { value, grad } = step(model)
      const { updates, state } = rule.update(grad, ruleState, model as unknown as Params)
      ruleState = state
      model = applyUpdates(model as unknown as Params, updates) as unknown as Model
      done++
      record(model, number(value))
      if (!Number.isFinite(number(value))) return snapshot(true, 'the loss diverged')
      if (k % o.every === 0 && !(k === o.adamSteps && o.lbfgsSteps === 0)) yield snapshot(false)
    }
    // L-BFGS on the loss as it stands: weights and points frozen, causal weights off.
    if (o.lbfgsSteps > 0) {
      phase = 'lbfgs'
      causalOn = false
      const batch = { i: column([0]) }
      const it = live(
        methodTraining((p: Params) => totalLoss(p as unknown as Model), batch, { method: 'lbfgs' }),
        {
          params: model as unknown as Params,
        },
      )
      it.next()
      for (let k = 1; k <= o.lbfgsSteps; k++) {
        const next = it.next()
        if (next.done) break
        const s = next.value.state as MethodTrainingState<Params>
        if (s.diverged) break
        model = s.params as unknown as Model
        done++
        record(model, s.loss)
        if (s.stopped) break
        if (k % o.every === 0 && k < o.lbfgsSteps) yield snapshot(false)
      }
    }
    return snapshot(true)
  } catch (e) {
    return snapshot(true, e instanceof Error ? e.message : String(e))
  }
}
