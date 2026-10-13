/**
 * Bayesian optimisation as a step-through algorithm, one objective evaluation per step (Jones, Schonlau and Welch,
 * 1998; Snoek, Larochelle and Adams, 2012; the reviews of Shahriari et al., 2016, and Frazier, 2018).
 *
 * A Gaussian-process surrogate (`aifn-methods/learning/gaussian-processes`) is fitted to the evaluations so far, an
 * acquisition function scores every candidate of a fixed grid, and the objective is evaluated where the score is
 * largest:
 *
 * $$ \xvec_{t+1} = \operatorname*{arg\,max}_{\xvec \in G} \alpha_t(\xvec), $$
 *
 * with $\alpha_t$ the expected improvement, the probability of improvement, the confidence bound, max-value entropy
 * search, or a Thompson draw (one joint posterior sample on the grid, whose optimum is taken). The surrogate works on
 * inputs scaled to the unit cube, $u_j = (x_j - \mathrm{lo}_j) / (\mathrm{hi}_j - \mathrm{lo}_j)$, and on targets
 * standardised by their mean and standard deviation, so the kernel's hyperparameters are on that scale (a lengthscale
 * of 0.2 is a fifth of the box's side); the means, standard deviations and acquisitions a state reports are in the
 * objective's units. Before each choice the hyperparameters can be refitted by maximising the log marginal likelihood
 * (`fitGp`, type-II maximum likelihood), each time from the starting kernel.
 */

import type { Status, Vector } from 'aifn-compute/foundation/contracts'
import { child, permutation, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { matern52, type Kernel, type KernelParams } from 'aifn-compute/learning/kernels'
import { sobol } from 'aifn-compute/numerics/quadrature'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fitGp, gpPosterior } from 'aifn-methods/learning/gaussian-processes'
import { expectedImprovement, probabilityOfImprovement, ucbBeta, upperConfidenceBound } from './acquisitions'
import { maxValueEntropySearch, maxValueSamples } from './entropy'

/** The acquisition functions `bayesianOptimisationSteps` can maximise. */
export type AcquisitionName = 'ei' | 'pi' | 'ucb' | 'thompson' | 'mes'

/**
 * The objective: its value at a point of the box (a vector of $d$ values). It is given a stream of its own for each
 * evaluation, so a noisy objective (such as `svmTuningSurface` with `noise`) draws its noise reproducibly; a
 * deterministic one ignores it. An `Evaluation` (`{ value }`) is accepted too.
 */
export type BayesianObjective = (x: Vector, stream: Stream) => number | { value: number }

/** The search box: the lower and upper corners, $d$ values each (a `TestFunction` is one). */
export type SearchBox = {
  /** The lower corner $\mathrm{lo}$. */
  lo: Vector | readonly number[]
  /** The upper corner $\mathrm{hi}$, each coordinate above the lower one's. */
  hi: Vector | readonly number[]
}

/** The options of `bayesianOptimisationSteps`. */
export interface BayesianOptimisationOptions {
  /**
   * The surrogate's kernel on the unit cube, and the start of every refit (default `matern52` with lengthscale 0.2
   * and variance 1).
   */
  kernel?: Kernel
  /**
   * The acquisition function (default `'ei'`). `'thompson'` draws jointly on the whole grid, factoring its
   * $m \times m$ posterior covariance at a cost of order $m^3$ each step: quick on a one-dimensional grid of a few
   * hundred points, seconds per step on a 41 by 41 grid, so give it a coarser grid in two dimensions.
   */
  acquisition?: AcquisitionName
  /**
   * The initial design: a number of points (default $2d + 1$) placed by `design`, or the points themselves, an
   * $n_0 \times d$ tensor in the box's coordinates (`[n0]` when $d = 1$).
   */
  init?: number | Tensor
  /**
   * How a number of initial points is placed: `'lhs'` (default), a Latin hypercube, one point in each of $n_0$ equal
   * slices of every coordinate; `'sobol'`, the first $n_0$ Sobol points under a random shift modulo 1 (a
   * Cranley–Patterson rotation); `'uniform'`, independent uniform points. All draw from the run's stream.
   */
  design?: 'lhs' | 'sobol' | 'uniform'
  /**
   * The candidates the acquisition is maximised over: a number of points per coordinate (default 201 in one dimension,
   * 41 in two, and $\lfloor 4096^{1/d} \rfloor$ beyond) or one count per coordinate, for a regular grid that includes
   * the box's corners; or the candidates themselves, an $m \times d$ tensor (`[m]` when $d = 1$).
   */
  grid?: number | readonly number[] | Tensor
  /**
   * Refit the kernel's hyperparameters before each choice by maximising the log marginal likelihood
   * (`'marginal-likelihood'`, the default, once there are at least three evaluations), or keep the starting kernel
   * (`false`).
   */
  refit?: 'marginal-likelihood' | false
  /**
   * The observation-noise variance of the standardised targets: a number (default $10^{-6}$, a nearly noise-free
   * objective) held fixed, or `'fit'` to refit it with the kernel (needs `refit`).
   */
  noise?: number | 'fit'
  /** The improvement margin $\xi \ge 0$ of `'ei'` and `'pi'`, in the objective's units (default 0). */
  xi?: number
  /**
   * The exploration weight $\beta$ of `'ucb'`: a number (default 4), or `'srinivas'` for the schedule `ucbBeta` at
   * step $t + 1$ over the grid's size.
   */
  beta?: number | 'srinivas'
  /** The failure probability $\delta$ of the `'srinivas'` schedule (default 0.1). */
  delta?: number
  /** The number of samples of the optimal value for `'mes'` (default 10). */
  maxValues?: number
  /** Minimise the objective (default true); false maximises it. */
  minimise?: boolean
}

/** One state of `bayesianOptimisationSteps`: the evaluations so far, the surrogate on the grid and the next choice. */
export interface BayesianOptimisationState extends Status {
  /** Steps taken: evaluations after the initial design. */
  t: number
  /** Objective evaluations so far, $n$, the initial design included. */
  evaluations: number
  /** The points evaluated, $n \times d$, in the box's coordinates and in order. */
  x: Tensor
  /** The values observed there, `[n]`. */
  y: Tensor
  /** The surrogate's posterior mean $\mu(\xvec)$ on the grid, `[m]`, in the objective's units. */
  mean: Tensor
  /** The posterior standard deviation $\sigma(\xvec)$ of the latent objective on the grid, `[m]`. */
  sd: Tensor
  /** The acquisition on the grid, `[m]`; its largest entry chose `next`. */
  acquisition: Tensor
  /** For `'thompson'`: the posterior draw on the grid, `[m]`, in the objective's units (null otherwise). */
  draw: Tensor | null
  /** For `'mes'`: the samples of the optimal value, `[K]`, by the Gumbel approximation (null otherwise). */
  maxValues: Tensor | null
  /** The next point to evaluate, $d$ values: the grid point where the acquisition is largest. */
  next: Tensor
  /** The index of `next` in the grid (the first of any ties). */
  nextIndex: number
  /** The incumbent: the best value observed (the smallest when minimising). */
  best: number
  /** Where the incumbent was observed, $d$ values. */
  bestX: Tensor
  /** The index of the incumbent among the evaluations (the first of any ties). */
  bestIndex: number
  /** The kernel's hyperparameters used for this choice, on the unit cube and standardised scale. */
  hyper: KernelParams
  /** The observation-noise variance used, on the standardised scale. */
  noiseVariance: number
  /** The log marginal likelihood of the standardised evaluations at these hyperparameters. */
  logMarginal: number
  /** The mean the targets were standardised by: the GP models $(y - \mathrm{shift}) / \mathrm{scale}$. */
  shift: number
  /** The standard deviation the targets were standardised by (1 when it is 0 or there is one evaluation). */
  scale: number
}

/** The algorithm, with the grid its states refer to. */
export interface BayesianOptimisation extends Algorithm<void, BayesianOptimisationState> {
  /** The candidates, $m \times d$ in the box's coordinates; a state's `mean`, `sd` and `acquisition` follow them. */
  readonly grid: Tensor
  /**
   * The grid's counts per coordinate for a regular grid, the last coordinate varying fastest (so in two dimensions
   * entry $i n_2 + j$ is the $i$-th value of $x_1$ with the $j$-th of $x_2$); `[m]` for given candidates.
   */
  readonly shape: readonly number[]
  /** The starting kernel, on the unit cube. */
  readonly kernel: Kernel
}

/**
 * A regular grid on the unit cube with `counts[j]` points along coordinate $j$, corners included, the last coordinate
 * varying fastest.
 *
 * @param counts The number of points per coordinate (each at least 2).
 * @returns The $\prod_j n_j \times d$ points, row-major.
 */
function unitGrid(counts: readonly number[]): Float64Array {
  const d = counts.length
  const m = counts.reduce((a, b) => a * b, 1)
  const out = new Float64Array(m * d)
  for (let i = 0; i < m; i++) {
    let rest = i
    for (let j = d - 1; j >= 0; j--) {
      const k = rest % counts[j]
      rest = Math.floor(rest / counts[j])
      out[i * d + j] = k / (counts[j] - 1)
    }
  }
  return out
}

/**
 * Points given in the box's coordinates as row-major values of an $n \times d$ matrix.
 *
 * @param points A `[n, d]` tensor, or `[n]` when $d = 1$.
 * @param d The dimension.
 * @param what What the points are, for error messages.
 * @returns The $nd$ values.
 */
function rows(points: Tensor, d: number, what: string): Float64Array {
  const shape = points.shape
  if (!((shape.length === 2 && shape[1] === d) || (shape.length === 1 && d === 1)))
    throw new ShapeError(
      'bayesianOptimisationSteps',
      `bayesianOptimisationSteps: ${what} must be [n, ${d}], got [${shape.join(', ')}]`,
    )
  return Float64Array.from(toFlat(points))
}

/**
 * Bayesian optimisation of `objective` over `box` as a step-through algorithm: `init` evaluates the initial design and
 * fits the surrogate; each step evaluates the point the last state chose, refits, and chooses the next. Every state
 * holds the evaluations, the surrogate's mean and standard deviation and the acquisition on the grid, the next point,
 * the incumbent and the hyperparameters, so a figure can replay the run one evaluation at a time. The acquisition is
 * maximised over the grid alone (no local refinement), so the run is deterministic given its stream, and a point may be
 * chosen twice. All randomness (the initial design, Thompson draws, max-value samples, and the objective's own noise)
 * comes from the runner's streams. A state whose value is not finite is flagged `diverged`. Throws `DomainError` for a
 * box whose upper corner is not above its lower one, `'fit'` noise without `refit`, a negative noise or margin, and
 * `ShapeError` for points of the wrong dimension.
 *
 * @param objective The function to optimise, as `BayesianObjective`: a `TestFunction`'s `value` will do.
 * @param box The search box, as `SearchBox`: a `TestFunction` (which carries `lo` and `hi`) will do.
 * @param options The surrogate, acquisition, initial design, grid and refitting, as `BayesianOptimisationOptions`.
 * @returns The algorithm, to run with `run` or `trace` from no start (`undefined`), with its `grid`.
 *
 * @example Expected improvement on a one-dimensional function, by evaluation
 * // f(x) = (6x - 2)^2 sin(12x - 4) on [0, 1], minimum about -6.0207 at x = 0.7572
 * const f = (x) => (6 * x.data[0] - 2) ** 2 * Math.sin(12 * x.data[0] - 4)
 * const alg = bayesianOptimisationSteps(f, { lo: [0], hi: [1] }, { grid: 101 })
 * const tr = trace(alg, undefined, 8, { stream: stream(0), keep: 'all' })
 * print('chosen x:', tr.steps.map((s) => s.next.data[0].toFixed(3)))
 * print('incumbent:', tr.steps.map((s) => s.best.toFixed(3)))
 *
 * @example A Thompson draw chooses the next point, and the posterior at the evaluations matches the data
 * const f = (x) => Math.sin(3 * x.data[0]) + x.data[0] ** 2
 * const alg = bayesianOptimisationSteps(f, { lo: [-2], hi: [2] }, { acquisition: 'thompson', grid: 41, refit: false })
 * const s = run(alg, undefined, 3, { stream: stream(1) })
 * print('evaluated at', s.x, ' values', s.y)
 * print('the draw is smallest at', alg.grid.data[s.nextIndex], '= next', s.next)
 */
export function bayesianOptimisationSteps(
  objective: BayesianObjective,
  box: SearchBox,
  options: BayesianOptimisationOptions = {},
): BayesianOptimisation {
  const where = 'bayesianOptimisationSteps'
  const lo = Float64Array.from(isTensor(box.lo) ? toFlat(box.lo) : box.lo)
  const hi = Float64Array.from(isTensor(box.hi) ? toFlat(box.hi) : box.hi)
  const d = lo.length
  if (d === 0 || hi.length !== d)
    throw new ShapeError(where, `${where}: the box's corners must have the same, positive length`)
  for (let j = 0; j < d; j++)
    if (!(hi[j] > lo[j])) throw new DomainError(where, `${where}: the box's upper corner must be above its lower one`)
  const {
    kernel = matern52({ lengthscale: 0.2, variance: 1 }),
    acquisition = 'ei',
    init = 2 * d + 1,
    design = 'lhs',
    refit = 'marginal-likelihood',
    noise = 1e-6,
    xi = 0,
    beta = 4,
    delta = 0.1,
    maxValues = 10,
    minimise = true,
  } = options
  if (noise === 'fit' && !refit) throw new DomainError(where, `${where}: noise 'fit' needs refit`)
  if (typeof noise === 'number' && !(noise >= 0)) throw new DomainError(where, `${where}: noise must be ≥ 0`)
  if (!(xi >= 0)) throw new DomainError(where, `${where}: xi must be ≥ 0`)

  const toUnit = (v: Float64Array): Float64Array => v.map((x, i) => (x - lo[i % d]) / (hi[i % d] - lo[i % d]))
  const fromUnit = (v: Float64Array): Float64Array => v.map((u, i) => lo[i % d] + u * (hi[i % d] - lo[i % d]))

  // The candidates, on the unit cube and in the box's coordinates.
  const option = options.grid
  let gridShape: number[]
  let gridUnit: Float64Array
  if (isTensor(option)) {
    gridUnit = toUnit(rows(option, d, 'the grid'))
    gridShape = [gridUnit.length / d]
  } else {
    const fallback = d === 1 ? 201 : d === 2 ? 41 : Math.max(2, Math.floor(4096 ** (1 / d)))
    gridShape = Array.isArray(option)
      ? [...(option as readonly number[])]
      : Array.from({ length: d }, () => (option as number | undefined) ?? fallback)
    if (gridShape.length !== d || gridShape.some((c) => !(Number.isInteger(c) && c >= 2)))
      throw new DomainError(where, `${where}: a grid needs an integer count of at least 2 per coordinate`)
    gridUnit = unitGrid(gridShape)
  }
  const m = gridUnit.length / d
  const gridUnitT = fromData(gridUnit, [m, d])
  const grid = fromData(fromUnit(gridUnit), [m, d])

  const evaluate = (x: Float64Array, s: Stream): number => {
    const out = objective(fromData(Float64Array.from(x), [d]), s)
    return typeof out === 'number' ? out : out.value
  }

  const initialDesign = (s: Stream): Float64Array => {
    if (isTensor(init)) return rows(init, d, 'the initial points')
    if (!(Number.isInteger(init) && init >= 1))
      throw new DomainError(where, `${where}: init must be a positive integer or the points`)
    const n0 = init
    const u = new Float64Array(n0 * d)
    if (design === 'uniform') u.set(toFlat(uniform(child(s, 'uniform'), 0, 1, { shape: [n0, d] }) as Tensor))
    else if (design === 'sobol') {
      const base = toFlat(sobol(n0, d))
      const shift = toFlat(uniform(child(s, 'shift'), 0, 1, { shape: [d] }) as Tensor)
      for (let i = 0; i < n0 * d; i++) u[i] = (base[i] + shift[i % d]) % 1
    } else {
      for (let j = 0; j < d; j++) {
        const order = toFlat(permutation(child(s, 'slice', j), n0))
        const jitter = toFlat(uniform(child(s, 'jitter', j), 0, 1, { shape: [n0] }) as Tensor)
        for (let i = 0; i < n0; i++) u[i * d + j] = (order[i] + jitter[i]) / n0
      }
    }
    return fromUnit(u)
  }

  /** Fit the surrogate to the evaluations, score the grid, and choose the next point. */
  const choose = (
    x: Float64Array,
    y: Float64Array,
    t: number,
    s: Stream,
  ): Omit<BayesianOptimisationState, 't' | 'evaluations' | 'x' | 'y'> => {
    const n = y.length
    let bestIndex = 0
    for (let i = 1; i < n; i++) if (minimise ? y[i] < y[bestIndex] : y[i] > y[bestIndex]) bestIndex = i
    const best = y[bestIndex]
    const bestX = fromData(x.slice(bestIndex * d, bestIndex * d + d), [d])
    let shift = 0
    for (const v of y) shift += v / n
    let variance = 0
    for (const v of y) variance += (v - shift) ** 2 / n
    const scale = n > 1 && variance > 0 ? Math.sqrt(variance) : 1
    const X = fromData(toUnit(x), [n, d])
    const Z = fromData(
      y.map((v) => (v - shift) / scale),
      [n],
    )
    let k = kernel
    let noiseVariance = noise === 'fit' ? 0.1 : noise
    if (refit && n >= 3) {
      const fit = fitGp(kernel, X, Z, {
        noiseVariance: noise === 'fit' ? 0.1 : Math.max(noise, 1e-12),
        fitNoise: noise === 'fit',
        maxSteps: 100,
      })
      if (Number.isFinite(fit.logMarginal)) {
        k = fit.kernel
        if (noise === 'fit') noiseVariance = fit.noiseVariance
      }
    }
    const post = gpPosterior(k, X, Z, { noiseVariance })
    const p = post.predict(gridUnitT)
    const mu = toFlat(p.mean)
    const v = toFlat(p.variance)
    const mean = fromData(
      Float64Array.from(mu, (z) => shift + scale * z),
      [m],
    )
    const sd = fromData(
      Float64Array.from(v, (w) => scale * Math.sqrt(Math.max(w, 0))),
      [m],
    )
    let draw: Tensor | null = null
    let samples: Tensor | null = null
    let score: Tensor
    if (acquisition === 'ei') score = expectedImprovement(mean, sd, best, { xi, minimise })
    else if (acquisition === 'pi') score = probabilityOfImprovement(mean, sd, best, { xi, minimise })
    else if (acquisition === 'ucb') {
      const b = beta === 'srinivas' ? ucbBeta(t + 1, { candidates: m, delta }) : beta
      score = upperConfidenceBound(mean, sd, { beta: b, minimise })
    } else if (acquisition === 'thompson') {
      const z = toFlat(post.sample(child(s, 'draw'), gridUnitT, 1).draws)
      draw = fromData(
        Float64Array.from(z, (w) => shift + scale * w),
        [m],
      )
      score = fromData(
        Float64Array.from(toFlat(draw), (w) => (minimise ? -w : w)),
        [m],
      )
    } else if (acquisition === 'mes') {
      samples = maxValueSamples(mean, sd, { count: maxValues, stream: child(s, 'max-values'), best, minimise }).samples
      score = maxValueEntropySearch(mean, sd, { samples, minimise })
    } else throw new DomainError(where, `${where}: unknown acquisition '${acquisition as string}'`)
    const a = toFlat(score)
    let nextIndex = 0
    for (let i = 1; i < m; i++) if (a[i] > a[nextIndex] || Number.isNaN(a[nextIndex])) nextIndex = i
    return {
      mean,
      sd,
      acquisition: score,
      draw,
      maxValues: samples,
      next: fromData(Float64Array.from(toFlat(grid).slice(nextIndex * d, nextIndex * d + d)), [d]),
      nextIndex,
      best,
      bestX,
      bestIndex,
      hyper: k.params,
      noiseVariance,
      logMarginal: post.logMarginal.value,
      shift,
      scale,
    }
  }

  const finite = (y: Float64Array): boolean => y.every(Number.isFinite)

  return {
    name: 'bayesian-optimisation',
    grid,
    shape: gridShape,
    kernel,
    init: (_start, stream) => {
      const x = initialDesign(child(stream, 'design'))
      const n = x.length / d
      const y = Float64Array.from({ length: n }, (_, i) =>
        evaluate(x.subarray(i * d, i * d + d), child(stream, 'evaluate', i)),
      )
      const base = { t: 0, evaluations: n, x: fromData(x, [n, d]), y: fromData(y, [n]) }
      if (!finite(y)) return { ...base, ...placeholder(m, d), diverged: true }
      return { ...base, ...choose(x, y, 0, child(stream, 'acquisition')) }
    },
    step: (state, ctx) => {
      const n = state.evaluations + 1
      const x = new Float64Array(n * d)
      x.set(toFlat(state.x))
      x.set(toFlat(state.next), (n - 1) * d)
      const y = new Float64Array(n)
      y.set(toFlat(state.y))
      y[n - 1] = evaluate(x.subarray((n - 1) * d), child(ctx.stream, 'evaluate'))
      const base = { t: state.t + 1, evaluations: n, x: fromData(x, [n, d]), y: fromData(y, [n]) }
      if (!finite(y)) return { ...base, ...placeholder(m, d), diverged: true }
      return { ...base, ...choose(x, y, state.t + 1, child(ctx.stream, 'acquisition')) }
    },
  }
}

/**
 * The fields of a state that has no surrogate, because an evaluation was not finite: NaN everywhere.
 *
 * @param m The grid's size.
 * @param d The dimension.
 * @returns The surrogate, acquisition and incumbent fields, all NaN.
 */
function placeholder(m: number, d: number): Omit<BayesianOptimisationState, 't' | 'evaluations' | 'x' | 'y'> {
  const nan = (k: number) => fromData(new Float64Array(k).fill(NaN), [k])
  return {
    mean: nan(m),
    sd: nan(m),
    acquisition: nan(m),
    draw: null,
    maxValues: null,
    next: nan(d),
    nextIndex: -1,
    best: NaN,
    bestX: nan(d),
    bestIndex: -1,
    hyper: {},
    noiseVariance: NaN,
    logMarginal: NaN,
    shift: NaN,
    scale: NaN,
  }
}
