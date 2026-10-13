/**
 * Bayesian optimisation as a step-through algorithm, one objective evaluation (or one batch of them) per step (Jones,
 * Schonlau and Welch, 1998; Snoek, Larochelle and Adams, 2012; the reviews of Shahriari et al., 2016, and Frazier,
 * 2018).
 *
 * A surrogate is fitted to the evaluations so far, an acquisition function scores every candidate of a fixed grid,
 * and the objective is evaluated where the score is largest:
 *
 * $$ \xvec_{t+1} = \operatorname*{arg\,max}_{\xvec \in G} \alpha_t(\xvec), $$
 *
 * with $\alpha_t$ the expected improvement, the probability of improvement, the confidence bound, max-value entropy
 * search, or a Thompson draw (one joint posterior sample on the grid, whose optimum is taken). The surrogate is a
 * Gaussian process (`aifn-methods/learning/gaussian-processes`) or, as in SMAC (Hutter, Hoos and Leyton-Brown, 2011),
 * a random forest of regression trees whose mean and spread over trees stand for the posterior mean and standard
 * deviation. It works on inputs scaled to the unit cube, $u_j = (x_j - \mathrm{lo}_j) / (\mathrm{hi}_j -
 * \mathrm{lo}_j)$, and on targets standardised by their mean and standard deviation, so the kernel's hyperparameters
 * are on that scale (a lengthscale of 0.2 is a fifth of the box's side); the means, standard deviations and
 * acquisitions a state reports are in the objective's units. Before each choice the hyperparameters can be refitted by
 * maximising the log marginal likelihood (`fitGp`, type-II maximum likelihood), each time from the starting kernel.
 *
 * The options extend the loop in the directions of the literature:
 *
 * - a log-transformed cost (`logCost`), modelled as $\ln f$ with the log-normal expected improvement, and random points
 *   interleaved every $k$-th choice (`interleaveRandom`), both as SMAC;
 * - batches of $q$ points per step (`batch`), by the kriging believer and constant liars of Ginsbourger, Le Riche and
 *   Carraro (2010), local penalisation (González et al., 2016) or $q$ Thompson draws;
 * - unknown constraints $c_k(\xvec) \le \lambda_k$ (`constraints`; Schonlau, Welch and Jones, 1998; Gardner et al.,
 *   2014; Gelbart, Snoek and Adams, 2014), one Gaussian process per constraint, with the expected improvement over
 *   the best feasible value weighted by the probability of feasibility
 *   $\prod_k \Phi((\lambda_k - \mu_k(\xvec)) / \sigma_k(\xvec))$.
 *
 * The objective is a function, or a registered objective named by its key (`'forrester'`, `'branin'`,
 * `'svm-tuning-surface'`, with parameters as `{ name, ...parameters }`), whose box is then the default; a named run is
 * plain data, so it can be sent to a worker.
 */

import type { Status, Vector } from 'aifn-compute/foundation/contracts'
import { child, integers, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { dataset } from 'aifn-compute/learning/estimators'
import { matern52, type Kernel, type KernelParams } from 'aifn-compute/learning/kernels'
import { normalCdf } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { gpPosterior } from 'aifn-methods/learning/gaussian-processes'
import {
  randomForestRegressor,
  type RandomForestRegressorParams,
} from 'aifn-methods/learning/trees-and-ensembles/bagging'
import {
  expectedImprovement,
  expectedImprovementLogNormal,
  probabilityOfImprovement,
  ucbBeta,
  upperConfidenceBound,
} from './acquisitions'
import { gridLipschitz, localPenalisation } from './batch'
import {
  argmax,
  bestIndexOf,
  candidateGrid,
  designPoints,
  gpSurrogate,
  readBox,
  standardise,
  type SearchBox,
} from './common'
import { maxValueEntropySearch, maxValueSamples } from './entropy'
import { isNamedObjective, resolveObjective, type NamedObjective } from './named'

export type { SearchBox } from './common'
export type { NamedObjective, ObjectiveName } from './named'

/** The acquisition functions `bayesianOptimisationSteps` can maximise. */
export type AcquisitionName = 'ei' | 'pi' | 'ucb' | 'thompson' | 'mes'

/** How `bayesianOptimisationSteps` chooses a batch of points. */
export type BatchStrategy =
  | 'kriging-believer'
  | 'constant-liar-min'
  | 'constant-liar-mean'
  | 'constant-liar-max'
  | 'local-penalisation'
  | 'thompson'

/**
 * The value of an evaluation: a number, or `{ value }` (an `Evaluation`), or with constraints `{ value, constraints }`,
 * the constraint functions $c_k(\xvec)$ in the order of the `constraints` option.
 */
export type ObjectiveValue = number | { value: number; constraints?: readonly number[] | Vector }

/**
 * The objective: its value at a point of the box (a vector of $d$ values). It is given a stream of its own for each
 * evaluation, so a noisy objective (such as `svmTuningSurface` with `noise`) draws its noise reproducibly; a
 * deterministic one ignores it.
 */
export type BayesianObjective = (x: Vector, stream: Stream) => ObjectiveValue

/** The batch option of `bayesianOptimisationSteps`. */
export type BatchOptions = {
  /** The number $q \ge 1$ of points evaluated per step. */
  size: number
  /**
   * How the $q$ points are chosen (default `'kriging-believer'`). `'kriging-believer'` fits the surrogate to each
   * chosen point with its posterior mean as the value (hyperparameters kept) and maximises the acquisition again;
   * `'constant-liar-min'`, `'-mean'` and `'-max'` do the same with the smallest, mean or largest value observed
   * (Ginsbourger et al., 2010); `'local-penalisation'` multiplies the acquisition by `localPenalisation` factors around
   * the chosen points, without refitting (González et al., 2016), with $L$ the largest slope of the mean on the unit
   * cube and $M$ the better of the incumbent and the best mean on the grid; `'thompson'` takes the optima of $q$ joint
   * posterior draws on the grid (Gaussian-process surrogate only).
   */
  strategy?: BatchStrategy
}

/** The options of `bayesianOptimisationSteps`. */
export interface BayesianOptimisationOptions {
  /**
   * The surrogate: a Gaussian process (`'gp'`, the default) or a random forest of regression trees
   * (`'random-forest'`, `randomForestRegressor`), whose mean and the standard deviation of its trees' predictions
   * (SMAC's) are the surrogate's mean and standard deviation.
   */
  surrogate?: 'gp' | 'random-forest'
  /** The forest of `'random-forest'` (default 10 trees, scikit-learn's other defaults), as `randomForestRegressor`. */
  forest?: RandomForestRegressorParams
  /**
   * The Gaussian process's kernel on the unit cube, and the start of every refit (default `matern52` with lengthscale
   * 0.2 and variance 1); also the kernel of every constraint's process.
   */
  kernel?: Kernel
  /**
   * The acquisition function (default `'ei'`). `'thompson'` draws jointly on the whole grid, factoring its
   * $m \times m$ posterior covariance at a cost of order $m^3$ each step: quick on a one-dimensional grid of a few
   * hundred points, seconds per step on a 41 by 41 grid, so give it a coarser grid in two dimensions. It needs the
   * Gaussian-process surrogate.
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
  /**
   * Model the logarithm of a positive cost (default false), as SMAC does for runtimes: the surrogate is fitted to
   * $\ln y$, a state's `mean` and `sd` are those of $\ln f$, and `'ei'` is `expectedImprovementLogNormal`, the
   * expected improvement in the cost itself (its margin $\xi$ is not used); the other acquisitions score $\ln f$
   * against $\ln f^+$. Needs `minimise`; a value that is not positive throws `DomainError`.
   */
  logCost?: boolean
  /**
   * Choose every $k$-th point uniformly at random from the grid instead of by the acquisition (SMAC interleaves
   * random configurations so that a misleading surrogate cannot stall the search): the choice made in state $t$ is
   * random when $t + 1$ is a multiple of $k$. Left out, or 0, never.
   */
  interleaveRandom?: number
  /** Evaluate a batch of points per step, as `BatchOptions` (default one point). */
  batch?: BatchOptions
  /**
   * Constraints $c_k(\xvec) \le \lambda_k$, one per entry, with $\lambda_k$ its `threshold`: the objective then
   * returns `{ value, constraints }`. Each constraint has a Gaussian process of its own, the acquisition (`'ei'` or
   * `'pi'`, against the best feasible value) is weighted by the probability of feasibility, and it is that probability
   * alone while no evaluation is feasible. Not with `batch`.
   */
  constraints?: readonly { threshold: number }[]
}

/** One state of `bayesianOptimisationSteps`: the evaluations so far, the surrogate on the grid and the next choice. */
export interface BayesianOptimisationState extends Status {
  /** Steps taken: batches evaluated after the initial design. */
  t: number
  /** Objective evaluations so far, $n$, the initial design included. */
  evaluations: number
  /** The points evaluated, $n \times d$, in the box's coordinates and in order. */
  x: Tensor
  /** The values observed there, `[n]`. */
  y: Tensor
  /**
   * The surrogate's mean $\mu(\xvec)$ on the grid, `[m]`, in the objective's units (of $\ln f$ with `logCost`); for a
   * forest, the mean of its trees' predictions.
   */
  mean: Tensor
  /**
   * The surrogate's standard deviation $\sigma(\xvec)$ on the grid, `[m]`: of the latent objective for a Gaussian
   * process, of the trees' predictions (divided by their number) for a forest.
   */
  sd: Tensor
  /**
   * The acquisition on the grid, `[m]`; its largest entry chose `next` (unless `random`). With a batch it is the first
   * row of `batchAcquisition` (a negated Thompson draw for the `'thompson'` strategy).
   */
  acquisition: Tensor
  /** For `'thompson'`: the posterior draw on the grid, `[m]`, in the objective's units (null otherwise). */
  draw: Tensor | null
  /** For `'mes'`: the samples of the optimal value, `[K]`, by the Gumbel approximation (null otherwise). */
  maxValues: Tensor | null
  /** The next point to evaluate, $d$ values: the grid point where the acquisition is largest (the batch's first). */
  next: Tensor
  /** The index of `next` in the grid (the first of any ties). */
  nextIndex: number
  /** True when `next` (and the rest of its batch) was drawn at random by `interleaveRandom`. */
  random: boolean
  /** For a forest: every tree's prediction on the grid, $B \times m$, in the units of `mean` (null otherwise). */
  treePredictions: Tensor | null
  /** With `batch`: the $q$ points to evaluate next, $q \times d$ (null otherwise). */
  batch: Tensor | null
  /** With `batch`: their indices in the grid. */
  batchIndices: number[] | null
  /**
   * With a kriging believer or constant liar: the value each batch point was taken to have when the next was chosen,
   * `[q]`, in the units of `mean` (null otherwise).
   */
  fantasies: Tensor | null
  /**
   * With `batch`: the score each batch point was chosen by, $q \times m$ (row $j$ chose point $j$): the acquisition
   * after the fantasies or penalties of the points before it, or a negated Thompson draw.
   */
  batchAcquisition: Tensor | null
  /** With `constraints`: the constraint values observed, $n \times K$ (null otherwise). */
  constraintValues: Tensor | null
  /** With `constraints`: each constraint's posterior mean on the grid, $K \times m$. */
  constraintMean: Tensor | null
  /** With `constraints`: each constraint's posterior standard deviation on the grid, $K \times m$. */
  constraintSd: Tensor | null
  /** With `constraints`: the probability of feasibility on the grid, $\prod_k \Phi((\lambda_k - \mu_k) / \sigma_k)$. */
  feasibility: Tensor | null
  /** With `constraints`: the best value among the feasible evaluations (NaN when none is, or without constraints). */
  feasibleBest: number
  /** With `constraints`: the index of that evaluation (-1 when none). */
  feasibleBestIndex: number
  /** The incumbent: the best value observed (the smallest when minimising), feasible or not. */
  best: number
  /** Where the incumbent was observed, $d$ values. */
  bestX: Tensor
  /** The index of the incumbent among the evaluations (the first of any ties). */
  bestIndex: number
  /** The kernel's hyperparameters used for this choice, on the unit cube and standardised scale ({} for a forest). */
  hyper: KernelParams
  /** The observation-noise variance used, on the standardised scale (NaN for a forest). */
  noiseVariance: number
  /** The log marginal likelihood of the standardised evaluations at these hyperparameters (NaN for a forest). */
  logMarginal: number
  /** The mean the targets were standardised by: the surrogate models $(y - \mathrm{shift}) / \mathrm{scale}$. */
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
  /** The search box's corners, $d$ values each. */
  readonly box: { lo: Tensor; hi: Tensor }
  /** The registry key of a named objective (null for a function). */
  readonly objective: string | null
  /** The known minimum value of a named objective (NaN otherwise), for regret. */
  readonly minimum: number
}

/** A fitted surrogate on the grid, in standardised units. */
type Model = {
  /** The mean on the grid. */
  mu: Float64Array
  /** The variance on the grid. */
  v: Float64Array
  /** A forest's tree predictions on the grid, $B \times m$ (null for a Gaussian process). */
  trees: Float64Array | null
  /** The number of trees. */
  B: number
  /** Joint posterior draws on the grid, `count` rows of $m$ (null for a forest). */
  draw: ((s: Stream, count: number) => Float64Array) | null
  /** The kernel used (null for a forest). */
  kernel: Kernel | null
  /** The hyperparameters, noise and log marginal likelihood reported. */
  hyper: KernelParams
  noiseVariance: number
  logMarginal: number
}

/**
 * Bayesian optimisation of `objective` over `box` as a step-through algorithm: `init` evaluates the initial design and
 * fits the surrogate; each step evaluates the point (or the batch) the last state chose, refits, and chooses again.
 * Every state holds the evaluations, the surrogate's mean and standard deviation and the acquisition on the grid, the
 * next point, the incumbent and the hyperparameters, so a figure can replay the run one evaluation at a time; batches,
 * forests and constraints add their own fields (see `BayesianOptimisationState`). The acquisition is maximised over
 * the grid alone (no local refinement), so the run is deterministic given its stream, and a point may be chosen twice.
 * All randomness (the initial design, forests, random interleaving, Thompson draws, max-value samples, and the
 * objective's own noise) comes from the runner's streams. A state whose value is not finite is flagged `diverged`.
 *
 * The objective is a function with a box, or a registered objective by name (`NamedObjective`), whose box is the
 * default; with a name the box may be left out altogether, the options then being the second argument. Throws
 * `DomainError` for a box whose upper corner is not above its lower one, an unknown name, `'fit'` noise without
 * `refit`, a negative noise or margin, `logCost` when maximising, Thompson sampling with a forest, constraints with a
 * batch or with an acquisition other than `'ei'` and `'pi'`, and a batch size below 1; and `ShapeError` for points of
 * the wrong dimension or an evaluation without its constraints.
 *
 * @param objective The function to optimise, as `BayesianObjective` (a `TestFunction`'s `value` will do), or a
 *   registered objective by name, as `NamedObjective`.
 * @param box The search box, as `SearchBox` (a `TestFunction`, which carries `lo` and `hi`, will do); for a named
 *   objective, left out (or null) for the objective's own box, or replaced by the options.
 * @param options The surrogate, acquisition, initial design, grid, refitting, batch and constraints, as
 *   `BayesianOptimisationOptions`.
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
 *
 * @example A named objective, with a random-forest surrogate and a random point every third choice (SMAC)
 * const alg = bayesianOptimisationSteps('forrester', { surrogate: 'random-forest', interleaveRandom: 3, grid: 51 })
 * const tr = trace(alg, undefined, 6, { stream: stream(2), keep: 'all' })
 * print('chosen x:', tr.steps.map((s) => s.next.data[0].toFixed(2)), ' random:', tr.steps.map((s) => s.random))
 * print('best:', tr.final.best.toFixed(3), ' known minimum:', alg.minimum.toFixed(3))
 *
 * @example A batch of three points per step by the kriging believer, and their fantasised values
 * const alg = bayesianOptimisationSteps('forrester', { batch: { size: 3 }, grid: 51, refit: false })
 * const s = run(alg, undefined, 1, { stream: stream(0) })
 * print('evaluations:', s.evaluations, ' next batch:', s.batch, ' fantasies:', s.fantasies)
 *
 * @example A constraint that rules out the unconstrained minimum: minimise Forrester where sin(10x) is at most 0
 * const forr = (u) => (6 * u - 2) ** 2 * Math.sin(12 * u - 4)
 * const f = (x) => ({ value: forr(x.data[0]), constraints: [Math.sin(10 * x.data[0])] })
 * const alg = bayesianOptimisationSteps(f, { lo: [0], hi: [1] }, { constraints: [{ threshold: 0 }], grid: 51 })
 * const s = run(alg, undefined, 8, { stream: stream(0) })
 * print('feasible best:', s.feasibleBest.toFixed(3), ' at x =', s.x.data[s.feasibleBestIndex])
 * print('best, feasible or not:', s.best.toFixed(3), ' at x =', s.bestX.data[0])
 * print('next x:', s.next, ' P(feasible) there:', s.feasibility.data[s.nextIndex].toFixed(3))
 */
export function bayesianOptimisationSteps(
  objective: NamedObjective,
  options?: BayesianOptimisationOptions,
): BayesianOptimisation
export function bayesianOptimisationSteps(
  objective: BayesianObjective | NamedObjective,
  box: SearchBox | null | undefined,
  options?: BayesianOptimisationOptions,
): BayesianOptimisation
export function bayesianOptimisationSteps(
  objective: BayesianObjective | NamedObjective,
  box?: SearchBox | BayesianOptimisationOptions | null,
  options?: BayesianOptimisationOptions,
): BayesianOptimisation {
  const where = 'bayesianOptimisationSteps'
  const named = isNamedObjective(objective) ? resolveObjective(objective, where) : null
  const isBox = (b: unknown): b is SearchBox => typeof b === 'object' && b !== null && 'lo' in b && 'hi' in b
  let searchBox: SearchBox | null | undefined
  let opts: BayesianOptimisationOptions
  if (named && box && !isBox(box) && options === undefined) {
    searchBox = undefined
    opts = box
  } else {
    searchBox = box as SearchBox | null | undefined
    opts = options ?? {}
  }
  if (!searchBox) {
    if (!named) throw new DomainError(where, `${where}: a function objective needs a box`)
    searchBox = named.fn
  }
  const theBox = readBox(searchBox, where)
  const { d } = theBox
  const {
    surrogate = 'gp',
    forest = {},
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
    logCost = false,
    interleaveRandom = 0,
    batch,
    constraints,
  } = opts
  if (noise === 'fit' && !refit) throw new DomainError(where, `${where}: noise 'fit' needs refit`)
  if (typeof noise === 'number' && !(noise >= 0)) throw new DomainError(where, `${where}: noise must be ≥ 0`)
  if (!(xi >= 0)) throw new DomainError(where, `${where}: xi must be ≥ 0`)
  if (logCost && !minimise) throw new DomainError(where, `${where}: logCost needs minimise`)
  if (surrogate !== 'gp' && surrogate !== 'random-forest')
    throw new DomainError(where, `${where}: unknown surrogate '${surrogate as string}'`)
  const q = batch?.size ?? 1
  const strategy: BatchStrategy = batch?.strategy ?? 'kriging-believer'
  if (!(Number.isInteger(q) && q >= 1)) throw new DomainError(where, `${where}: the batch size must be at least 1`)
  if (!(Number.isInteger(interleaveRandom) && interleaveRandom >= 0))
    throw new DomainError(where, `${where}: interleaveRandom must be a whole number`)
  if (surrogate === 'random-forest' && (acquisition === 'thompson' || (q > 1 && strategy === 'thompson')))
    throw new DomainError(where, `${where}: Thompson sampling needs the Gaussian-process surrogate`)
  const K = constraints?.length ?? 0
  if (K > 0 && batch) throw new DomainError(where, `${where}: constraints cannot be combined with a batch`)
  if (K > 0 && acquisition !== 'ei' && acquisition !== 'pi')
    throw new DomainError(where, `${where}: constraints need the 'ei' or 'pi' acquisition`)
  const thresholds = Float64Array.from(constraints ?? [], (c) => c.threshold)

  const cand = candidateGrid(opts.grid, theBox, where)
  const { m, unit: gridUnit, unitTensor: gridUnitT, grid } = cand
  const gridRow = (i: number) => fromData(Float64Array.from(toFlat(grid).slice(i * d, i * d + d)), [d])

  const evaluate = (x: Float64Array, s: Stream): { value: number; c: Float64Array } => {
    const point = fromData(Float64Array.from(x), [d])
    const out = named ? named.value(point, s) : (objective as BayesianObjective)(point, s)
    const value = typeof out === 'number' ? out : out.value
    if (logCost && !(value > 0)) throw new DomainError(where, `${where}: logCost needs positive values, got ${value}`)
    if (K === 0) return { value, c: new Float64Array(0) }
    const given = typeof out === 'number' ? undefined : out.constraints
    const c = Float64Array.from(given === undefined ? [] : isTensor(given) ? toFlat(given) : given)
    if (c.length !== K) throw new ShapeError(where, `${where}: the objective must return ${K} constraint values`)
    return { value, c }
  }

  /** Fit the surrogate to points on the unit cube and standardised targets, with the kernel fixed when given. */
  const fitModel = (X: Float64Array, Z: Float64Array, s: Stream, fixed?: { kernel: Kernel; noise: number }): Model => {
    const n = Z.length
    const Xt = fromData(X, [n, d])
    const Zt = fromData(Z, [n])
    if (surrogate === 'random-forest') {
      const model = randomForestRegressor({ trees: 10, ...forest }).fit(dataset(Xt, Zt), { stream: child(s, 'forest') })
      const trees = Float64Array.from(toFlat(model.treePredictions(gridUnitT)))
      const B = trees.length / m
      const mu = new Float64Array(m)
      const v = new Float64Array(m)
      for (let b = 0; b < B; b++) for (let i = 0; i < m; i++) mu[i] += trees[b * m + i] / B
      for (let b = 0; b < B; b++) for (let i = 0; i < m; i++) v[i] += (trees[b * m + i] - mu[i]) ** 2 / B
      return { mu, v, trees, B, draw: null, kernel: null, hyper: {}, noiseVariance: NaN, logMarginal: NaN }
    }
    const fit = fixed
      ? {
          kernel: fixed.kernel,
          noiseVariance: fixed.noise,
          post: gpPosterior(fixed.kernel, Xt, Zt, { noiseVariance: fixed.noise }),
        }
      : gpSurrogate(kernel, Xt, Zt, { refit, noise })
    const p = fit.post.predict(gridUnitT)
    return {
      mu: Float64Array.from(toFlat(p.mean)),
      v: Float64Array.from(toFlat(p.variance), (w) => Math.max(w, 0)),
      trees: null,
      B: 0,
      draw: (ds, count) => Float64Array.from(toFlat(fit.post.sample(ds, gridUnitT, count).draws)),
      kernel: fit.kernel,
      hyper: fit.kernel.params,
      noiseVariance: fit.noiseVariance,
      logMarginal: fit.post.logMarginal.value,
    }
  }

  /** The acquisition on the grid of a surrogate's mean and sd (target units) against an incumbent (target units). */
  const scoreOf = (
    mean: Tensor,
    sd: Tensor,
    model: Model,
    shift: number,
    scale: number,
    incumbent: number,
    t: number,
    s: Stream,
  ): { score: Float64Array; draw: Tensor | null; samples: Tensor | null } => {
    if (acquisition === 'ei') {
      const score = logCost
        ? expectedImprovementLogNormal(mean, sd, Math.exp(incumbent))
        : expectedImprovement(mean, sd, incumbent, { xi, minimise })
      return { score: Float64Array.from(toFlat(score)), draw: null, samples: null }
    }
    if (acquisition === 'pi')
      return {
        score: Float64Array.from(toFlat(probabilityOfImprovement(mean, sd, incumbent, { xi, minimise }))),
        draw: null,
        samples: null,
      }
    if (acquisition === 'ucb') {
      const b = beta === 'srinivas' ? ucbBeta(t + 1, { candidates: m, delta }) : beta
      return {
        score: Float64Array.from(toFlat(upperConfidenceBound(mean, sd, { beta: b, minimise }))),
        draw: null,
        samples: null,
      }
    }
    if (acquisition === 'thompson') {
      const z = model.draw!(child(s, 'draw'), 1)
      const draw = fromData(
        Float64Array.from(z, (w) => shift + scale * w),
        [m],
      )
      return { score: Float64Array.from(toFlat(draw), (w) => (minimise ? -w : w)), draw, samples: null }
    }
    if (acquisition === 'mes') {
      const { samples } = maxValueSamples(mean, sd, {
        count: maxValues,
        stream: child(s, 'max-values'),
        best: incumbent,
        minimise,
      })
      return {
        score: Float64Array.from(toFlat(maxValueEntropySearch(mean, sd, { samples, minimise }))),
        draw: null,
        samples,
      }
    }
    throw new DomainError(where, `${where}: unknown acquisition '${acquisition as string}'`)
  }

  /** The surrogate's mean and sd on the grid in target units. */
  const momentsOf = (model: Model, shift: number, scale: number) => ({
    mean: fromData(
      Float64Array.from(model.mu, (z) => shift + scale * z),
      [m],
    ),
    sd: fromData(
      Float64Array.from(model.v, (w) => scale * Math.sqrt(w)),
      [m],
    ),
  })

  /** Fit the surrogate (and the constraints' processes) to the evaluations, score the grid, and choose. */
  const choose = (x: Float64Array, y: Float64Array, c: Float64Array, t: number, s: Stream): Chosen => {
    const n = y.length
    const bestIndex = bestIndexOf(y, minimise)
    const best = y[bestIndex]
    const bestX = fromData(x.slice(bestIndex * d, bestIndex * d + d), [d])
    const target = logCost ? y.map(Math.log) : y
    const { shift, scale, z: Z } = standardise(target)
    const X = theBox.toUnit(x)
    const model = fitModel(X, Z, s)
    const { mean, sd } = momentsOf(model, shift, scale)
    let incumbent = logCost ? Math.log(best) : best

    // Constraints: one process each, the probability of feasibility, and the best feasible evaluation.
    let constraintMean: Tensor | null = null
    let constraintSd: Tensor | null = null
    let feasibility: Float64Array | null = null
    let feasibleBestIndex = -1
    if (K > 0) {
      const cm = new Float64Array(K * m)
      const cs = new Float64Array(K * m)
      feasibility = new Float64Array(m).fill(1)
      for (let k = 0; k < K; k++) {
        const ck = Float64Array.from({ length: n }, (_, i) => c[i * K + k])
        const st = standardise(ck)
        const fit = gpSurrogate(kernel, fromData(X, [n, d]), fromData(st.z, [n]), { refit, noise })
        const p = fit.post.predict(gridUnitT)
        const mu = toFlat(p.mean)
        const v = toFlat(p.variance)
        for (let i = 0; i < m; i++) {
          const mk = st.shift + st.scale * mu[i]
          const sk = st.scale * Math.sqrt(Math.max(v[i], 0))
          cm[k * m + i] = mk
          cs[k * m + i] = sk
          feasibility[i] *= sk > 0 ? normalCdf((thresholds[k] - mk) / sk) : mk <= thresholds[k] ? 1 : 0
        }
      }
      constraintMean = fromData(cm, [K, m])
      constraintSd = fromData(cs, [K, m])
      for (let i = 0; i < n; i++) {
        let ok = true
        for (let k = 0; k < K; k++) if (!(c[i * K + k] <= thresholds[k])) ok = false
        if (ok && (feasibleBestIndex < 0 || (minimise ? y[i] < y[feasibleBestIndex] : y[i] > y[feasibleBestIndex])))
          feasibleBestIndex = i
      }
      if (feasibleBestIndex >= 0) incumbent = logCost ? Math.log(y[feasibleBestIndex]) : y[feasibleBestIndex]
    }

    const first = scoreOf(mean, sd, model, shift, scale, incumbent, t, s)
    let score = first.score
    if (feasibility) {
      const pf = feasibility
      score = feasibleBestIndex >= 0 ? score.map((a, i) => a * pf[i]) : Float64Array.from(pf)
    }

    const random = interleaveRandom > 0 && (t + 1) % interleaveRandom === 0
    const indices: number[] = []
    let fantasies: Float64Array | null = null
    const rows: Float64Array[] = [score]
    if (random) {
      const r = child(s, 'random')
      for (let j = 0; j < q; j++) indices.push(integers(r, m))
      for (let j = 1; j < q; j++) rows.push(score)
    } else if (q === 1) indices.push(argmax(score))
    else if (strategy === 'thompson') {
      rows.length = 0
      const draws = model.draw!(child(s, 'batch-draws'), q)
      for (let j = 0; j < q; j++) {
        const row = Float64Array.from(
          draws.subarray(j * m, j * m + m),
          (w) => (minimise ? -1 : 1) * (shift + scale * w),
        )
        rows.push(row)
        indices.push(argmax(row))
      }
    } else if (strategy === 'local-penalisation') {
      const L = gridLipschitz(toFlat(mean), gridUnit, d)
      // M: the best of the incumbent and the mean on the grid, so no pending point's ball has a negative radius.
      let M = incumbent
      for (const v of toFlat(mean)) M = minimise ? Math.min(M, v) : Math.max(M, v)
      for (let j = 0; j < q; j++) {
        const pending = fromData(
          Float64Array.from(indices.flatMap((i) => Array.from(gridUnit.subarray(i * d, i * d + d)))),
          [indices.length, d],
        )
        const row =
          j === 0
            ? score
            : Float64Array.from(
                toFlat(
                  localPenalisation(fromData(score, [m]), gridUnitT, pending, {
                    mean,
                    sd,
                    lipschitz: L,
                    minimum: M,
                    minimise,
                  }),
                ),
              )
        if (j > 0) rows.push(row)
        indices.push(argmax(row))
      }
    } else {
      // Kriging believer and constant liars: refit with each chosen point at its fantasised value.
      fantasies = new Float64Array(q)
      let lie = 0
      if (strategy === 'constant-liar-mean') lie = Z.reduce((a, b) => a + b, 0) / n
      else if (strategy === 'constant-liar-min') lie = Math.min(...Z)
      else if (strategy === 'constant-liar-max') lie = Math.max(...Z)
      else if (strategy !== 'kriging-believer')
        throw new DomainError(where, `${where}: unknown batch strategy '${strategy as string}'`)
      let Xa = X
      let Za = Z
      let current = model
      let inc = incumbent
      for (let j = 0; j < q; j++) {
        let row = score
        if (j > 0) {
          const mo = momentsOf(current, shift, scale)
          row = scoreOf(mo.mean, mo.sd, current, shift, scale, inc, t, child(s, 'batch', j)).score
          rows.push(row)
        }
        const idx = argmax(row)
        indices.push(idx)
        const zf = strategy === 'kriging-believer' ? current.mu[idx] : lie
        fantasies[j] = shift + scale * zf
        if (j === q - 1) break
        const Xn = new Float64Array(Xa.length + d)
        Xn.set(Xa)
        Xn.set(gridUnit.subarray(idx * d, idx * d + d), Xa.length)
        const Zn = new Float64Array(Za.length + 1)
        Zn.set(Za)
        Zn[Za.length] = zf
        Xa = Xn
        Za = Zn
        current = fitModel(
          Xa,
          Za,
          child(s, 'fantasy', j),
          current.kernel ? { kernel: current.kernel, noise: current.noiseVariance } : undefined,
        )
        inc = minimise ? Math.min(inc, fantasies[j]) : Math.max(inc, fantasies[j])
      }
    }

    const nextIndex = indices[0]
    const pointsOf = (ix: number[]) =>
      fromData(Float64Array.from(ix.flatMap((i) => Array.from(toFlat(gridRow(i))))), [ix.length, d])
    const batchRows = new Float64Array(q * m)
    if (q > 1) rows.forEach((r, j) => batchRows.set(r, j * m))
    return {
      mean,
      sd,
      acquisition: fromData(rows[0], [m]),
      draw: first.draw,
      maxValues: first.samples,
      next: gridRow(nextIndex),
      nextIndex,
      random,
      treePredictions: model.trees
        ? fromData(
            Float64Array.from(model.trees, (w) => shift + scale * w),
            [model.B, m],
          )
        : null,
      batch: q > 1 ? pointsOf(indices) : null,
      batchIndices: q > 1 ? indices : null,
      fantasies: q > 1 && fantasies ? fromData(fantasies, [q]) : null,
      batchAcquisition: q > 1 ? fromData(batchRows, [q, m]) : null,
      constraintValues: K > 0 ? fromData(Float64Array.from(c), [n, K]) : null,
      constraintMean,
      constraintSd,
      feasibility: feasibility ? fromData(feasibility, [m]) : null,
      feasibleBest: feasibleBestIndex >= 0 ? y[feasibleBestIndex] : NaN,
      feasibleBestIndex,
      best,
      bestX,
      bestIndex,
      hyper: model.hyper,
      noiseVariance: model.noiseVariance,
      logMarginal: model.logMarginal,
      shift,
      scale,
    }
  }

  const finite = (y: Float64Array, c: Float64Array): boolean => y.every(Number.isFinite) && c.every(Number.isFinite)

  const state = (
    t: number,
    x: Float64Array,
    y: Float64Array,
    c: Float64Array,
    s: Stream,
  ): BayesianOptimisationState => {
    const n = y.length
    const base = { t, evaluations: n, x: fromData(x, [n, d]), y: fromData(y, [n]) }
    if (!finite(y, c)) return { ...base, ...placeholder(m, d, K, n, c), diverged: true }
    return { ...base, ...choose(x, y, c, t, s) }
  }

  return {
    name: 'bayesian-optimisation',
    grid,
    shape: cand.shape,
    kernel,
    box: { lo: fromData(Float64Array.from(theBox.lo), [d]), hi: fromData(Float64Array.from(theBox.hi), [d]) },
    objective: named ? named.key : null,
    minimum: named ? named.fn.minimumValue : NaN,
    init: (_start, stream) => {
      const x = designPoints(init, design, theBox, child(stream, 'design'), where)
      const n = x.length / d
      const y = new Float64Array(n)
      const c = new Float64Array(n * K)
      for (let i = 0; i < n; i++) {
        const e = evaluate(x.subarray(i * d, i * d + d), child(stream, 'evaluate', i))
        y[i] = e.value
        c.set(e.c, i * K)
      }
      return state(0, x, y, c, child(stream, 'acquisition'))
    },
    step: (prev, ctx) => {
      const points = prev.batch ? toFlat(prev.batch) : toFlat(prev.next)
      const p = points.length / d
      const n = prev.evaluations + p
      const x = new Float64Array(n * d)
      x.set(toFlat(prev.x))
      x.set(points, prev.evaluations * d)
      const y = new Float64Array(n)
      y.set(toFlat(prev.y))
      const c = new Float64Array(n * K)
      if (prev.constraintValues) c.set(toFlat(prev.constraintValues))
      for (let j = 0; j < p; j++) {
        const i = prev.evaluations + j
        const s = p === 1 ? child(ctx.stream, 'evaluate') : child(ctx.stream, 'evaluate', j)
        const e = evaluate(x.subarray(i * d, i * d + d), s)
        y[i] = e.value
        c.set(e.c, i * K)
      }
      return state(prev.t + 1, x, y, c, child(ctx.stream, 'acquisition'))
    },
  }
}

/** The fields of a state chosen by the surrogate (all but the evaluations). */
type Chosen = Omit<BayesianOptimisationState, 't' | 'evaluations' | 'x' | 'y'>

/**
 * The fields of a state that has no surrogate, because an evaluation was not finite: NaN everywhere.
 *
 * @param m The grid's size.
 * @param d The dimension.
 * @param K The number of constraints.
 * @param n The number of evaluations.
 * @param c The constraint values observed, $n \times K$ row-major.
 * @returns The surrogate, acquisition and incumbent fields, all NaN.
 */
function placeholder(m: number, d: number, K: number, n: number, c: Float64Array): Chosen {
  const nan = (k: number) => fromData(new Float64Array(k).fill(NaN), [k])
  return {
    mean: nan(m),
    sd: nan(m),
    acquisition: nan(m),
    draw: null,
    maxValues: null,
    next: nan(d),
    nextIndex: -1,
    random: false,
    treePredictions: null,
    batch: null,
    batchIndices: null,
    fantasies: null,
    batchAcquisition: null,
    constraintValues: K > 0 ? fromData(Float64Array.from(c), [n, K]) : null,
    constraintMean: null,
    constraintSd: null,
    feasibility: null,
    feasibleBest: NaN,
    feasibleBestIndex: -1,
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
