/**
 * Multi-fidelity Bayesian optimisation as a step-through algorithm: each step chooses a point and a fidelity to
 * evaluate it at, weighing what an evaluation tells about the highest fidelity against what it costs.
 *
 * The surrogate is the autoregressive multi-fidelity GP of Kennedy and O'Hagan (2000), `fitMultiFidelityGp` of
 * `aifn-methods/learning/gaussian-processes`, $f_t = \rho_{t-1} f_{t-1} + \delta_t$ over fidelities
 * $t = 0, \dots, M - 1$ with costs $\lambda_t$, on inputs scaled to the unit cube and targets divided by the standard
 * deviation of every observation (not centred, since one shift of every fidelity changes the model). The objective is minimised at the highest fidelity. Three acquisitions
 * are scored on a grid, at every fidelity:
 *
 * - `'mf-gp-ucb'` (Kandasamy, Dasarathy, Oliva, Schneider and Póczos, 2016), in its minimising form: each fidelity
 *   bounds $f_{M-1}$ from below by $L_t(\xvec) = \mu_t(\xvec) - \sqrt{\beta}\,\sigma_t(\xvec) - \zeta_t$, $\zeta_t$
 *   bounding $\lvert f_{M-1} - f_t \rvert$; the point maximises $-\max_t L_t(\xvec)$, and the fidelity is the lowest
 *   $t$ whose uncertainty there is still large, $\sqrt{\beta}\,\sigma_t(\xvec) \ge \gamma_t$ (else the highest).
 * - `'mf-mes'` (Takeno et al., 2020): the information an evaluation of $f_t$ gives about the optimal value $y^*$ of
 *   $f_{M-1}$, per unit cost. With samples $y^*_k$ (as `maxValueSamples`, Gumbel), the correlation $r$ of $f_t(\xvec)$
 *   and $f_{M-1}(\xvec)$ and, in the minimising form, $\gamma_k = (\mu_{M-1}(\xvec) - y^*_k) / \sigma_{M-1}(\xvec)$,
 *
 * $$ \alpha_t(\xvec) = \frac{1}{\lambda_t K} \sum_{k=1}^{K} \left[ \frac{r^2 \gamma_k \phi(\gamma_k)}{2\Phi(\gamma_k)}
 * - \log \Phi(\gamma_k) + \expect\left[ \log \Phi(\eta_k) \mid f_{M-1}(\xvec) \ge y^*_k \right] \right], $$
 *
 *   where $u \sim \Gauss(0, 1)$ is the standardised value of $-f_t(\xvec)$, $\eta_k = (\gamma_k - r u) /
 *   \sqrt{1 - r^2}$, and the expectation over $u$ is computed by the trapezoidal rule. At the highest fidelity
 *   ($r = 1$) it is max-value entropy search (`maxValueEntropySearch`) divided by the cost.
 * - `'ei-per-cost'`: the expected improvement of $f_{M-1}$ over the best value observed at the highest fidelity,
 *   discounted by $\lvert r \rvert$ and divided by the cost, $\mathrm{EI}_{M-1}(\xvec)\,\lvert r_t(\xvec) \rvert /
 *   \lambda_t$ (the correlation factor of Huang et al.'s 2006 augmented EI, and the cost of Swersky, Snoek and Adams,
 *   2013).
 *
 * For the last two, the chosen point and fidelity maximise the score over the grid and every fidelity.
 */

import type { Status, Vector } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, permutation, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { matern52, type Kernel } from 'aifn-compute/learning/kernels'
import { normalCdf, normalLogCdf, normalLogPdf, normalPdf } from 'aifn-compute/numerics/special'
import { fitMultiFidelityGp } from 'aifn-methods/learning/gaussian-processes'
import { expectedImprovement } from './acquisitions'
import { maxValueEntropySearch, maxValueSamples } from './entropy'
import type { SearchBox } from './common'

/** The multi-fidelity acquisitions `multiFidelityBayesianOptimisationSteps` can maximise. */
export type MultiFidelityAcquisitionName = 'mf-gp-ucb' | 'mf-mes' | 'ei-per-cost'

/**
 * The objective: its value at a point of the box ($d$ values) at a fidelity (one of `levels`), given a stream of its
 * own for any noise. An `Evaluation` (`{ value }`) is accepted too.
 */
export type MultiFidelityObjective = (x: Vector, fidelity: number, stream: Stream) => number | { value: number }

/** The fidelities: what the objective is called with at each, and what an evaluation there costs. */
export type Fidelities = {
  /** The value passed to the objective at each fidelity, lowest first (default $0, 1, \dots, M - 1$). */
  levels?: readonly number[]
  /** The cost $\lambda_t > 0$ of an evaluation at each fidelity, $M$ values; the last is the target fidelity. */
  cost: readonly number[]
}

/** Options of `multiFidelityBayesianOptimisationSteps`. */
export interface MultiFidelityBayesianOptimisationOptions {
  /** The acquisition (default `'mf-gp-ucb'`). */
  acquisition?: MultiFidelityAcquisitionName
  /**
   * The kernels on the unit cube, one per fidelity ($k_0$, then each discrepancy's), and the start of every refit
   * (default `matern52` with lengthscale 0.2 and variance 1 for each).
   */
  kernels?: readonly Kernel[]
  /**
   * The initial design: the number of points at each fidelity (default $(2d + 1) 2^{M - 1 - t}$ at fidelity $t$), or
   * one number for all. The points are nested: fidelity $t$ gets the first $n_t$ points of one Latin hypercube.
   */
  init?: number | readonly number[]
  /**
   * The candidates: a number of points per coordinate (default 201 in one dimension, 41 in two, and
   * $\lfloor 4096^{1/d} \rfloor$ beyond), or one count per coordinate, for a regular grid that includes the corners.
   */
  grid?: number | readonly number[]
  /** Refit the kernels by marginal likelihood each step, once every fidelity has three evaluations (default true). */
  refit?: boolean
  /** The noise variance of the scaled targets at every fidelity (default $10^{-6}$). */
  noise?: number
  /** The exploration weight $\beta$ of `'mf-gp-ucb'` (default 4). */
  beta?: number
  /**
   * The bounds $\zeta_t$ on $\lvert f_{M-1} - f_t \rvert$ of `'mf-gp-ucb'`, $M$ values in the objective's units (the
   * last 0). By default each is estimated as the largest gap between the posterior means of fidelity $t$ and of the
   * highest on the grid.
   */
  zeta?: readonly number[]
  /**
   * The thresholds $\gamma_t$ of `'mf-gp-ucb'`, $M - 1$ values in the objective's units (default 0.1 times the
   * standard deviation of the observations).
   */
  gamma?: readonly number[]
  /** The number of samples of the optimal value for `'mf-mes'` (default 10). */
  maxValues?: number
}

/** One state of `multiFidelityBayesianOptimisationSteps`. */
export interface MultiFidelityBayesianOptimisationState extends Status {
  /** Steps taken: evaluations after the initial design. */
  t: number
  /** Evaluations so far, $n$, the initial design included. */
  evaluations: number
  /** The points evaluated, $n \times d$, in the box's coordinates and in order. */
  x: Tensor
  /** The values observed, `[n]`. */
  y: Tensor
  /** The fidelity of each evaluation, an index from 0 to $M - 1$. */
  fidelity: number[]
  /** The cost spent so far, the sum of the evaluations' costs. */
  cost: number
  /** The posterior mean of each fidelity on the grid, $M \times m$, in the objective's units. */
  mean: Tensor
  /** The posterior standard deviation of each fidelity on the grid, $M \times m$. */
  sd: Tensor
  /** The acquisition of an evaluation at each fidelity and grid point, $M \times m$ (see the file comment). */
  acquisition: Tensor
  /** The score of each grid point, `[m]`: the minimum over fidelities for `'mf-gp-ucb'`, else the maximum. */
  combined: Tensor
  /** For `'mf-mes'`: the samples of the optimal value, `[K]` (null otherwise). */
  maxValues: Tensor | null
  /** The scales $\rho_0, \dots, \rho_{M-2}$ of the surrogate. */
  rho: number[]
  /** The next point to evaluate, $d$ values. */
  next: Tensor
  /** Its index in the grid. */
  nextIndex: number
  /** The fidelity to evaluate it at, an index from 0 to $M - 1$. */
  nextFidelity: number
  /** The incumbent: the smallest value observed at the highest fidelity (NaN before there is one). */
  best: number
  /** Where it was observed, $d$ values (NaN before there is one). */
  bestX: Tensor
  /** Its index among the evaluations ($-1$ before there is one). */
  bestIndex: number
  /**
   * The standard deviation (about their mean) of the observations, which the targets were divided by (1 when it is
   * 0). They are not centred: the autoregressive model is not invariant to one shift of every fidelity.
   */
  scale: number
}

/** The algorithm, with the grid its states refer to. */
export interface MultiFidelityBayesianOptimisation extends Algorithm<void, MultiFidelityBayesianOptimisationState> {
  /** The candidates, $m \times d$ in the box's coordinates. */
  readonly grid: Tensor
  /** The grid's counts per coordinate, the last coordinate varying fastest. */
  readonly shape: readonly number[]
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

/** The nodes of the trapezoidal rule for the expectation over $u \sim \mathcal{N}(0, 1)$ in `'mf-mes'`. */
const NODES = Array.from({ length: 321 }, (_, k) => -8 + k * 0.05)
/** The trapezoidal weights times the standard normal density at `NODES`. */
const WEIGHTS = NODES.map((u, k) => (k === 0 || k === NODES.length - 1 ? 0.025 : 0.05) * (normalPdf(u) as number))

/**
 * The information about the optimal value that an evaluation correlated $r$ with the target gives, for one sample
 * (minimising; see the file comment).
 *
 * @param gamma $\gamma = (\mu_{M-1} - y^*) / \sigma_{M-1}$.
 * @param r The correlation of the evaluated fidelity with the target fidelity at the point.
 * @returns The mutual information, never negative.
 */
function multiFidelityInformation(gamma: number, r: number): number {
  const logPhiG = normalLogCdf(gamma) as number
  const phiG = Math.exp(logPhiG)
  const ratio = Math.exp((normalLogPdf(gamma) as number) - logPhiG)
  const r2 = Math.min(1, r * r)
  let tail = 0
  if (r2 < 1 - 1e-12) {
    const w = Math.sqrt(1 - r2)
    let total = 0
    for (let k = 0; k < NODES.length; k++) {
      const eta = (gamma - r * NODES[k]) / w
      const p = normalCdf(eta) as number
      if (p > 0) total += WEIGHTS[k] * p * Math.log(p)
    }
    tail = total / phiG
  }
  return Math.max(0, (r2 * gamma * ratio) / 2 - logPhiG + tail)
}

/**
 * Multi-fidelity Bayesian optimisation of `objective` over `box` as a step-through algorithm (see the file comment):
 * `init` evaluates a nested initial design at every fidelity and fits the surrogate; each step evaluates the point and
 * fidelity the last state chose, refits, and chooses again. Every state holds the evaluations with their fidelities,
 * the cumulative cost, each fidelity's posterior mean and standard deviation and acquisition on the grid, the chosen
 * point and fidelity, and the incumbent at the highest fidelity, so a figure can replay the run. All randomness (the
 * design, the max-value samples and the objective's noise) comes from the runner's streams. A state whose value is not
 * finite is flagged `diverged`. Throws `DomainError` for a box whose upper corner is not above its lower one, costs
 * that are not positive, counts of the wrong length, or an unknown acquisition.
 *
 * @param objective The function to minimise, at a point and a fidelity, as `MultiFidelityObjective`.
 * @param box The search box, as `SearchBox`.
 * @param fidelities The fidelities' values and costs, as `Fidelities`.
 * @param options The acquisition, surrogate, design and grid, as `MultiFidelityBayesianOptimisationOptions`.
 * @returns The algorithm, to run with `run` or `trace` from no start (`undefined`), with its `grid`.
 *
 * @example Forrester's function with a low fidelity a quarter of the price: MF-GP-UCB picks the fidelity per step
 * import { forrester } from 'aifn-methods/data/objectives'
 * import { matern52 } from 'aifn-compute/learning/kernels'
 * const [hi, lo] = [forrester(), forrester({ fidelity: 'low' })]
 * const f = (x, z) => (z === 0 ? lo : hi).value([x.data[0]])
 * const kernels = [matern52({ lengthscale: 0.15 }), matern52({ lengthscale: 1, variance: 0.1 })]
 * const options = { acquisition: 'mf-gp-ucb', grid: 51, refit: false, kernels, init: [8, 4] }
 * const tr = trace(multiFidelityBayesianOptimisationSteps(f, hi, { cost: [1, 4] }, options), undefined, 6, {
 *   stream: stream(0),
 *   keep: 'all',
 * })
 * print('x chosen:', tr.steps.map((s) => s.next.data[0].toFixed(2)))
 * print('fidelity:', tr.steps.map((s) => s.nextFidelity), ' cost:', tr.steps.map((s) => s.cost))
 * print('incumbent at full fidelity:', tr.steps[0].best.toFixed(3), '->', tr.final.best.toFixed(3))
 * print('rho =', tr.final.rho[0].toFixed(3), ' (the truth is 2)')
 */
export function multiFidelityBayesianOptimisationSteps(
  objective: MultiFidelityObjective,
  box: SearchBox,
  fidelities: Fidelities,
  options: MultiFidelityBayesianOptimisationOptions = {},
): MultiFidelityBayesianOptimisation {
  const where = 'multiFidelityBayesianOptimisationSteps'
  const lo = Float64Array.from(isTensor(box.lo) ? toFlat(box.lo) : box.lo)
  const hi = Float64Array.from(isTensor(box.hi) ? toFlat(box.hi) : box.hi)
  const d = lo.length
  if (d === 0 || hi.length !== d)
    throw new ShapeError(where, `${where}: the box's corners must have the same, positive length`)
  for (let j = 0; j < d; j++)
    if (!(hi[j] > lo[j])) throw new DomainError(where, `${where}: the box's upper corner must be above its lower one`)
  const costs = [...fidelities.cost]
  const M = costs.length
  if (M === 0 || !costs.every((c) => c > 0)) throw new DomainError(where, `${where}: the costs must be positive`)
  const levels = fidelities.levels ? [...fidelities.levels] : costs.map((_, t) => t)
  if (levels.length !== M) throw new DomainError(where, `${where}: ${levels.length} levels for ${M} costs`)
  const {
    acquisition = 'mf-gp-ucb',
    kernels = costs.map(() => matern52({ lengthscale: 0.2, variance: 1 })),
    refit = true,
    noise = 1e-6,
    beta = 4,
    maxValues = 10,
  } = options
  if (!['mf-gp-ucb', 'mf-mes', 'ei-per-cost'].includes(acquisition))
    throw new DomainError(where, `${where}: unknown acquisition '${acquisition}'`)
  if (kernels.length !== M) throw new DomainError(where, `${where}: ${kernels.length} kernels for ${M} fidelities`)
  const counts =
    options.init === undefined
      ? costs.map((_, t) => (2 * d + 1) * 2 ** (M - 1 - t))
      : typeof options.init === 'number'
        ? costs.map(() => options.init as number)
        : [...options.init]
  if (counts.length !== M || !counts.every((c) => Number.isInteger(c) && c >= 1))
    throw new DomainError(where, `${where}: init needs a positive integer count for each of the ${M} fidelities`)
  if (options.zeta && options.zeta.length !== M) throw new DomainError(where, `${where}: zeta needs ${M} values`)
  if (options.gamma && options.gamma.length !== M - 1)
    throw new DomainError(where, `${where}: gamma needs ${M - 1} values`)
  if (!(beta >= 0)) throw new DomainError(where, `${where}: beta must be non-negative`)

  const toUnit = (v: ArrayLike<number>) => Float64Array.from(v, (x, i) => (x - lo[i % d]) / (hi[i % d] - lo[i % d]))
  const fromUnit = (v: ArrayLike<number>) => Float64Array.from(v, (u, i) => lo[i % d] + u * (hi[i % d] - lo[i % d]))
  const fallback = d === 1 ? 201 : d === 2 ? 41 : Math.max(2, Math.floor(4096 ** (1 / d)))
  const gridShape = Array.isArray(options.grid)
    ? [...(options.grid as readonly number[])]
    : Array.from({ length: d }, () => (options.grid as number | undefined) ?? fallback)
  if (gridShape.length !== d || gridShape.some((c) => !(Number.isInteger(c) && c >= 2)))
    throw new DomainError(where, `${where}: a grid needs an integer count of at least 2 per coordinate`)
  const gridUnit = unitGrid(gridShape)
  const m = gridUnit.length / d
  const gridUnitT = fromData(gridUnit, [m, d])
  const grid = fromData(fromUnit(gridUnit), [m, d])
  const gridFlat = toFlat(grid) as ArrayLike<number>

  const evaluate = (x: ArrayLike<number>, t: number, s: Stream): number => {
    const out = objective(fromData(Float64Array.from(x), [d]), levels[t], s)
    return typeof out === 'number' ? out : out.value
  }

  /** Fits the surrogate to the evaluations, scores the grid at every fidelity, and chooses the next evaluation. */
  const choose = (
    x: Float64Array,
    y: Float64Array,
    z: readonly number[],
    s: Stream,
  ): Omit<MultiFidelityBayesianOptimisationState, 't' | 'evaluations' | 'x' | 'y' | 'fidelity' | 'cost'> => {
    const n = y.length
    let bestIndex = -1
    for (let i = 0; i < n; i++) if (z[i] === M - 1 && (bestIndex < 0 || y[i] < y[bestIndex])) bestIndex = i
    const best = bestIndex < 0 ? NaN : y[bestIndex]
    const bestX = fromData(bestIndex < 0 ? new Float64Array(d).fill(NaN) : x.slice(bestIndex * d, bestIndex * d + d), [
      d,
    ])
    const average = y.reduce((a, v) => a + v, 0) / n
    const variance = y.reduce((a, v) => a + (v - average) ** 2, 0) / n
    const scale = variance > 0 ? Math.sqrt(variance) : 1
    const counted = Array.from({ length: M }, (_, t) => z.filter((v) => v === t).length)
    const gp = fitMultiFidelityGp(
      fromData(toUnit(x), [n, d]),
      [...z],
      fromData(
        y.map((v) => v / scale),
        [n],
      ),
      { kernels, noise, fitKernels: refit && counted.every((c) => c >= 3), maxSteps: 50, rounds: 3 },
    )
    const means: Float64Array[] = []
    const sds: Float64Array[] = []
    for (let t = 0; t < M; t++) {
      const p = gp.predict(gridUnitT, t)
      means.push(Float64Array.from(toFlat(p.mean) as ArrayLike<number>, (v) => scale * v))
      sds.push(Float64Array.from(toFlat(p.sd) as ArrayLike<number>, (v) => scale * v))
    }
    const top = M - 1
    // The correlation of f_t(x) and f_top(x) on the grid.
    const corr = (t: number): Float64Array => {
      if (t === top) return new Float64Array(m).fill(1)
      const c = toFlat(gp.covariance(gridUnitT, t, top)) as ArrayLike<number>
      return Float64Array.from(c, (v, i) => {
        const den = (sds[t][i] / scale) * (sds[top][i] / scale)
        return den > 0 ? Math.max(-1, Math.min(1, v / den)) : 0
      })
    }
    const scores = Array.from({ length: M }, () => new Float64Array(m))
    let samples: Tensor | null = null
    const meanTop = fromData(means[top], [m])
    const sdTop = fromData(sds[top], [m])
    if (acquisition === 'mf-gp-ucb') {
      const w = Math.sqrt(beta)
      for (let t = 0; t < M; t++) {
        let zeta = options.zeta ? options.zeta[t] : 0
        if (!options.zeta && t < top)
          for (let i = 0; i < m; i++) zeta = Math.max(zeta, Math.abs(means[t][i] - means[top][i]))
        for (let i = 0; i < m; i++) scores[t][i] = -(means[t][i] - w * sds[t][i] - zeta)
      }
    } else if (acquisition === 'mf-mes') {
      samples = maxValueSamples(meanTop, sdTop, {
        count: maxValues,
        stream: child(s, 'max-values'),
        best: bestIndex < 0 ? undefined : best,
      }).samples
      const ys = toFlat(samples) as ArrayLike<number>
      scores[top].set(toFlat(maxValueEntropySearch(meanTop, sdTop, { samples })) as ArrayLike<number>)
      for (let t = 0; t < top; t++) {
        const r = corr(t)
        for (let i = 0; i < m; i++) {
          const sTop = sds[top][i]
          if (!(sTop > 0) || !(sds[t][i] > 0)) continue
          let total = 0
          for (const ystar of Array.from(ys)) total += multiFidelityInformation((means[top][i] - ystar) / sTop, r[i])
          scores[t][i] = total / ys.length
        }
      }
      for (let t = 0; t < M; t++) for (let i = 0; i < m; i++) scores[t][i] /= costs[t]
    } else {
      const incumbent = bestIndex < 0 ? Math.min(...means[top]) : best
      const ei = toFlat(expectedImprovement(meanTop, sdTop, incumbent)) as ArrayLike<number>
      for (let t = 0; t < M; t++) {
        const r = corr(t)
        for (let i = 0; i < m; i++) scores[t][i] = (ei[i] * Math.abs(r[i])) / costs[t]
      }
    }
    const combined = new Float64Array(m)
    let nextIndex = 0
    let nextFidelity = top
    if (acquisition === 'mf-gp-ucb') {
      for (let i = 0; i < m; i++) combined[i] = Math.min(...scores.map((row) => row[i]))
      for (let i = 1; i < m; i++) if (combined[i] > combined[nextIndex]) nextIndex = i
      const w = Math.sqrt(beta)
      for (let t = 0; t < top; t++) {
        const gamma = options.gamma ? options.gamma[t] : 0.1 * scale
        if (w * sds[t][nextIndex] >= gamma) {
          nextFidelity = t
          break
        }
      }
    } else {
      for (let i = 0; i < m; i++) combined[i] = Math.max(...scores.map((row) => row[i]))
      let bestScore = -Infinity
      for (let t = 0; t < M; t++)
        for (let i = 0; i < m; i++)
          if (scores[t][i] > bestScore) {
            bestScore = scores[t][i]
            nextIndex = i
            nextFidelity = t
          }
    }
    const stack = (rows: Float64Array[]) => {
      const out = new Float64Array(M * m)
      rows.forEach((r, t) => out.set(r, t * m))
      return fromData(out, [M, m])
    }
    return {
      mean: stack(means),
      sd: stack(sds),
      acquisition: stack(scores),
      combined: fromData(combined, [m]),
      maxValues: samples,
      rho: gp.rho,
      next: fromData(
        Float64Array.from({ length: d }, (_, j) => gridFlat[nextIndex * d + j]),
        [d],
      ),
      nextIndex,
      nextFidelity,
      best,
      bestX,
      bestIndex,
      scale,
    }
  }

  const placeholder = (): Omit<
    MultiFidelityBayesianOptimisationState,
    't' | 'evaluations' | 'x' | 'y' | 'fidelity' | 'cost'
  > => {
    const nan = (shape: number[]) => fromData(new Float64Array(shape.reduce((a, b) => a * b, 1)).fill(NaN), shape)
    return {
      mean: nan([M, m]),
      sd: nan([M, m]),
      acquisition: nan([M, m]),
      combined: nan([m]),
      maxValues: null,
      rho: [],
      next: nan([d]),
      nextIndex: -1,
      nextFidelity: -1,
      best: NaN,
      bestX: nan([d]),
      bestIndex: -1,
      scale: NaN,
    }
  }

  return {
    name: 'multi-fidelity-bayesian-optimisation',
    grid,
    shape: gridShape,
    init: (_start, stream) => {
      // A Latin hypercube of max(n_t) points; fidelity t takes its first n_t.
      const size = Math.max(...counts)
      const u = new Float64Array(size * d)
      for (let j = 0; j < d; j++) {
        const order = toFlat(permutation(child(stream, 'slice', j), size))
        const jitter = toFlat(uniform(child(stream, 'jitter', j), 0, 1, { shape: [size] }) as Tensor)
        for (let i = 0; i < size; i++) u[i * d + j] = (order[i] + jitter[i]) / size
      }
      const design = fromUnit(u)
      const xs: number[] = []
      const ys: number[] = []
      const z: number[] = []
      let cost = 0
      for (let t = 0; t < M; t++)
        for (let i = 0; i < counts[t]; i++) {
          const point = design.subarray(i * d, i * d + d)
          xs.push(...point)
          ys.push(evaluate(point, t, child(stream, 'evaluate', t, i)))
          z.push(t)
          cost += costs[t]
        }
      const x = Float64Array.from(xs)
      const y = Float64Array.from(ys)
      const base = {
        t: 0,
        evaluations: y.length,
        x: fromData(x, [y.length, d]),
        y: fromData(y, [y.length]),
        fidelity: z,
        cost,
      }
      if (!y.every(Number.isFinite)) return { ...base, ...placeholder(), diverged: true }
      return { ...base, ...choose(x, y, z, child(stream, 'acquisition')) }
    },
    step: (state, ctx) => {
      const n = state.evaluations + 1
      const x = new Float64Array(n * d)
      x.set(toFlat(state.x) as ArrayLike<number>)
      x.set(toFlat(state.next) as ArrayLike<number>, (n - 1) * d)
      const y = new Float64Array(n)
      y.set(toFlat(state.y) as ArrayLike<number>)
      y[n - 1] = evaluate(x.subarray((n - 1) * d), state.nextFidelity, child(ctx.stream, 'evaluate'))
      const z = [...state.fidelity, state.nextFidelity]
      const base = {
        t: state.t + 1,
        evaluations: n,
        x: fromData(x, [n, d]),
        y: fromData(y, [n]),
        fidelity: z,
        cost: state.cost + costs[state.nextFidelity],
      }
      if (!y.every(Number.isFinite)) return { ...base, ...placeholder(), diverged: true }
      return { ...base, ...choose(x, y, z, child(ctx.stream, 'acquisition')) }
    },
  }
}
