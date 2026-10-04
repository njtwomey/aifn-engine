/**
 * Active learning with label proportions (Poyiadzis, Santos-Rodriguez and Twomey 2019, ICASSP): the learner holds a
 * few bags with known class proportions and a pool U of points in no bag; it builds a bag of k points from U and asks
 * an LLP-oracle, which answers with the bag's class proportion only (the true label when k = 1). The learner is LP-LLP
 * (`lpllpSteps`, Poyiadzi et al. 2018) in the papers' ±1 encoding: points in no bag start at the uninformative score
 * and are not constrained. Query strategies (§3.3 and §4):
 *
 * - **US-Mass**: the pool point closest to the decision boundary (smallest |f_i − ½| in [0, 1] scores), and the k − 1
 *   pool points closest to it under L_U = (I − αS)⁻¹ restricted to U, the matrix LP-LLP already computed: aims at a pure
 *   bag around the most uncertain point;
 * - **US-LP**: the k most uncertain pool points, answered with their proportion;
 * - **Random**: k random pool points, answered with their proportion;
 * - **US-Exact**: the k most uncertain points, answered with their k true labels (k singleton bags).
 *
 * Accuracy is measured on held-out test points, which sit in the graph (transductive) but are never queried.
 */

import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, permutation, stream as makeStream, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { trace } from 'aifn-compute/foundation/trace'
import { lpllpGraph, lpllpSteps, type LpLlpOptions } from './label-proportions'

/** The query strategies of the paper. */
export type ActiveStrategy = 'us-mass' | 'us-lp' | 'random' | 'us-exact'

/** All strategies, in the paper's order. */
export const ACTIVE_STRATEGIES: readonly ActiveStrategy[] = ['us-lp', 'us-mass', 'us-exact', 'random']

/** The setting of an active-learning run: the points, the oracle's labels, the starting bags, the pool and the test set. */
export interface ActiveProportionsProblem {
  readonly x: MatrixLike
  /** True labels 0/1, read only by the oracle and the accuracy. */
  readonly labels: ArrayLike<number>
  /** The starting bags (−1: in no bag) and their proportions [B, 2]. */
  readonly bags: ArrayLike<number>
  readonly proportions: MatrixLike
  /** Points that are never queried and on which accuracy is measured. */
  readonly test: ArrayLike<number>
}

/** Options of an active-learning run. */
export interface ActiveProportionsOptions extends LpLlpOptions {
  strategy?: ActiveStrategy
  /** Points per query k (default 10). */
  bagSize?: Size
  /** LP-LLP steps per refit (default 300). */
  maxSteps?: Size
  seed?: number | string
}

/** One state: the bags after t queries, LP-LLP's fit to them, and the next query's ranking. */
export interface ActiveProportionsState extends Status {
  /** Each point's bag (−1 in none) and the bags' proportions [B, 2]. */
  readonly bags: Int32Array
  readonly proportions: Float64Array
  /** LP-LLP's scores f_i = F_{i,1} ∈ [0, 1] and labels. */
  readonly scores: Float64Array
  readonly labels: Int32Array
  /** Accuracy on the test points. */
  readonly accuracy: number
  /** The points of the last query (empty at t = 0), its seed point for US-Mass (−1 otherwise) and the oracle's answer. */
  readonly query: readonly number[]
  readonly seed: number
  readonly answer: number
  /** Uncertainty |f_i − ½| of every point (Infinity outside the pool): what the next uncertainty query ranks. */
  readonly uncertainty: Float64Array
}

const binaryProportions = (P: MatrixLike) => {
  const m = dense.toMatrixF64(P, 'activeProportionsSteps')
  if (m.n !== 2)
    throw new DomainError('activeProportionsSteps', 'activeProportionsSteps: two classes only, as in the paper')
  return Float64Array.from(m.data)
}

/**
 * Active learning with an LLP-oracle as steps: step 0 fits LP-LLP to the starting bags; each step queries a bag of k
 * pool points by the strategy (module notes), adds the oracle's answer as a new bag (k singleton bags for US-Exact) and
 * refits. Stops when the pool is empty.
 */
export function activeProportionsSteps(
  problem: ActiveProportionsProblem,
  options: ActiveProportionsOptions = {},
): Algorithm<void, ActiveProportionsState> {
  const { strategy = 'us-lp', bagSize = 10, maxSteps = 300, seed = 0, ...lp } = options
  const y = Int32Array.from(problem.labels)
  const n = y.length
  const test = new Set(Array.from(problem.test))
  // L = (I − αS)⁻¹ up to the factor (1 − α), which does not change the ranking US-Mass reads from it.
  const L = dense.data(lpllpGraph(problem.x, lp).propagation)
  const root = makeStream(`activeProportions/${seed}`)
  const fit = (
    bags: Int32Array,
    proportions: Float64Array,
    t: number,
    query: number[],
    seedPoint: number,
    answer: number,
  ) => {
    const s = trace(
      lpllpSteps(problem.x, bags, fromData(proportions, [proportions.length / 2, 2]), lp),
      undefined,
      maxSteps,
      {
        keep: 'none',
      },
    ).final
    const F = dense.data(s.scores)
    const scores = Float64Array.from({ length: n }, (_, i) => F[2 * i + 1])
    const labels = Int32Array.from(toFlat(s.labels))
    let hit = 0
    for (const i of test) if (labels[i] === y[i]) hit++
    const uncertainty = Float64Array.from(scores, (f, i) =>
      bags[i] < 0 && !test.has(i) ? Math.abs(f - 0.5) : Infinity,
    )
    const pool = uncertainty.filter((u) => u < Infinity).length
    return {
      t,
      bags,
      proportions,
      scores,
      labels,
      accuracy: test.size > 0 ? hit / test.size : NaN,
      query,
      seed: seedPoint,
      answer,
      uncertainty,
      terminated: pool === 0,
    }
  }
  return {
    name: 'activeProportions',
    init: () => fit(Int32Array.from(problem.bags), binaryProportions(problem.proportions), 0, [], -1, NaN),
    step: (s) => {
      const poolIdx = Array.from({ length: n }, (_, i) => i).filter((i) => s.uncertainty[i] < Infinity)
      const k = Math.min(bagSize, poolIdx.length)
      const byUncertainty = [...poolIdx].sort((a, b) => s.uncertainty[a] - s.uncertainty[b] || a - b)
      let query: number[]
      let seedPoint = -1
      if (strategy === 'random') {
        const order = toFlat(permutation(child(root, 'query', s.t), poolIdx.length))
        query = Array.from(order.slice(0, k), (o) => poolIdx[o])
      } else if (strategy === 'us-mass') {
        seedPoint = byUncertainty[0]
        const near = poolIdx
          .filter((j) => j !== seedPoint)
          .sort((a, b) => L[seedPoint * n + b] - L[seedPoint * n + a] || a - b)
        query = [seedPoint, ...near.slice(0, k - 1)]
      } else query = byUncertainty.slice(0, k)
      const bags = Int32Array.from(s.bags)
      const B = s.proportions.length / 2
      let answer: number
      let proportions: Float64Array
      if (strategy === 'us-exact') {
        // The exact oracle: every queried point becomes a bag of one with its true label.
        proportions = new Float64Array(2 * (B + query.length))
        proportions.set(s.proportions)
        query.forEach((i, r) => {
          bags[i] = B + r
          proportions[2 * (B + r) + y[i]] = 1
        })
        answer = query.reduce((a, i) => a + y[i], 0) / query.length
      } else {
        answer = query.reduce((a, i) => a + y[i], 0) / query.length
        proportions = new Float64Array(2 * (B + 1))
        proportions.set(s.proportions)
        proportions[2 * B] = 1 - answer
        proportions[2 * B + 1] = answer
        for (const i of query) bags[i] = B
      }
      return fit(bags, proportions, s.t + 1, query, seedPoint, answer)
    },
  }
}

/**
 * The paper's starting point for one labelled dataset: a test split (`testShare`, default 0.3), and two bags of
 * `startSize` points (default 16) drawn from outside it with class-1 shares `startProportions` (default 0.75 and
 * 0.25); every other point is in the pool. The bags' proportions are their realised shares.
 */
export function activeProportionsProblem(
  s: Stream,
  data: { x: Tensor; y: Tensor },
  options: { startSize?: Size; startProportions?: readonly number[]; testShare?: number } = {},
): ActiveProportionsProblem {
  const { startSize = 16, startProportions = [0.75, 0.25], testShare = 0.3 } = options
  const y = Int32Array.from(toFlat(data.y))
  const n = y.length
  const order = Array.from(toFlat(permutation(child(s, 'split'), n)))
  const test = order.slice(0, Math.round(testShare * n))
  const rest = order.slice(test.length)
  const bags = new Int32Array(n).fill(-1)
  const ones = rest.filter((i) => y[i] === 1)
  const zeros = rest.filter((i) => y[i] === 0)
  const proportions: number[] = []
  startProportions.forEach((p, b) => {
    const want = Math.round(p * startSize)
    const picked = [...ones.splice(0, want), ...zeros.splice(0, startSize - want)]
    for (const i of picked) bags[i] = b
    const realised = picked.reduce((a, i) => a + y[i], 0) / Math.max(1, picked.length)
    proportions.push(1 - realised, realised)
  })
  return {
    x: data.x,
    labels: y,
    bags,
    proportions: fromData(Float64Array.from(proportions), [startProportions.length, 2]),
    test,
  }
}

/** Options of {@link activeProportionsCurves}. */
export interface ActiveCurvesOptions extends LpLlpOptions {
  /** One labelled dataset per repeat (features [n, d], labels 0/1). */
  datasets: readonly { x: Tensor; y: Tensor }[]
  strategies?: readonly ActiveStrategy[]
  bagSize?: Size
  /** Queries per run (default 4, as the paper). */
  queries?: Size
  /** The two starting bags' size (default 16) and their proportions of class 1 (default 0.75 and 0.25, as the paper). */
  startSize?: Size
  startProportions?: readonly number[]
  /** Share of each dataset held out for testing (default 0.3). */
  testShare?: number
  seed?: number | string
}

/** Mean and sd of the test accuracy after 0 … queries queries, per strategy, over the datasets done so far. */
export interface ActiveCurves {
  readonly strategies: readonly ActiveStrategy[]
  readonly mean: number[][]
  readonly sd: number[][]
  readonly done: number
  readonly total: number
}

/**
 * The paper's experiment as a generator: for each dataset, two starting bags of `startSize` points with proportions
 * (0.75, 0.25) and (0.25, 0.75) of class 1, a test split, and every strategy run for `queries` queries from the same
 * start; yields the accuracy curves after each run so a figure fills in while the worker computes.
 */
export function* activeProportionsCurves(options: ActiveCurvesOptions): Generator<ActiveCurves, ActiveCurves> {
  const {
    datasets,
    strategies = ACTIVE_STRATEGIES,
    bagSize = 10,
    queries = 4,
    startSize = 16,
    startProportions = [0.75, 0.25],
    testShare = 0.3,
    seed = 0,
    ...lp
  } = options
  const root = makeStream(`activeProportionsCurves/${seed}`)
  const runs = strategies.map(() => [] as number[][])
  const total = datasets.length * strategies.length
  let done = 0
  const snapshot = (): ActiveCurves => {
    const mean = runs.map((rs) =>
      Array.from({ length: queries + 1 }, (_, q) => (rs.length ? rs.reduce((a, r) => a + r[q], 0) / rs.length : NaN)),
    )
    const sd = runs.map((rs, s) =>
      Array.from({ length: queries + 1 }, (_, q) =>
        rs.length > 1 ? Math.sqrt(rs.reduce((a, r) => a + (r[q] - mean[s][q]) ** 2, 0) / (rs.length - 1)) : 0,
      ),
    )
    return { strategies, mean, sd, done, total }
  }
  for (let d = 0; d < datasets.length; d++) {
    const problem = activeProportionsProblem(child(root, 'split', d), datasets[d], {
      startSize,
      startProportions,
      testShare,
    })
    for (let s = 0; s < strategies.length; s++) {
      const run = trace(
        activeProportionsSteps(problem, { ...lp, strategy: strategies[s], bagSize, seed: `${seed}/${d}` }),
        undefined,
        queries,
        {
          keep: 'all',
        },
      )
      const curve = run.steps.map((st) => st.accuracy)
      while (curve.length < queries + 1) curve.push(curve[curve.length - 1])
      runs[s].push(curve)
      done++
      yield snapshot()
    }
  }
  return snapshot()
}

/**
 * One active-learning run as a generator for a worker: yields the states so far (step 0, then one more per query) so a
 * figure can show each query as it lands.
 */
export function* activeProportionsRun(
  problem: ActiveProportionsProblem,
  options: ActiveProportionsOptions & { queries?: Size } = {},
): Generator<ActiveProportionsState[], ActiveProportionsState[]> {
  const { queries = 6, ...rest } = options
  const alg = activeProportionsSteps(problem, rest)
  const root = makeStream('activeProportionsRun')
  let s = alg.init(undefined, child(root, 'init'))
  const states = [s]
  yield [...states]
  for (let q = 0; q < queries && !s.terminated; q++) {
    s = alg.step(s, { t: q, stream: child(root, 'step', q) })
    states.push(s)
    yield [...states]
  }
  return states
}
