/**
 * Active learning with label proportions (Poyiadzi, Santos-Rodriguez and Twomey 2019, ICASSP): the learner holds a
 * few bags with known class proportions and a pool $\Ucal$ of points in no bag; it builds a bag of $k$ points from
 * $\Ucal$ and asks an LLP-oracle, which answers with the bag's class proportion only (the true label when $k = 1$). The
 * learner is LP-LLP (`lpllpSteps`, Poyiadzi et al. 2018) with two classes, labelled 0 and 1 here where the papers write
 * $\pm 1$; its score $f_i \in [0, 1]$ is a point's class-1 column, and points in no bag start at the uninformative
 * score $\tfrac{1}{2}$ and are not constrained. Query strategies (§3.3 and §4):
 *
 * - **US-Mass**: the pool point closest to the decision boundary (smallest $\lvert f_i - \tfrac{1}{2} \rvert$), and
 *   the $k - 1$ pool points closest to it under $\Lmat_{\Ucal} = (\Imat - \alpha\Smat)^{-1}$ restricted to $\Ucal$,
 *   the matrix LP-LLP already computed: aims at a pure bag around the most uncertain point;
 * - **US-LP**: the $k$ most uncertain pool points, answered with their proportion;
 * - **Random**: $k$ random pool points, answered with their proportion;
 * - **US-Exact**: the $k$ most uncertain points, answered with their $k$ true labels ($k$ singleton bags).
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

/**
 * The setting of an active-learning run: the points, the oracle's labels, the starting bags, the pool and the test
 * set.
 */
export interface ActiveProportionsProblem {
  /** The points, $n \times d$. */
  readonly x: MatrixLike
  /** True labels 0/1, read only by the oracle and the accuracy. */
  readonly labels: ArrayLike<number>
  /** The starting bags: each point's bag, or $-1$ for none (the pool and the test points). */
  readonly bags: ArrayLike<number>
  /** The starting bags' class proportions, $B \times 2$. */
  readonly proportions: MatrixLike
  /** Points that are never queried and on which accuracy is measured. */
  readonly test: ArrayLike<number>
}

/** Options of an active-learning run. */
export interface ActiveProportionsOptions extends LpLlpOptions {
  /** The query strategy (default `us-lp`). */
  strategy?: ActiveStrategy
  /** Points per query $k$ (default 10; fewer when the pool runs low). */
  bagSize?: Size
  /** The most LP-LLP steps per refit (default 300). */
  maxSteps?: Size
  /** Seed of the random strategy's draws (default 0). */
  seed?: number | string
}

/** One state: the bags after t queries, LP-LLP's fit to them, and the next query's ranking. */
export interface ActiveProportionsState extends Status {
  /** Each point's bag, or $-1$ for none. */
  readonly bags: Int32Array
  /** The bags' proportions, $B \times 2$ row-major. */
  readonly proportions: Float64Array
  /** LP-LLP's scores $f_i = F_{i,1} \in [0, 1]$. */
  readonly scores: Float64Array
  /** LP-LLP's labels. */
  readonly labels: Int32Array
  /** Accuracy on the test points (NaN with none). */
  readonly accuracy: number
  /** The points of the last query (empty at $t = 0$). */
  readonly query: readonly number[]
  /** The last query's seed point for US-Mass ($-1$ otherwise). */
  readonly seed: number
  /** The oracle's answer: the share of class 1 in the last query (NaN at $t = 0$). */
  readonly answer: number
  /**
   * Uncertainty $\lvert f_i - \tfrac{1}{2} \rvert$ of every point (Infinity outside the pool): what the next
   * uncertainty query ranks.
   */
  readonly uncertainty: Float64Array
}

/**
 * The starting proportions as a row-major copy, checking that there are two classes (`DomainError` otherwise).
 *
 * @param P The proportions, $B \times 2$.
 * @returns The proportions, $2B$ values row-major.
 */
const binaryProportions = (P: MatrixLike) => {
  const m = dense.toMatrixF64(P, 'activeProportionsSteps')
  if (m.n !== 2)
    throw new DomainError('activeProportionsSteps', 'activeProportionsSteps: two classes only, as in the paper')
  return Float64Array.from(m.data)
}

/**
 * Active learning with an LLP-oracle as steps: step 0 fits LP-LLP to the starting bags; each step queries a bag of $k$
 * pool points by the strategy (see the file's notes), adds the oracle's answer as a new bag ($k$ singleton bags for
 * US-Exact) and refits LP-LLP from scratch. The state is `terminated` when the pool is empty. Ties in uncertainty go to
 * the smaller index. Throws `DomainError` unless the proportions have two classes.
 *
 * @param problem The points, labels, starting bags and test set.
 * @param options The strategy, the query size, LP-LLP's options and steps per refit, and the seed.
 * @returns The algorithm, to run with `run` or `trace` (its input is unused).
 *
 * @example Two US-Mass queries of four points each
 * const s = stream(30)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const data = { x: tensor(x), y: tensor(y) }
 * const problem = activeProportionsProblem(stream(31), data, { startSize: 6 })
 * const alg = activeProportionsSteps(problem, { strategy: 'us-mass', bagSize: 4, maxSteps: 20 })
 * for (const st of trace(alg, undefined, 2, { keep: 'all' }).steps) {
 *   print('query', st.t, ':', st.query, ' share of class 1:', st.answer, ' test accuracy:', st.accuracy)
 * }
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
 * 0.25); every other point is in the pool. Each bag takes `startSize` times its share of class-1 points, rounded, and
 * fills up with class 0, each in a random order; a bag is smaller when a class runs out. The bags' proportions are
 * their realised shares.
 *
 * @param s The stream of the random order that makes the split and fills the bags.
 * @param data The points `x` ($n \times d$) and their labels `y` (0 or 1).
 * @param options `startSize`, `startProportions` (one class-1 share per starting bag) and `testShare`.
 * @returns The problem: points, labels, starting bags with their proportions, and test points.
 *
 * @example Two starting bags of six points, and a test split
 * const s = stream(30)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const data = { x: tensor(x), y: tensor(y) }
 * const problem = activeProportionsProblem(stream(31), data, { startSize: 6 })
 * print('bags:', problem.bags)
 * print('proportions:', problem.proportions)
 * print('test points:', problem.test.length, ' pool:', problem.bags.filter((b) => b < 0).length - problem.test.length)
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
  /** One labelled dataset per repeat (features $n \times d$, labels 0/1). */
  datasets: readonly { x: Tensor; y: Tensor }[]
  /** The strategies to run (default all, in the paper's order). */
  strategies?: readonly ActiveStrategy[]
  /** Points per query (default 10). */
  bagSize?: Size
  /** Queries per run (default 4, as the paper). */
  queries?: Size
  /** The starting bags' size (default 16). */
  startSize?: Size
  /** The starting bags' proportions of class 1 (default 0.75 and 0.25, as the paper). */
  startProportions?: readonly number[]
  /** Share of each dataset held out for testing (default 0.3). */
  testShare?: number
  /** Seed of the splits, the starting bags and the random strategy (default 0). */
  seed?: number | string
}

/** Mean and sd of the test accuracy after $0, \dots, q$ queries, per strategy, over the datasets done so far. */
export interface ActiveCurves {
  /** The strategies run. */
  readonly strategies: readonly ActiveStrategy[]
  /** `mean[s][q]`: the mean test accuracy of strategy `s` after `q` queries (NaN before its first run). */
  readonly mean: number[][]
  /** `sd[s][q]`: the sample standard deviation of the same (0 with fewer than two runs). */
  readonly sd: number[][]
  /** Runs (dataset and strategy) done so far. */
  readonly done: number
  /** Runs in all. */
  readonly total: number
}

/**
 * The paper's experiment as a generator: for each dataset, a test split and starting bags of `startSize` points
 * (`activeProportionsProblem`), and every strategy run for `queries` queries from the same start; yields the accuracy
 * curves after each run so a figure fills in while the worker computes. A run whose pool empties early carries its last
 * accuracy forward.
 *
 * @param options The datasets, the strategies, the query size and count, the start, LP-LLP's options and the seed.
 * @returns A generator of the curves so far, one after each run; it returns the final ones.
 *
 * @example US-LP against random queries on one small dataset
 * const s = stream(30)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const data = { x: tensor(x), y: tensor(y) }
 * const options = { strategies: ['us-lp', 'random'], queries: 2, bagSize: 4, startSize: 6, maxSteps: 20 }
 * let last
 * for (const curves of activeProportionsCurves({ datasets: [data], ...options })) last = curves
 * print(last.strategies, ' test accuracy after 0, 1 and 2 queries:', last.mean)
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
 * figure can show each query as it lands. Stops after `queries` queries (default 6) or when the pool is empty.
 *
 * @param problem The points, labels, starting bags and test set.
 * @param options The options of `activeProportionsSteps`, and `queries`.
 * @returns A generator of the states so far; it returns them all.
 *
 * @example Three exact queries of three points
 * const s = stream(30)
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const x = y.map((c) => [normal(s, c === 1 ? 1.5 : -1.5, 1), normal(s, 0, 1)])
 * const data = { x: tensor(x), y: tensor(y) }
 * const problem = activeProportionsProblem(stream(31), data, { startSize: 6 })
 * const options = { strategy: 'us-exact', bagSize: 3, queries: 3, maxSteps: 20 }
 * let states
 * for (const st of activeProportionsRun(problem, options)) states = st
 * print('queried:', states.map((st) => st.query))
 * print('test accuracy:', states.map((st) => st.accuracy))
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
