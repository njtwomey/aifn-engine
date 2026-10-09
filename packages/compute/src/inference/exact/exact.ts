/**
 * Exact inference on small discrete factor graphs: brute-force enumeration of every joint assignment, and variable
 * elimination (Zhang & Poole 1994, "A simple approach to Bayesian network computations"; Koller & Friedman 2009,
 * §9.3), each as a traceable algorithm.
 *
 * A graph has variables $0, \dots, V - 1$ and the joint $p(\xvec) = \frac{1}{Z} \prod_a f_a(\xvec_a)$. Joint
 * assignments are visited in row-major order (the last variable changes fastest), and weights are kept in log space.
 * Both methods are exponential in something: enumeration in the number of variables, elimination in the width of the
 * largest factor it builds, which the elimination order decides.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { logAddExp } from 'aifn-compute/numerics/special'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  factorMarginalise,
  factorProductAll,
  factorReduce,
  logPotential,
  normaliseFactor,
  tableSize,
  valuesOf,
  type DiscreteFactor,
  type DiscreteFactorGraph,
} from 'aifn-compute/inference/model'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── Enumeration ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link enumerationSteps}. */
export interface EnumerationOptions {
  /** Assignments visited per step (default 1). */
  chunk?: number
}

/** The state of enumeration after visiting the first `visited` assignments in row-major order. */
export interface EnumerationState extends Status {
  /** The graph being enumerated. */
  graph: DiscreteFactorGraph
  /** Assignments visited per step. */
  chunk: number
  /** Assignments visited so far. */
  visited: Size
  /** Assignments in total ($\prod_v K_v$, the product of the cardinalities). */
  total: Size
  /** The last assignment visited (int32, one entry per variable). */
  assignment: Tensor
  /** The unnormalised log-probability of `assignment` ($-\infty$ before the first step). */
  logWeight: number
  /** $\log \sum$ of the weights visited so far; $\log Z$ once done. */
  logZ: number
  /**
   * Per variable: $\log \sum$ of the weights visited so far with that variable at each value (length: its
   * cardinality).
   */
  logMarginals: Tensor[]
  /** The heaviest assignment so far (int32). */
  best: Tensor
  /** The unnormalised log-probability of `best`. */
  bestLogWeight: number
  /** Whether every assignment has been visited. */
  done: boolean
}

/**
 * The joint assignment at a row-major index (the last variable changes fastest).
 *
 * @param flat The index of the assignment, from 0 to the product of the cardinalities minus one.
 * @param cards The cardinality of each variable.
 * @returns The value of each variable (int32, one entry per variable).
 */
function decode(flat: number, cards: readonly number[]): Int32Array {
  const a = new Int32Array(cards.length)
  for (let i = cards.length - 1; i >= 0; i--) {
    a[i] = flat % cards[i]
    flat = Math.floor(flat / cards[i])
  }
  return a
}

/**
 * Brute-force enumeration as a traceable algorithm: each step visits `chunk` joint assignments (row-major), adding
 * each one's weight $\prod_a f_a(\xvec_a)$ (in log space) to $\log Z$, to the running per-variable marginals and to
 * the running maximum. Exponential in the number of variables: for small models and as the reference for other
 * engines. Ties for the maximum go to the assignment visited first.
 *
 * @param graph The discrete factor graph; read, not modified.
 * @param options How many assignments each step visits.
 * @param options.chunk Assignments visited per step (default 1); the last step visits what is left.
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; it is `done` once every assignment is visited.
 *
 * @example Two steps of two assignments
 * // Three binary variables, eight assignments; a prior on x₀ and agreement factors along x₀ - x₁ - x₂.
 * const agree = tensor([[3, 1], [1, 3]])
 * const graph = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([1, 2]) },
 *     { scope: [0, 1], table: agree },
 *     { scope: [1, 2], table: agree },
 *   ],
 * }
 * const s = run(enumerationSteps(graph, { chunk: 2 }), undefined, 2)
 * print('visited', s.visited, 'of', s.total)
 * print('last assignment =', s.assignment, 'weight', Math.exp(s.logWeight))
 * print('Z so far =', Math.exp(s.logZ))
 * const end = run(enumerationSteps(graph, { chunk: 2 }), undefined, 10)
 * print('Z =', Math.exp(end.logZ), 'best =', end.best)
 */
export function enumerationSteps(
  graph: DiscreteFactorGraph,
  { chunk = 1 }: EnumerationOptions = {},
): Algorithm<void, EnumerationState> {
  return {
    name: 'enumeration',
    init: () => {
      const V = graph.cardinalities.length
      return {
        t: 0,
        graph,
        chunk,
        visited: 0,
        total: tableSize(graph.cardinalities),
        assignment: fromData(new Int32Array(V), [V]),
        logWeight: -Infinity,
        logZ: -Infinity,
        logMarginals: graph.cardinalities.map((k) => fromData(new Float64Array(k).fill(-Infinity), [k])),
        best: fromData(new Int32Array(V), [V]),
        bestLogWeight: -Infinity,
        done: false,
      }
    },
    step: (s) => {
      const marg = s.logMarginals.map((m) => Float64Array.from(m.data))
      let { logZ, bestLogWeight, best, assignment, logWeight } = s
      const end = Math.min(s.total, s.visited + s.chunk)
      for (let flat = s.visited; flat < end; flat++) {
        const a = decode(flat, s.graph.cardinalities)
        const w = logPotential(s.graph, a)
        logZ = logAddExp(logZ, w)
        a.forEach((value, v) => (marg[v][value] = logAddExp(marg[v][value], w)))
        if (w > bestLogWeight) {
          bestLogWeight = w
          best = fromData(a, [a.length])
        }
        assignment = fromData(a, [a.length])
        logWeight = w
      }
      return {
        ...s,
        t: s.t + 1,
        visited: end,
        assignment,
        logWeight,
        logZ,
        logMarginals: marg.map((m) => fromData(m, [m.length])),
        best,
        bestLogWeight,
        done: end >= s.total,
      }
    },
    done: (s) => s.done,
  }
}

/** The result of exact inference: the marginals, $\log Z$ and the MAP assignment. */
export interface ExactResult {
  /** The marginal $p(x_v)$ of each variable $v$ (one float64 vector per variable, of its cardinality). */
  marginals: Tensor[]
  /** The log normaliser $\log Z$. */
  logZ: number
  /** The most probable joint assignment (int32, one entry per variable). */
  map: Tensor
  /** The normalised log-probability of `map`. */
  mapLogProbability: number
}

/**
 * Exact marginals, $\log Z$ and the MAP assignment of a small factor graph by enumerating every assignment: one run of
 * `enumerationSteps` that visits them all. Exponential in the number of variables. Ties for the MAP go to the
 * assignment first in row-major order.
 *
 * @param graph The discrete factor graph; read, not modified.
 * @returns The marginal of every variable, $\log Z$, and the MAP assignment with its probability.
 *
 * @example Rain and wet grass
 * // p(rain) = 0.2, and the grass is wet with probability 0.1 without rain and 0.8 with it.
 * const graph = {
 *   cardinalities: [2, 2],
 *   names: ['rain', 'wet'],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const r = enumerate(graph)
 * print('p(rain) =', r.marginals[0])
 * print('p(wet) =', r.marginals[1])
 * print('Z =', Math.exp(r.logZ))
 * print('MAP =', r.map, 'with probability', Math.exp(r.mapLogProbability))
 */
export function enumerate(graph: DiscreteFactorGraph): ExactResult {
  const total = tableSize(graph.cardinalities)
  const s = run(enumerationSteps(graph, { chunk: total }), undefined, 1)
  return {
    marginals: s.logMarginals.map((m) =>
      fromData(
        m.data.map((l) => Math.exp(l - s.logZ)),
        m.shape,
      ),
    ),
    logZ: s.logZ,
    map: s.best,
    mapLogProbability: s.bestLogWeight - s.logZ,
  }
}

/**
 * The full normalised joint $p(\xvec)$ as a tensor with one axis per variable (small graphs only: it holds every
 * assignment).
 *
 * @param graph The discrete factor graph; read, not modified.
 * @returns The float64 tensor of shape `graph.cardinalities` whose entry at $\xvec$ is $p(\xvec)$; it sums to one.
 *
 * @example An unnormalised pair, normalised
 * // One factor that prefers agreement 3 to 1: the joint is the table divided by its sum.
 * const graph = { cardinalities: [2, 2], factors: [{ scope: [0, 1], table: tensor([[3, 1], [1, 3]]) }] }
 * print('p(x) =', jointDistribution(graph))
 */
export function jointDistribution(graph: DiscreteFactorGraph): Tensor {
  const total = tableSize(graph.cardinalities)
  const logs = new Float64Array(total)
  let logZ = -Infinity
  for (let flat = 0; flat < total; flat++) {
    logs[flat] = logPotential(graph, decode(flat, graph.cardinalities))
    logZ = logAddExp(logZ, logs[flat])
  }
  return fromData(
    logs.map((l) => Math.exp(l - logZ)),
    graph.cardinalities,
  )
}

// ── Variable elimination ────────────────────────────────────────────────────────────────────────────────────────────

/** How the next variable to eliminate is chosen: a fixed order, or greedily by min-degree or min-fill. */
export type EliminationOrder = readonly number[] | 'min-degree' | 'min-fill'

/** Options of {@link variableEliminationSteps}. */
export interface EliminationOptions {
  /** Variables to keep (the query); all others are eliminated. Default none: the result is $Z$. */
  query?: readonly number[]
  /** Observed values, as variable index to value. Observed variables are conditioned on, not eliminated. */
  evidence?: Readonly<Record<number, number>>
  /**
   * The elimination order (default `'min-fill'`). A fixed order must list every variable that is neither queried nor
   * observed (any others it lists are skipped).
   */
  order?: EliminationOrder
  /** `'sum'` for marginals and $Z$, `'max'` for the max-marginal (MAP value). Default `'sum'`. */
  mode?: 'sum' | 'max'
}

/** One elimination: the variable, the factors multiplied and the new factor (a message $\tau$) that replaced them. */
export interface EliminationEvent {
  /** The variable eliminated. */
  variable: number
  /** Indices (into the previous state's `factors`) of the factors that mentioned the variable. */
  used: number[]
  /** The scope of their product before the variable was summed out (its width is the cost of this step). */
  productScope: number[]
  /**
   * The new factor $\tau$ over the other variables of the product, normalised to sum to one (left as it is when all
   * zero).
   */
  created: DiscreteFactor
}

/** The state of variable elimination: the current factor list, and what the last step did. */
export interface EliminationState extends Status {
  /** The graph, as given (without the evidence applied). */
  graph: DiscreteFactorGraph
  /** Whether variables are summed (`'sum'`) or maximised (`'max'`) out. */
  mode: 'sum' | 'max'
  /** The elimination order, fixed or greedy. */
  order: EliminationOrder
  /** The factors still in play; each new factor is normalised, its log-scale moved to `logScale`. */
  factors: DiscreteFactor[]
  /** Variables still to eliminate. */
  remaining: number[]
  /** Variables eliminated so far, in order. */
  eliminated: number[]
  /** The query variables, which are kept. */
  query: number[]
  /** The sum of the log normalisers taken out of the new factors. */
  logScale: number
  /** The largest product scope so far (the induced width $+ 1$). */
  maxScope: number
  /** What the last step did (null before the first). */
  last: EliminationEvent | null
  /** Whether every variable outside the query and the evidence is eliminated. */
  done: boolean
}

/**
 * Variables sharing a factor with $v$ among the current factors.
 *
 * @param factors The factors in play.
 * @param v The variable whose neighbours are wanted.
 * @returns The other variables of every factor that mentions $v$.
 */
function neighbourSet(factors: readonly DiscreteFactor[], v: number): Set<number> {
  const out = new Set<number>()
  for (const f of factors) if (f.scope.includes(v)) for (const u of f.scope) if (u !== v) out.add(u)
  return out
}

/**
 * Fill-in edges eliminating $v$ would add between its neighbours: the pairs of them that share no factor.
 *
 * @param factors The factors in play.
 * @param v The candidate variable.
 * @returns The number of new edges.
 */
function fillIn(factors: readonly DiscreteFactor[], v: number): number {
  const nb = [...neighbourSet(factors, v)]
  let fill = 0
  for (let i = 0; i < nb.length; i++)
    for (let j = i + 1; j < nb.length; j++)
      if (!factors.some((f) => f.scope.includes(nb[i]) && f.scope.includes(nb[j]))) fill++
  return fill
}

/**
 * The variable to eliminate next: the first remaining one for a fixed order, else the remaining one with the fewest
 * neighbours (min-degree) or fill-in edges (min-fill) among the current factors, ties to the earliest.
 *
 * @param s The elimination state; its `remaining` must not be empty.
 * @returns The variable's index.
 */
function nextVariable(s: EliminationState): number {
  if (Array.isArray(s.order)) return s.remaining[0]
  let best = s.remaining[0]
  let score = Infinity
  for (const v of s.remaining) {
    const c = s.order === 'min-degree' ? neighbourSet(s.factors, v).size : fillIn(s.factors, v)
    if (c < score) [best, score] = [v, c]
  }
  return best
}

/**
 * Variable elimination as a traceable algorithm (Zhang & Poole 1994; Koller & Friedman 2009, §9.3). Evidence is
 * applied at `init`, by reducing every factor to the observed values; each step eliminates one variable: it multiplies
 * the factors that mention it, sums (or maximises) it out and puts the new factor back, normalised, with its log-scale
 * added to `logScale`. The order is fixed or greedy (min-degree, min-fill), recomputed on the current factors at each
 * step. `eliminationResult` turns the finished state into the query marginal and $\log Z$. Throws `DomainError` at
 * `init` when a fixed order leaves out a variable that must be eliminated.
 *
 * @param graph The discrete factor graph; its factors are read, not modified.
 * @param options What to keep, what is observed, the order and the mode.
 * @param options.query Variables to keep (default none: everything is eliminated, and the result is $Z$).
 * @param options.evidence Observed values, as variable index to value (default none).
 * @param options.order A fixed order of the variables to eliminate, or `'min-degree'` or `'min-fill'` to choose
 *   greedily at each step (default `'min-fill'`).
 * @param options.mode `'sum'` to sum variables out (marginals and $Z$) or `'max'` to maximise them out (the
 *   max-marginal and the MAP value). Default `'sum'`.
 * @returns The algorithm, to run with `run(alg, undefined, steps)`; one step per eliminated variable.
 *
 * @example Eliminating a chain one variable at a time
 * // x₀ - x₁ - x₂ with a prior on x₀; keep x₂ and eliminate x₀ then x₁.
 * const copy = tensor([[0.9, 0.1], [0.1, 0.9]])
 * const graph = {
 *   cardinalities: [2, 2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.7, 0.3]) },
 *     { scope: [0, 1], table: copy },
 *     { scope: [1, 2], table: copy },
 *   ],
 * }
 * const alg = variableEliminationSteps(graph, { query: [2], order: [0, 1] })
 * for (const steps of [1, 2]) {
 *   const s = run(alg, undefined, steps)
 *   print(`step ${steps}: eliminated x${s.last.variable}, multiplied`, s.last.used, 'into', s.last.created.table)
 * }
 * print('p(x2) =', eliminationResult(run(alg, undefined, 2)).marginal)
 *
 * @example Min-fill picks the cheap end of a chain first
 * const copy = tensor([[0.9, 0.1], [0.1, 0.9]])
 * const graph = {
 *   cardinalities: [2, 2, 2, 2],
 *   factors: [
 *     { scope: [0, 1], table: copy },
 *     { scope: [1, 2], table: copy },
 *     { scope: [2, 3], table: copy },
 *   ],
 * }
 * const s = run(variableEliminationSteps(graph), undefined, 4)
 * print('order =', s.eliminated)
 * print('largest product =', s.maxScope, 'variables')
 * print('Z =', Math.exp(eliminationResult(s).logZ))
 */
export function variableEliminationSteps(
  graph: DiscreteFactorGraph,
  { query = [], evidence = {}, order = 'min-fill', mode = 'sum' }: EliminationOptions = {},
): Algorithm<void, EliminationState> {
  return {
    name: 'variable-elimination',
    init: () => {
      const ev = new Map(Object.entries(evidence).map(([k, v]) => [Number(k), v]))
      const factors = graph.factors.map((f) => factorReduce(f, ev))
      const remaining = Array.isArray(order)
        ? order.filter((v) => !query.includes(v) && !ev.has(v))
        : graph.cardinalities.map((_, v) => v).filter((v) => !query.includes(v) && !ev.has(v))
      if (Array.isArray(order)) {
        const missing = graph.cardinalities
          .map((_, v) => v)
          .filter((v) => !query.includes(v) && !ev.has(v) && !order.includes(v))
        if (missing.length)
          throw new DomainError(
            'variableElimination',
            `variableElimination: order omits variables ${missing.join(', ')}`,
          )
      }
      return {
        t: 0,
        graph,
        mode,
        order,
        factors,
        remaining,
        eliminated: [],
        query: [...query],
        logScale: 0,
        maxScope: 0,
        last: null,
        done: remaining.length === 0,
      }
    },
    step: (s) => {
      const v = nextVariable(s)
      const used: number[] = []
      const keep: DiscreteFactor[] = []
      s.factors.forEach((f, k) => (f.scope.includes(v) ? used.push(k) : keep.push(f)))
      const product = factorProductAll(
        used.map((k) => s.factors[k]),
        s.graph.cardinalities,
      )
      const { factor, logNormaliser } = normaliseFactor(factorMarginalise(product, [v], s.mode))
      const created = { ...factor, name: `τ${v}` }
      const remaining = s.remaining.filter((u) => u !== v)
      return {
        ...s,
        t: s.t + 1,
        factors: [...keep, created],
        remaining,
        eliminated: [...s.eliminated, v],
        logScale: s.logScale + logNormaliser,
        maxScope: Math.max(s.maxScope, product.scope.length),
        last: { variable: v, used, productScope: [...product.scope], created },
        done: remaining.length === 0,
      }
    },
    done: (s) => s.done,
  }
}

/** The result of variable elimination. */
export interface EliminationResult {
  /**
   * $p(\text{query} \mid \text{evidence})$ as a table over `query` in the given order (normalised), or the
   * max-marginal with `'max'` (normalised to sum to one). A scalar table when the query is empty.
   */
  marginal: Tensor
  /**
   * $\log Z$ with the evidence applied, that is $\log p(\text{evidence})$ for a normalised model (with `'max'`, the
   * log of the largest unnormalised joint).
   */
  logZ: number
  /** The variables in the order they were eliminated. */
  order: number[]
  /** The largest factor built, in variables: the induced width $+ 1$. */
  maxScope: number
}

/**
 * Run variable elimination to the end and combine the remaining factors over the query: `variableEliminationSteps`
 * then `eliminationResult`. Throws `DomainError` when a fixed order leaves out a variable that must be eliminated.
 *
 * @param graph The discrete factor graph; its factors are read, not modified.
 * @param query The variables to keep, in the order of the axes of the result (default none: only $Z$ is wanted).
 * @param options The evidence, the order and the mode, as for `variableEliminationSteps`.
 * @returns The query marginal, $\log Z$, the elimination order and the largest factor built.
 *
 * @example Rain given wet grass
 * // p(rain) = 0.2, p(wet | no rain) = 0.1, p(wet | rain) = 0.8. By Bayes, p(rain | wet) = 0.2 · 0.8 / p(wet), with
 * // p(wet) = 0.8 · 0.1 + 0.2 · 0.8.
 * const graph = {
 *   cardinalities: [2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const r = variableElimination(graph, [0], { evidence: { 1: 1 } })
 * print('p(rain | wet) =', r.marginal)
 * print('p(wet) =', Math.exp(r.logZ))
 *
 * @example The MAP value with max
 * const graph = {
 *   cardinalities: [2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const r = variableElimination(graph, [], { mode: 'max' })
 * print('largest joint probability =', Math.exp(r.logZ))
 */
export function variableElimination(
  graph: DiscreteFactorGraph,
  query: readonly number[] = [],
  options: Omit<EliminationOptions, 'query'> = {},
): EliminationResult {
  const s = run(variableEliminationSteps(graph, { ...options, query }), undefined, graph.cardinalities.length)
  return eliminationResult(s)
}

/**
 * Combine a finished elimination state's factors into the query marginal and $\log Z$: their product, with its axes
 * in the order of the query, normalised.
 *
 * @param s A state of `variableEliminationSteps` with every variable eliminated (`done`); on an unfinished one the
 *   product also holds the remaining variables.
 * @returns The query marginal, $\log Z$, the elimination order and the largest factor built.
 *
 * @example Stepping, then reading the result
 * const graph = {
 *   cardinalities: [2, 2],
 *   factors: [
 *     { scope: [0], table: tensor([0.8, 0.2]) },
 *     { scope: [0, 1], table: tensor([[0.9, 0.1], [0.2, 0.8]]) },
 *   ],
 * }
 * const s = run(variableEliminationSteps(graph, { query: [1] }), undefined, 10)
 * const r = eliminationResult(s)
 * print('p(wet) =', r.marginal)
 * print('eliminated', r.order, 'Z =', Math.exp(r.logZ))
 */
export function eliminationResult(s: EliminationState): EliminationResult {
  let product = factorProductAll(s.factors, s.graph.cardinalities)
  // Order the axes as the query was given.
  if (s.query.length) {
    const cards = s.graph.cardinalities
    const from = product
    product = factorProductAll(
      [
        {
          scope: s.query,
          table: fromData(
            new Float64Array(tableSize(s.query.map((v) => cards[v]))).fill(1),
            s.query.map((v) => cards[v]),
          ),
        },
        from,
      ],
      cards,
    )
  }
  const values = valuesOf(product.table)
  let total = 0
  for (const x of values) total += x
  return {
    marginal: fromData(
      values.map((x) => x / total),
      product.table.shape,
    ),
    logZ: s.logScale + Math.log(s.mode === 'max' ? Math.max(...values) : total),
    order: s.eliminated,
    maxScope: s.maxScope,
  }
}
