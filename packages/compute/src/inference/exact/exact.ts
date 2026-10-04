/**
 * Exact inference on small discrete factor graphs: brute-force enumeration of every joint assignment, and variable
 * elimination (Zhang & Poole 1994, "A simple approach to Bayesian network computations"; Koller & Friedman 2009,
 * §9.3), each as a traceable algorithm.
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
  graph: DiscreteFactorGraph
  chunk: number
  /** Assignments visited so far, and in total (Π cardinalities). */
  visited: Size
  total: Size
  /** The last assignment visited (int32, one entry per variable), and its unnormalised log-probability. */
  assignment: Tensor
  logWeight: number
  /** log Σ of the weights visited so far; log Z once done. */
  logZ: number
  /** Per variable: log Σ of the weights visited so far with that variable at each value (length = its cardinality). */
  logMarginals: Tensor[]
  /** The heaviest assignment so far (int32) and its unnormalised log-probability. */
  best: Tensor
  bestLogWeight: number
  done: boolean
}

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
 * each one's weight Πₐ fₐ(x) (in log space) to log Z, to the running per-variable marginals and to the running
 * maximum. Exponential in the number of variables: for small models and as the reference for other engines.
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

/** The result of exact inference: marginals p(xᵥ) (one float64 vector per variable), log Z and the MAP assignment. */
export interface ExactResult {
  marginals: Tensor[]
  logZ: number
  /** The most probable joint assignment (int32) and its normalised log-probability. */
  map: Tensor
  mapLogProbability: number
}

/** Exact marginals, log Z and the MAP assignment of a small factor graph by enumerating every assignment. */
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

/** The full normalised joint p(x) as a tensor with one axis per variable (small graphs only). */
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
  /** Variables to keep (the query); all others are eliminated. Default none: the result is Z. */
  query?: readonly number[]
  /** Observed values: variable → value. */
  evidence?: Readonly<Record<number, number>>
  /** Default `min-fill`. */
  order?: EliminationOrder
  /** `sum` for marginals and Z, `max` for the max-marginal (MAP value). Default `sum`. */
  mode?: 'sum' | 'max'
}

/** One elimination: the variable, the factors multiplied and the new factor (a message τ) that replaced them. */
export interface EliminationEvent {
  variable: number
  /** Indices (into the previous state's `factors`) of the factors that mentioned the variable. */
  used: number[]
  /** The scope of their product before the variable was summed out (its width is the cost of this step). */
  productScope: number[]
  created: DiscreteFactor
}

/** The state of variable elimination: the current factor list, and what the last step did. */
export interface EliminationState extends Status {
  graph: DiscreteFactorGraph
  mode: 'sum' | 'max'
  order: EliminationOrder
  /** The factors still in play; each new factor is normalised, its log-scale moved to `logScale`. */
  factors: DiscreteFactor[]
  /** Variables still to eliminate, and those eliminated so far (in order). */
  remaining: number[]
  eliminated: number[]
  query: number[]
  logScale: number
  /** The largest product scope so far (the induced width + 1). */
  maxScope: number
  last: EliminationEvent | null
  done: boolean
}

/** Variables sharing a factor with v among the current factors. */
function neighbourSet(factors: readonly DiscreteFactor[], v: number): Set<number> {
  const out = new Set<number>()
  for (const f of factors) if (f.scope.includes(v)) for (const u of f.scope) if (u !== v) out.add(u)
  return out
}

/** Fill-in edges eliminating v would add between its neighbours. */
function fillIn(factors: readonly DiscreteFactor[], v: number): number {
  const nb = [...neighbourSet(factors, v)]
  let fill = 0
  for (let i = 0; i < nb.length; i++)
    for (let j = i + 1; j < nb.length; j++)
      if (!factors.some((f) => f.scope.includes(nb[i]) && f.scope.includes(nb[j]))) fill++
  return fill
}

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
 * Variable elimination as a traceable algorithm. Evidence is applied at `init`; each step eliminates one variable:
 * it multiplies the factors that mention it, sums (or maximises) it out and puts the new factor back. The order is
 * fixed or greedy (min-degree, min-fill), recomputed on the current factors at each step.
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
  /** p(query | evidence) as a table over `query` in the given order (normalised), or the max-marginal with `max`. */
  marginal: Tensor
  /** log Z with the evidence applied, i.e. log p(evidence) for a normalised model (log max for `max`). */
  logZ: number
  order: number[]
  /** The largest factor built, in variables: the induced width + 1. */
  maxScope: number
}

/** Run variable elimination to the end and combine the remaining factors over the query. */
export function variableElimination(
  graph: DiscreteFactorGraph,
  query: readonly number[] = [],
  options: Omit<EliminationOptions, 'query'> = {},
): EliminationResult {
  const s = run(variableEliminationSteps(graph, { ...options, query }), undefined, graph.cardinalities.length)
  return eliminationResult(s)
}

/** Combine a finished elimination state's factors into the query marginal and log Z. */
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
