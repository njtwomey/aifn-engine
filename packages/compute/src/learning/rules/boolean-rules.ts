/**
 * Boolean decision rules by column generation (Dash, Günlük and Wei, 2018, "Boolean decision rules via column
 * generation", NeurIPS, §2 and §3): a DNF rule set of least Hamming loss within a complexity budget, from linear
 * programs over a growing set of clauses.
 *
 * A DNF rule set is a set of clauses, each a conjunction of binary features; a row is predicted positive when any
 * clause holds. With $\mathcal{P}$ the positive rows, $\mathcal{Z}$ the negative ones and $\mathcal{K}_i$ the clauses
 * row $i$ satisfies, the master integer program is
 *
 * $$
 * \min \sum_{i \in \mathcal{P}} \xi_i + \sum_{i \in \mathcal{Z}} \sum_{k \in \mathcal{K}_i} w_k \quad
 * \text{s.t.} \quad \xi_i + \sum_{k \in \mathcal{K}_i} w_k \ge 1 \; (i \in \mathcal{P}), \quad
 * \sum_k c_k w_k \le C, \quad w_k \in \{0, 1\},
 * $$
 *
 * with clause complexity $c_k = 1 + \lvert k \rvert$. Column generation solves its linear relaxation over a growing
 * set of clauses: the restricted master LP (`linprog`) gives duals $\mu_i \ge 0$ for the positive rows and
 * $\lambda \ge 0$ for the budget, and the pricing problem finds the clause of least reduced cost
 * $\sum_{i \in \mathcal{Z}} \delta_i - \sum_{i \in \mathcal{P}} \mu_i \delta_i + \lambda c$ ($\delta_i = 1$ when
 * row $i$ satisfies the clause); a clause with negative reduced cost is added and the LP solved again, until none is
 * left, when the LP is solved to optimality. The lower bound $\lceil z_{\text{RMLP}} + (C/2)\,z_{\text{CG}} \rceil$ on
 * the integer optimum holds at every iteration ($z_{\text{CG}}$ the least reduced cost; every clause costs at least
 * 2), and the final rule set solves the restricted integer program (`milp`) over the generated clauses. Identical rows
 * are aggregated, so each row carries a weight (its count) in the objective.
 *
 * Differences from the paper: the pricing problem is solved by enumerating every clause of at most $D$ features (the
 * paper solves an integer program, with a beam heuristic and sampling for large data), which is exact for clauses of
 * at most $D$ features; one clause, the best, is added per iteration; the restricted integer program adds
 * $\varepsilon c_k$ per selected clause, with $\varepsilon C < 1$, so that among rule sets of equal Hamming loss the
 * one of least complexity is chosen. The duals follow `linprog`'s convention (scipy's, so they are negated here).
 */

import { linprog, milp } from 'aifn-compute/optim/programming'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** Binary rows with identical rows merged: each distinct (row, label) pair once, with its count. */
export type AggregatedBinary = {
  /** The distinct rows, each a 0/1 vector of the $d$ features. */
  rows: Uint8Array[]
  /** The label of each distinct row (1 positive, 0 negative). */
  y: number[]
  /** How many original rows each stands for. */
  count: number[]
}

/**
 * Merge identical binary rows with the same label.
 *
 * @param rows The binary feature rows (0 and 1).
 * @param y The labels (truthy positive, 0 negative).
 * @returns The distinct rows, their labels and counts, in order of first appearance.
 *
 * @example Six rows, three distinct (row, label) pairs
 * print(aggregateBinary([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]], [1, 1, 0, 0, 0, 0]))
 */
export function aggregateBinary(rows: readonly ArrayLike<number>[], y: ArrayLike<number>): AggregatedBinary {
  const index = new Map<string, number>()
  const out: AggregatedBinary = { rows: [], y: [], count: [] }
  rows.forEach((r, i) => {
    const key = `${Array.from(r).join('')}|${y[i]}`
    const at = index.get(key)
    if (at === undefined) {
      index.set(key, out.rows.length)
      out.rows.push(Uint8Array.from(r))
      out.y.push(y[i] ? 1 : 0)
      out.count.push(1)
    } else out.count[at]++
  })
  return out
}

/** A clause of a DNF: the indices of the binary features it requires, ascending. */
export type DnfClause = number[]

/** One iteration of column generation. */
export type ColumnGenerationStep = {
  /** The objective of the restricted master LP before this iteration's clause is added. */
  lpObjective: number
  /** The least reduced cost found by pricing, $z_{\text{CG}}$. */
  reducedCost: number
  /** The clause that attains it. */
  clause: DnfClause
  /** Whether the clause was added (its reduced cost is negative). */
  added: boolean
  /** The lower bound $\lceil z_{\text{RMLP}} + (C/2)\,z_{\text{CG}} \rceil$ on the integer optimum. */
  lowerBound: number
  /** The dual $\lambda$ of the budget. */
  lambda: number
  /** The largest dual $\mu_i$ of the positive rows (0 when there are none). */
  maxMu: number
  /** The LP weights $w_k$ of the clauses generated so far. */
  weights: number[]
}

/** The result of column generation and the restricted integer program. */
export type BooleanRuleResult = {
  /** Every clause generated, in order. */
  clauses: DnfClause[]
  /** The iterations, the last one finding no improving clause (or stopped at `maxIterations`). */
  trace: ColumnGenerationStep[]
  /** Whether the LP relaxation was solved to optimality over all clauses of at most $D$ features. */
  lpOptimal: boolean
  /** Indices into `clauses` of the clauses the restricted integer program selects. */
  selected: number[]
  /** The Hamming loss of the selected rule set. */
  hammingLoss: number
  /** Its 0-1 training errors. */
  errors: number
  /** The best lower bound proved on the Hamming loss of any rule set within the budget (clauses of at most $D$). */
  lowerBound: number
}

/** Options of `booleanRuleColumnGeneration`. */
export type BooleanRuleOptions = {
  /** The complexity budget $C$ (default 10). */
  budget?: number
  /** The most features per clause $D$ (default 3). */
  maxDegree?: number
  /** The most iterations (default 40). */
  maxIterations?: number
  /**
   * `compatible(a, b)`: false for two features that cannot both hold (two levels of one attribute); default all
   * compatible.
   */
  compatible?: (a: number, b: number) => boolean
  /** The same as plain data: the pairs of features that cannot both hold (both apply when both are given). */
  incompatible?: readonly (readonly [number, number])[]
}

/**
 * The rows (indices into the aggregated rows) a clause covers.
 *
 * @param data The aggregated rows.
 * @param clause The clause.
 * @returns One entry per aggregated row, 1 where every feature of the clause is 1.
 */
function coverOf(data: AggregatedBinary, clause: DnfClause): Uint8Array {
  const m = new Uint8Array(data.rows.length)
  for (let i = 0; i < data.rows.length; i++) {
    let ok = 1
    for (const j of clause) {
      if (!data.rows[i][j]) {
        ok = 0
        break
      }
    }
    m[i] = ok
  }
  return m
}

/**
 * Every clause of 1 to `maxDegree` features, in lexicographic order, skipping incompatible pairs.
 *
 * @param d The number of features.
 * @param maxDegree The most features per clause.
 * @param compatible Whether two features may appear together.
 * @returns The clauses.
 */
function allClauses(d: number, maxDegree: number, compatible: (a: number, b: number) => boolean): DnfClause[] {
  const out: DnfClause[] = []
  const grow = (c: DnfClause, from: number) => {
    if (c.length > 0) out.push(c)
    if (c.length === maxDegree) return
    for (let j = from; j < d; j++) if (c.every((k) => compatible(k, j))) grow([...c, j], j + 1)
  }
  grow([], 0)
  return out
}

/**
 * Column generation for a DNF rule set, then the restricted integer program (see the file comment). With the
 * incompatible pairs given as data (`incompatible`) it is a plain function of plain data, so it can run in a worker.
 *
 * @param data The aggregated binary rows.
 * @param options The budget $C$, the degree $D$, the iteration limit and the incompatible features.
 * @returns The clauses generated, the trace, the selected rule set and its losses, and the lower bound.
 *
 * @example The DNF (x0 AND x1) OR x2 recovered from its truth table
 * const rows = []
 * for (let m = 0; m < 8; m++) rows.push([m & 1, (m >> 1) & 1, (m >> 2) & 1])
 * const y = rows.map(([a, b, c]) => ((a && b) || c ? 1 : 0))
 * const r = booleanRuleColumnGeneration(aggregateBinary(rows, y), { budget: 5, maxDegree: 2 })
 * print('clauses:', r.selected.map((k) => r.clauses[k]), ' errors:', r.errors, ' bound:', r.lowerBound)
 */
export function booleanRuleColumnGeneration(
  data: AggregatedBinary,
  options: BooleanRuleOptions = {},
): BooleanRuleResult {
  const C = options.budget ?? 10
  const D = options.maxDegree ?? 3
  const maxIt = options.maxIterations ?? 40
  const d = data.rows.length > 0 ? data.rows[0].length : 0
  const banned = new Set((options.incompatible ?? []).map(([a, b]) => `${Math.min(a, b)},${Math.max(a, b)}`))
  const given = options.compatible ?? (() => true)
  const compatible = (a: number, b: number) => given(a, b) && !banned.has(`${Math.min(a, b)},${Math.max(a, b)}`)
  const pos: number[] = []
  const neg: number[] = []
  data.y.forEach((v, i) => (v ? pos : neg).push(i))
  const candidates = allClauses(d, D, compatible).map((c) => ({ c, cover: coverOf(data, c) }))
  const clauses: DnfClause[] = []
  const covers: Uint8Array[] = []
  const trace: ColumnGenerationStep[] = []
  const negCost = (cover: Uint8Array) => neg.reduce((a, i) => a + (cover[i] ? data.count[i] : 0), 0)
  const P = pos.length
  let lpOptimal = false
  let bestBound = -Infinity

  // The constraint rows: -xi_i - sum of the w_k of clauses covering positive row i <= -1, then the budget.
  const constraints = () => {
    const K = clauses.length
    const A: number[][] = pos.map((i, r) => {
      const row = new Array<number>(P + K).fill(0)
      row[r] = -1
      covers.forEach((cv, k) => {
        if (cv[i]) row[P + k] = -1
      })
      return row
    })
    A.push([...new Array<number>(P).fill(0), ...clauses.map((cl) => 1 + cl.length)])
    return { A, b: [...pos.map(() => -1), C] }
  }

  for (let it = 0; it <= maxIt; it++) {
    // Restricted master LP: variables xi (one per positive row), then w (one per clause).
    const c = [...pos.map((i) => data.count[i]), ...covers.map(negCost)]
    const { A, b } = constraints()
    const lp = linprog({ c, A_ub: A, b_ub: b })
    if (lp.status !== 'optimal' || !lp.report) break
    const duals = toFlat(lp.report.duals.ineq)
    const mu = Array.from(duals.slice(0, P), (v) => -v)
    const lambda = -duals[P]
    const weights = Array.from(toFlat(lp.x).slice(P))
    // Pricing by enumeration.
    let best = { cost: Infinity, k: -1 }
    candidates.forEach((cand, k) => {
      let cost = lambda * (1 + cand.c.length)
      for (const i of neg) if (cand.cover[i]) cost += data.count[i]
      pos.forEach((i, r) => {
        if (cand.cover[i]) cost -= mu[r]
      })
      if (cost < best.cost - 1e-12) best = { cost, k }
    })
    const z = Number(lp.objective)
    const zcg = Math.min(0, best.cost)
    const bound = Math.ceil(z + (C / 2) * zcg - 1e-9)
    bestBound = Math.max(bestBound, bound)
    const added = best.cost < -1e-9 && it < maxIt
    trace.push({
      lpObjective: z,
      reducedCost: best.cost,
      clause: best.k >= 0 ? candidates[best.k].c : [],
      added,
      lowerBound: bound,
      lambda,
      maxMu: mu.reduce((a, v) => Math.max(a, v), 0),
      weights,
    })
    if (best.cost >= -1e-9) {
      lpOptimal = true
      break
    }
    if (!added) break
    clauses.push(candidates[best.k].c)
    covers.push(candidates[best.k].cover)
  }

  // Restricted integer program over the generated clauses. A complexity term of eps per unit, with eps * C < 1, breaks
  // ties between rule sets of equal Hamming loss (an integer) in favour of the simpler one without changing the loss.
  const K = clauses.length
  const eps = 0.5 / (C + 1)
  const c = [...pos.map((i) => data.count[i]), ...covers.map((cv, k) => negCost(cv) + eps * (1 + clauses[k].length))]
  const { A, b } = constraints()
  const ip = milp({
    c,
    A_ub: A,
    b_ub: b,
    bounds: [...pos.map(() => [0, 1] as const), ...clauses.map(() => [0, 1] as const)],
    integrality: [...pos.map(() => 0), ...clauses.map(() => 1)],
  })
  const x = ip.status === 'optimal' ? toFlat(ip.x) : new Array<number>(P + K).fill(0)
  const selected = clauses.map((_, k) => k).filter((k) => x[P + k] > 0.5)
  const pred = dnfCover(
    data,
    selected.map((k) => clauses[k]),
  )
  let hamming = 0
  let errors = 0
  data.rows.forEach((_, i) => {
    if (data.y[i]) {
      if (!pred.any[i]) {
        hamming += data.count[i]
        errors += data.count[i]
      }
    } else {
      hamming += pred.hits[i] * data.count[i]
      if (pred.any[i]) errors += data.count[i]
    }
  })
  return { clauses, trace, lpOptimal, selected, hammingLoss: hamming, errors, lowerBound: bestBound }
}

/**
 * For each aggregated row, how many clauses of a DNF hold, and whether any does.
 *
 * @param data The aggregated rows.
 * @param dnf The clauses.
 * @returns `hits`, the number of clauses that hold, and `any`, whether one does, per aggregated row.
 *
 * @example Two clauses on three rows
 * const data = aggregateBinary([[1, 1, 0], [0, 0, 1], [1, 0, 1]], [1, 1, 0])
 * print(dnfCover(data, [[0, 1], [2]]))
 */
export function dnfCover(data: AggregatedBinary, dnf: readonly DnfClause[]): { hits: number[]; any: boolean[] } {
  const hits = data.rows.map((r) => dnf.filter((cl) => cl.every((j) => r[j] === 1)).length)
  return { hits, any: hits.map((h) => h > 0) }
}

/**
 * The predictions of a DNF on binary rows: 1 when any clause holds.
 *
 * @param dnf The clauses.
 * @param rows The binary rows.
 * @returns 0 or 1 per row.
 *
 * @example (x0 AND x1) OR x2
 * print(dnfPredict([[0, 1], [2]], [[1, 1, 0], [1, 0, 0], [0, 0, 1]]))
 */
export function dnfPredict(dnf: readonly DnfClause[], rows: readonly ArrayLike<number>[]): number[] {
  return rows.map((r) => (dnf.some((cl) => cl.every((j) => r[j] === 1)) ? 1 : 0))
}
