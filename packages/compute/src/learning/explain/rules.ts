/**
 * Rule-based explanations. A rule is a conjunction of interval predicates $\text{lower} < x_j \le \text{upper}$.
 *
 * - `anchor` (Ribeiro, Singh and Guestrin, 2018): a rule that holds at $\xvec$ and "anchors" the prediction: with the
 *   rule's features drawn within its intervals and the rest from the data, the model keeps $\xvec$'s label with
 *   precision at least $\tau$ (with confidence $1 - \delta$). Among such rules it seeks the one covering most of the
 *   data. Predicates are $\xvec$'s quantile bin per feature; rules grow one predicate at a time by beam search, the
 *   best `beam` candidates of each size chosen by KL-LUCB (Kaufmann and Kalyanakrishnan, 2013), a best-arm
 *   identification bandit with KL confidence bounds.
 * - `ruleList`: a decision list learned by sequential covering (Rivest, 1987; Clark and Niblett, 1989): repeatedly the
 *   conjunction (by beam search over threshold predicates) whose covered rows are most purely one class (Laplace
 *   estimate), then remove the rows it covers; a default class ends the list. Fitted to a black box's predictions it
 *   is a global surrogate.
 * - `treeRules`: the root-to-leaf rules of a tree; `fidelity`: how closely a surrogate reproduces a model (agreement
 *   for labels, $R^2$ for scores) (Craven and Shavlik, 1996).
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { child, integers, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { quantile } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { ShapTree } from './tree'

/**
 * The predicate $\text{lower} < x_{\text{feature}} \le \text{upper}$ on one `feature`; either end may be infinite.
 */
export type Predicate = { feature: Size; lower: number; upper: number }

/**
 * Whether a row meets every predicate of a rule (an empty rule is met by every row).
 *
 * @param row The row, indexed by feature.
 * @param rule The predicates.
 * @returns True when the row meets them all.
 *
 * @example The lower end is open, the upper end closed
 * const rule = [{ feature: 0, lower: 1, upper: 3 }]
 * print(satisfies([3, 9], rule), satisfies([1, 9], rule))
 */
export function satisfies(row: ArrayLike<number>, rule: readonly Predicate[]): boolean {
  for (const p of rule) {
    const v = row[p.feature]
    if (!(v > p.lower && v <= p.upper)) return false
  }
  return true
}

/**
 * The fraction of rows that meet a rule.
 *
 * @param X The rows ($n \times d$).
 * @param rule The predicates.
 * @returns The fraction, in $[0, 1]$ (0 when there are no rows).
 *
 * @example Half the rows
 * print(ruleCoverage([[0, 0], [1, 1], [2, 0], [3, 1]], [{ feature: 0, lower: 0.5, upper: 2.5 }]))
 */
export function ruleCoverage(X: MatrixLike, rule: readonly Predicate[]): number {
  const { data, m, n } = dense.toMatrixF64(X, 'ruleCoverage')
  let c = 0
  for (let i = 0; i < m; i++) if (satisfies(data.subarray(i * n, (i + 1) * n), rule)) c++
  return m > 0 ? c / m : 0
}

/**
 * A rule with the intervals on the same feature intersected, one predicate per feature, sorted by feature. An empty
 * intersection is kept as an interval with $\text{lower} \ge \text{upper}$, which no row meets.
 *
 * @param rule The predicates; not modified.
 * @returns The simplified rule.
 *
 * @example Two conditions on one feature become one
 * const rule = [
 *   { feature: 1, lower: 0, upper: 5 },
 *   { feature: 0, lower: -1, upper: 1 },
 *   { feature: 1, lower: 2, upper: 9 },
 * ]
 * print(simplifyRule(rule))
 */
export function simplifyRule(rule: readonly Predicate[]): Predicate[] {
  const by = new Map<number, Predicate>()
  for (const p of rule) {
    const q = by.get(p.feature)
    by.set(
      p.feature,
      q ? { feature: p.feature, lower: Math.max(q.lower, p.lower), upper: Math.min(q.upper, p.upper) } : { ...p },
    )
  }
  return [...by.values()].sort((a, b) => a.feature - b.feature)
}

/**
 * Inner bin edges per feature: the distinct $j/\text{bins}$ quantiles ($j = 1, \dots, \text{bins} - 1$, linear
 * interpolation) of each column. Throws `DomainError` when `bins` is not an integer of at least 2.
 *
 * @param X The data ($n \times d$).
 * @param bins The number of bins per feature.
 * @returns For each feature, its inner edges in increasing order (fewer than $\text{bins} - 1$ when quantiles
 *   coincide).
 *
 * @example Quartiles of a spread column and a two-valued one
 * print(quantileEdges([[1, 5], [2, 5], [3, 5], [4, 5], [5, 6], [6, 6], [7, 6], [8, 6]]))
 */
export function quantileEdges(X: MatrixLike, bins = 4): Float64Array[] {
  if (!(Number.isInteger(bins) && bins >= 2)) throw new DomainError('quantileEdges', 'quantileEdges: bins must be ≥ 2')
  const { data, m, n } = dense.toMatrixF64(X, 'quantileEdges')
  return Array.from({ length: n }, (_, j) => {
    const col = Array.from({ length: m }, (_, i) => data[i * n + j])
    const qs = Array.from({ length: bins - 1 }, (_, k) => quantile(col, (k + 1) / bins))
    return Float64Array.from(new Set(qs))
  })
}

/**
 * The bin $\xvec$ falls in on each feature, as a predicate: between the inner edges around $x_j$, open below and
 * closed above, with infinite ends outside the first and last edge.
 *
 * @param edges The inner bin edges of each feature, increasing, as `quantileEdges` returns them.
 * @param x The point $\xvec$ ($d$ values).
 * @returns One predicate per feature.
 *
 * @example The bins of one point
 * print(binPredicates([[2.75, 4.5, 6.25], [5.5]], [5, 7]))
 */
export function binPredicates(edges: readonly ArrayLike<number>[], x: VectorLike): Predicate[] {
  const xv = dense.toF64(x, 'binPredicates')
  return edges.map((e, j) => {
    let lower = -Infinity
    let upper = Infinity
    for (let k = 0; k < e.length; k++) {
      if (xv[j] <= e[k]) {
        upper = e[k]
        break
      }
      lower = e[k]
    }
    return { feature: j, lower, upper }
  })
}

// ── KL confidence bounds and KL-LUCB ─────────────────────────────────────────────────────────────────────────────────

/**
 * A probability clipped to $[10^{-7}, 1 - 10^{-16}]$, so that its logarithms are finite.
 *
 * @param p The probability.
 * @returns The clipped value.
 */
const clampP = (p: number) => Math.min(1 - 1e-16, Math.max(1e-7, p))

/**
 * The Bernoulli Kullback–Leibler divergence
 * $\KL(p \,\Vert\, q) = p \log(p/q) + (1 - p)\log((1 - p)/(1 - q))$, with both arguments clipped away from 0 and 1.
 *
 * @param p The first Bernoulli mean.
 * @param q The second Bernoulli mean.
 * @returns The divergence, at least 0.
 *
 * @example Zero at equality, growing with the gap
 * print(bernoulliKl(0.5, 0.5), bernoulliKl(0.5, 0.7), bernoulliKl(0.5, 0.9))
 */
export function bernoulliKl(p: number, q: number): number {
  const a = clampP(p)
  const b = clampP(q)
  return a * Math.log(a / b) + (1 - a) * Math.log((1 - a) / (1 - b))
}

/**
 * The KL confidence interval of a Bernoulli mean $\hat p$: the $q$ below and above $\hat p$ with
 * $\KL(\hat p \,\Vert\, q) = \text{level}$, each found by 30 bisection steps.
 *
 * @param p The empirical mean $\hat p$.
 * @param level The KL radius, such as $\beta/n$ for $n$ draws and exploration rate $\beta$.
 * @returns The `lower` and `upper` ends, within $[0, 1]$.
 *
 * @example The interval narrows with more draws
 * print('10 draws:', klBounds(0.8, Math.log(20) / 10))
 * print('1000 draws:', klBounds(0.8, Math.log(20) / 1000))
 */
export function klBounds(p: number, level: number): { lower: number; upper: number } {
  let lo = p
  let hi = Math.min(1, p + Math.sqrt(level / 2))
  for (let j = 0; j < 30; j++) {
    const q = (lo + hi) / 2
    if (bernoulliKl(p, q) > level) hi = q
    else lo = q
  }
  const upper = hi
  hi = p
  lo = Math.max(0, p - Math.sqrt(level / 2))
  for (let j = 0; j < 30; j++) {
    const q = (lo + hi) / 2
    if (bernoulliKl(p, q) > level) lo = q
    else hi = q
  }
  return { lower: lo, upper }
}

/**
 * The exploration rate $\beta(t, \delta) = \log(k_1 K t^\alpha / \delta) + \log\log(k_1 K t^\alpha / \delta)$, with
 * $k_1 = 405.5$ and $\alpha = 1.1$ (as the anchor package).
 *
 * @param arms The number of arms $K$.
 * @param t The round $t$.
 * @param delta The confidence parameter $\delta$.
 * @returns $\beta(t, \delta)$.
 */
const explorationRate = (arms: Size, t: number, delta: number) => {
  const v = Math.log((405.5 * arms * t ** 1.1) / delta)
  return v + Math.log(v)
}

/** Running counts of a Bernoulli arm: its `draws` so far and the `successes` among them. */
export type ArmStats = { draws: number; successes: number }

/**
 * KL-LUCB: identify the `top` arms of largest mean among Bernoulli arms, sampling `batch` draws at a time from the two
 * arms whose KL bounds most overlap the boundary between the top set and the rest, until the gap between the best
 * upper bound outside and the worst lower bound inside is at most $\epsilon$ (or the budget is spent). Arms never
 * drawn get one batch first, outside the budget.
 *
 * @param draw Draws from an arm: `draw(arm, count)` returns the successes in `count` new draws.
 * @param stats The counts of each arm, updated in place, so draws carry across calls.
 * @param options The number of arms wanted, the stopping rule and the sampling.
 * @param options.top The number of arms to identify (default 1, at most the number of arms).
 * @param options.epsilon The tolerance $\epsilon$ on the gap between the bounds (default 0.1).
 * @param options.delta The confidence parameter $\delta$ of the exploration rate (default 0.05).
 * @param options.batch Draws per arm per round (default 10).
 * @param options.budget The most draws spent in the loop (default 20000).
 * @returns `chosen`, the top arms, largest mean first; the `means` and the KL bounds `lower` and `upper` of every
 *   arm; and `rounds`, the rounds run (starting at 1).
 *
 * @example Find the best of three coins
 * const p = [0.2, 0.5, 0.8]
 * const s = stream(0)
 * const draw = (arm, m) => toArray(uniform(s, 0, 1, { shape: [m] })).filter((u) => u < p[arm]).length
 * const stats = p.map(() => ({ draws: 0, successes: 0 }))
 * const r = klLucb(draw, stats, { epsilon: 0.05 })
 * print('chosen =', r.chosen, ' means =', r.means, ' rounds =', r.rounds)
 * print('draws per arm =', stats.map((a) => a.draws))
 */
export function klLucb(
  draw: (arm: number, count: Size) => number,
  stats: ArmStats[],
  options: { top?: Size; epsilon?: number; delta?: number; batch?: Size; budget?: Size } = {},
): { chosen: number[]; means: Float64Array; lower: Float64Array; upper: Float64Array; rounds: Size } {
  const K = stats.length
  const { epsilon = 0.1, delta = 0.05, batch = 10, budget = 20000 } = options
  const top = Math.min(options.top ?? 1, K)
  for (let a = 0; a < K; a++)
    if (stats[a].draws === 0) {
      stats[a].successes += draw(a, batch)
      stats[a].draws += batch
    }
  const means = new Float64Array(K)
  const lower = new Float64Array(K)
  const upper = new Float64Array(K).fill(1)
  let t = 1
  let spent = 0
  const order = () => Array.from({ length: K }, (_, a) => a).sort((a, b) => means[a] - means[b] || b - a)
  const update = () => {
    for (let a = 0; a < K; a++) means[a] = stats[a].successes / stats[a].draws
    const beta = explorationRate(K, t, delta)
    const sorted = order()
    const J = sorted.slice(K - top)
    const notJ = sorted.slice(0, K - top)
    for (const a of [...J, ...notJ]) {
      const b = klBounds(means[a], beta / stats[a].draws)
      lower[a] = b.lower
      upper[a] = b.upper
    }
    if (notJ.length === 0) return null
    const ut = notJ.reduce((best, a) => (upper[a] > upper[best] ? a : best), notJ[0])
    const lt = J.reduce((best, a) => (lower[a] < lower[best] ? a : best), J[0])
    return { ut, lt }
  }
  let pair = update()
  while (pair && upper[pair.ut] - lower[pair.lt] > epsilon && spent < budget) {
    for (const a of [pair.ut, pair.lt]) {
      stats[a].successes += draw(a, batch)
      stats[a].draws += batch
      spent += batch
    }
    t++
    pair = update()
  }
  const chosen = order()
    .slice(K - top)
    .reverse()
  return { chosen, means, lower, upper, rounds: t }
}

// ── Anchors ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `anchor`. */
export type AnchorOptions = {
  /** Precision $\tau$ the anchor must reach (default 0.95). */
  threshold?: number
  /** Confidence parameter $\delta$ (default 0.1). */
  delta?: number
  /** The KL-LUCB tolerance $\epsilon$ (default 0.1). */
  epsilon?: number
  /** Candidates kept per size (default 2). */
  beam?: Size
  /** Draws per sampling batch (default 10). */
  batch?: Size
  /** Largest anchor size (default all features). */
  maxSize?: Size
  /** Quantile bins per feature, taken from the background (default 4). */
  bins?: Size
  /** Explicit inner bin edges per feature, used instead of `bins`. */
  edges?: readonly ArrayLike<number>[]
}

/** A candidate rule and what is known of it. */
export type AnchorCandidate = {
  /** The rule's predicates. */
  rule: Predicate[]
  /** The fraction of its draws that kept $\xvec$'s label. */
  precision: number
  /** The lower end of the precision's KL confidence interval. */
  lower: number
  /** The upper end of the precision's KL confidence interval. */
  upper: number
  /** The fraction of background rows meeting the rule. */
  coverage: number
  /** The perturbed samples drawn for it. */
  draws: Size
}

/** The result of `anchor`: the rule, its estimated precision and coverage, and the beam per size. */
export type AnchorResult = {
  /** The anchor, simplified (one predicate per feature, sorted). */
  rule: Predicate[]
  /** Its estimated precision. */
  precision: number
  /** The fraction of background rows meeting it. */
  coverage: number
  /** Whether the precision constraint was met (otherwise the best-precision rule seen). */
  valid: boolean
  /** The model's label at $\xvec$, the one the anchor keeps. */
  label: number
  /** The beam after each size: the candidates kept, with their statistics. */
  beams: AnchorCandidate[][]
  /** Model evaluations used. */
  evaluations: Size
}

/**
 * An anchor for a classifier at $\xvec$ (see the file comment), with the perturbation distribution built from
 * background rows: a draw takes a random background row and, for each predicate of the rule it breaks, replaces that
 * feature with the value of a random background row meeting the predicate. Coverage is the fraction of background rows
 * meeting the rule. A kept rule is accepted once its mean precision is at least $\tau$ and its lower bound above
 * $\tau - 0.05$ (more draws are taken until the bound settles, up to 200 batches); the search stops at a rule covering
 * every background row. Throws `ShapeError` when the background's width differs from $\xvec$'s.
 *
 * @param predict The classifier: from a batch of rows ($m \times d$ tensor) to a label per row.
 * @param x The instance $\xvec$ ($d$ values).
 * @param background The rows the perturbations and the coverage come from ($b \times d$).
 * @param stream The random stream: each sampling batch draws from its own `child(stream, 'draw', k)`.
 * @param options The precision, confidence, beam, sampling and binning.
 * @returns The anchor, its precision and coverage, whether it met the precision, the label, the beams and the model
 *   evaluations used.
 *
 * @example One condition anchors a threshold rule
 * // Label 1 where x0 > 0.5; x = (0.9, 0.2) is labelled 1 whatever x1 is.
 * const predict = (X) => toArray(X).map(([a]) => (a > 0.5 ? 1 : 0))
 * const background = uniform(stream(1), 0, 1, { shape: [40, 2] })
 * const r = anchor(predict, [0.9, 0.2], background, stream(0))
 * print('rule =', r.rule)
 * print('precision =', r.precision, ' coverage =', r.coverage, ' valid =', r.valid, ' evaluations =', r.evaluations)
 */
export function anchor(
  predict: (X: Tensor) => ArrayLike<number> | Tensor,
  x: VectorLike,
  background: MatrixLike,
  stream: Stream,
  options: AnchorOptions = {},
): AnchorResult {
  const xv = Float64Array.from(dense.toF64(x, 'anchor'))
  const bg = dense.toMatrixF64(background, 'anchor')
  const d = xv.length
  if (bg.n !== d) throw new ShapeError('anchor', 'anchor: the background must have as many features as x')
  const { threshold = 0.95, delta = 0.1, epsilon = 0.1, beam = 2, batch = 10, maxSize = d } = options
  const edges = options.edges ?? quantileEdges(fromData(Float64Array.from(bg.data), [bg.m, d]), options.bins ?? 4)
  const predicates = binPredicates(edges, xv)
  let evaluations = 0
  const labels = (rows: Float64Array, m: Size): ArrayLike<number> => {
    evaluations += m
    const out = predict(fromData(rows, [m, d]))
    return 'shape' in (out as object) && !ArrayBuffer.isView(out) ? toFlat(out as Tensor) : (out as ArrayLike<number>)
  }
  const label = labels(Float64Array.from(xv), 1)[0]
  // Background rows meeting each single predicate (to resample a broken feature within x's interval).
  const meeting = predicates.map((p) => {
    const rows: number[] = []
    for (let i = 0; i < bg.m; i++) if (satisfies(bg.data.subarray(i * d, (i + 1) * d), [p])) rows.push(i)
    return rows
  })
  const coverOf = (rule: number[]) => {
    let c = 0
    for (let i = 0; i < bg.m; i++)
      if (
        satisfies(
          bg.data.subarray(i * d, (i + 1) * d),
          rule.map((k) => predicates[k]),
        )
      )
        c++
    return c / bg.m
  }
  let drawCount = 0
  const sample = (rule: number[], m: Size): number => {
    const s = child(stream, 'draw', drawCount++)
    const rows = new Float64Array(m * d)
    for (let r = 0; r < m; r++) {
      const b = integers(s, bg.m) as number
      rows.set(bg.data.subarray(b * d, (b + 1) * d), r * d)
      for (const k of rule) {
        const p = predicates[k]
        const v = rows[r * d + p.feature]
        if (v > p.lower && v <= p.upper) continue
        const pool = meeting[k]
        rows[r * d + p.feature] =
          pool.length > 0 ? bg.data[pool[integers(s, pool.length) as number] * d + p.feature] : xv[p.feature]
      }
    }
    const y = labels(rows, m)
    let hits = 0
    for (let r = 0; r < m; r++) if (y[r] === label) hits++
    return hits
  }
  const stats = new Map<string, ArmStats>()
  const keyOf = (rule: number[]) => rule.join(',')
  const statOf = (rule: number[]) => {
    const k = keyOf(rule)
    let s = stats.get(k)
    if (!s) {
      s = { draws: 0, successes: 0 }
      stats.set(k, s)
    }
    return s
  }
  const coverage = new Map<string, number>()
  const coverageOf = (rule: number[]) => {
    const k = keyOf(rule)
    if (!coverage.has(k)) coverage.set(k, coverOf(rule))
    return coverage.get(k) as number
  }
  const describe = (rule: number[]): AnchorCandidate => {
    const s = statOf(rule)
    const p = s.draws > 0 ? s.successes / s.draws : 0
    const b = klBounds(p, Math.log(1 / delta) / Math.max(1, s.draws))
    return {
      rule: rule.map((k) => predicates[k]),
      precision: p,
      lower: b.lower,
      upper: b.upper,
      coverage: coverageOf(rule),
      draws: s.draws,
    }
  }
  let previous: number[][] = [[]]
  let best: number[] | null = null
  let bestCoverage = -1
  const beams: AnchorCandidate[][] = []
  const seen: number[][] = []
  const epsilonStop = 0.05
  for (let size = 1; size <= Math.min(maxSize, d); size++) {
    const pool = new Map<string, number[]>()
    for (const rule of previous)
      for (let k = 0; k < d; k++) {
        if (rule.includes(k)) continue
        const r = [...rule, k].sort((a, b) => a - b)
        if (coverageOf(r) <= bestCoverage) continue
        pool.set(keyOf(r), r)
      }
    const candidates = [...pool.values()]
    if (candidates.length === 0) break
    const arms = candidates.map(statOf)
    const res = klLucb((a, m) => sample(candidates[a], m), arms, {
      top: Math.min(beam, candidates.length),
      epsilon,
      delta,
      batch,
    })
    const kept = res.chosen.map((a) => candidates[a])
    seen.push(...candidates)
    let stop = false
    for (const rule of kept) {
      const s = statOf(rule)
      const beta = Math.log(1 / delta)
      let { lower, upper } = klBounds(s.successes / s.draws, beta / s.draws)
      let mean = s.successes / s.draws
      let guard = 0
      while (
        ((mean >= threshold && lower < threshold - epsilonStop) ||
          (mean < threshold && upper >= threshold + epsilonStop)) &&
        guard++ < 200
      ) {
        s.successes += sample(rule, batch)
        s.draws += batch
        mean = s.successes / s.draws
        ;({ lower, upper } = klBounds(mean, beta / s.draws))
      }
      const cov = coverageOf(rule)
      if (mean >= threshold && lower > threshold - epsilonStop && cov > bestCoverage) {
        bestCoverage = cov
        best = rule
        if (cov === 1) stop = true
      }
    }
    beams.push(kept.map(describe))
    if (stop) break
    previous = kept
  }
  let valid = true
  if (!best) {
    valid = false
    best = seen.reduce((a, r) => (describe(r).lower > describe(a).lower ? r : a), seen[0] ?? [])
  }
  const out = describe(best)
  return {
    rule: simplifyRule(out.rule),
    precision: out.precision,
    coverage: out.coverage,
    valid,
    label,
    beams,
    evaluations,
  }
}

// ── Rule lists and tree rules ────────────────────────────────────────────────────────────────────────────────────────

/**
 * One rule of a decision list: its conditions (`rule`), the `label` it gives, and its `support` (the rows it took)
 * and `precision` (the fraction of them with that label) when it was learned.
 */
export type ListRule = { rule: Predicate[]; label: number; support: Size; precision: number }

/** A decision list: the first of `rules` a row meets gives its label, else `defaultLabel`. */
export type RuleList = { rules: ListRule[]; defaultLabel: number }

/** Options of `ruleList`. */
export type RuleListOptions = {
  /** Threshold candidates per feature: the inner edges of this many quantile bins (default 8). */
  bins?: Size
  /** At most this many rules (default 8). */
  maxRules?: Size
  /** At most this many conditions per rule (default 3). */
  maxConditions?: Size
  /** A rule must cover at least this many remaining rows (default 5). */
  minSupport?: Size
  /** Candidates kept per refinement (default 5). */
  beam?: Size
}

/**
 * Learn a decision list by sequential covering (see the file comment). Each rule maximises the Laplace estimate
 * $(\text{correct} + 1)/(\text{covered} + K)$ of its majority class on the rows not yet covered ($K$ classes), over
 * conjunctions of predicates $x_j \le q$ and $x_j > q$ at quantile thresholds $q$; learning stops when the best rule is
 * no purer than the remaining rows' majority, when fewer than `minSupport` rows remain, or after `maxRules` rules.
 * Throws `ShapeError` when there is not one label per row.
 *
 * @param X The rows ($n \times d$).
 * @param y The labels ($n$ values): integers $0, \dots, K - 1$.
 * @param options The thresholds, the list's size, the support and the beam.
 * @returns The list; its default is the majority label of the rows left uncovered (of all rows when none are left).
 *
 * @example Learning an AND
 * // A 4 x 4 grid labelled 1 where both features exceed 0.5.
 * const X = []
 * for (const a of [0.1, 0.3, 0.6, 0.9]) for (const b of [0.1, 0.3, 0.6, 0.9]) X.push([a, b])
 * const y = X.map(([a, b]) => (a > 0.5 && b > 0.5 ? 1 : 0))
 * const condition = (p) => `${p.lower.toFixed(2)} < x${p.feature} <= ${p.upper.toFixed(2)}`
 * const show = (rule) => rule.map(condition).join(' and ')
 * const list = ruleList(X, y, { minSupport: 2, bins: 4 })
 * for (const r of list.rules) print(`if ${show(r.rule)} then ${r.label}`, ' support =', r.support)
 * print('else', list.defaultLabel)
 */
export function ruleList(X: MatrixLike, y: VectorLike, options: RuleListOptions = {}): RuleList {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'ruleList')
  const yv = Int32Array.from(dense.toF64(y, 'ruleList'))
  if (yv.length !== n) throw new ShapeError('ruleList', 'ruleList: one label per row')
  const { bins = 8, maxRules = 8, maxConditions = 3, minSupport = 5, beam = 5 } = options
  const K = Math.max(...yv) + 1
  const edges = quantileEdges(fromData(Float64Array.from(data), [n, d]), bins)
  const atoms: Predicate[] = []
  edges.forEach((e, j) =>
    e.forEach((q) => atoms.push({ feature: j, lower: -Infinity, upper: q }, { feature: j, lower: q, upper: Infinity })),
  )
  const row = (i: number) => data.subarray(i * d, (i + 1) * d)
  const majority = (rows: number[]) => {
    const c = new Array<number>(K).fill(0)
    for (const i of rows) c[yv[i]]++
    let best = 0
    for (let k = 1; k < K; k++) if (c[k] > c[best]) best = k
    return { label: best, correct: c[best] }
  }
  let remaining = Array.from({ length: n }, (_, i) => i)
  const rules: ListRule[] = []
  while (rules.length < maxRules && remaining.length >= minSupport) {
    const base = majority(remaining)
    const baseScore = (base.correct + 1) / (remaining.length + K)
    type Cand = { rule: Predicate[]; rows: number[]; label: number; score: number; correct: number }
    const score = (rule: Predicate[]): Cand | null => {
      const rows = remaining.filter((i) => satisfies(row(i), rule))
      if (rows.length < minSupport) return null
      const mj = majority(rows)
      return { rule, rows, label: mj.label, correct: mj.correct, score: (mj.correct + 1) / (rows.length + K) }
    }
    let frontier: Cand[] = [{ rule: [], rows: remaining, label: base.label, correct: base.correct, score: baseScore }]
    let best: Cand | null = null
    for (let depth = 0; depth < maxConditions; depth++) {
      const next = new Map<string, Cand>()
      for (const f of frontier)
        for (const a of atoms) {
          const r = simplifyRule([...f.rule, a])
          if (r.some((p) => !(p.lower < p.upper))) continue
          const key = JSON.stringify(r)
          if (next.has(key)) continue
          const c = score(r)
          if (c) next.set(key, c)
        }
      const ranked = [...next.values()].sort((a, b) => b.score - a.score || b.rows.length - a.rows.length)
      if (ranked.length === 0) break
      if (!best || ranked[0].score > best.score) best = ranked[0]
      frontier = ranked.slice(0, beam)
    }
    if (!best || best.score <= baseScore + 1e-12) break
    rules.push({
      rule: best.rule,
      label: best.label,
      support: best.rows.length,
      precision: best.correct / best.rows.length,
    })
    const taken = new Set(best.rows)
    remaining = remaining.filter((i) => !taken.has(i))
  }
  const defaultLabel =
    remaining.length > 0 ? majority(remaining).label : majority(Array.from({ length: n }, (_, i) => i)).label
  return { rules, defaultLabel }
}

/**
 * The label a decision list gives each row, and which rule gave it.
 *
 * @param list The decision list.
 * @param X The rows ($m \times d$).
 * @returns `labels`, one per row, and `fired`, the index of the rule that gave it ($-1$ for the default).
 *
 * @example Which rule fires where
 * // A 4 x 4 grid labelled 1 where both features exceed 0.5.
 * const X = []
 * for (const a of [0.1, 0.3, 0.6, 0.9]) for (const b of [0.1, 0.3, 0.6, 0.9]) X.push([a, b])
 * const y = X.map(([a, b]) => (a > 0.5 && b > 0.5 ? 1 : 0))
 * const list = ruleList(X, y, { minSupport: 2, bins: 4 })
 * print(applyRuleList(list, [[0.8, 0.8], [0.8, 0.2], [0.2, 0.9]]))
 */
export function applyRuleList(list: RuleList, X: MatrixLike): { labels: Int32Array; fired: Int32Array } {
  const { data, m, n } = dense.toMatrixF64(X, 'applyRuleList')
  const labels = new Int32Array(m)
  const fired = new Int32Array(m).fill(-1)
  for (let i = 0; i < m; i++) {
    const r = data.subarray(i * n, (i + 1) * n)
    const k = list.rules.findIndex((rule) => satisfies(r, rule.rule))
    fired[i] = k
    labels[i] = k >= 0 ? list.rules[k].label : list.defaultLabel
  }
  return { labels, fired }
}

/**
 * The rules of a tree in the `ShapTree` layout: for each leaf, in depth-first order with the left child first, its
 * path's predicates merged per feature (a left branch gives $x_j \le \text{threshold}$, a right one
 * $x_j > \text{threshold}$), its value vector, its cover and its node index.
 *
 * @param tree The tree.
 * @returns One rule per leaf.
 *
 * @example The three leaves of an AND tree
 * const leaf = (weight, v) => ({ feature: -1, threshold: 0, weight, value: [v], children: [] })
 * const tree = {
 *   root: 0,
 *   nodes: [
 *     { feature: 0, threshold: 0.5, weight: 4, value: [0.25], children: [1, 2] },
 *     leaf(2, 0),
 *     { feature: 1, threshold: 0.5, weight: 2, value: [0.5], children: [3, 4] },
 *     leaf(1, 0),
 *     leaf(1, 1),
 *   ],
 * }
 * const condition = (p) => `${p.lower.toFixed(2)} < x${p.feature} <= ${p.upper.toFixed(2)}`
 * const show = (rule) => rule.map(condition).join(' and ')
 * for (const r of treeRules(tree)) print(show(r.rule), ' value =', r.value, ' cover =', r.cover)
 */
export function treeRules(
  tree: ShapTree,
): { rule: Predicate[]; value: readonly number[]; cover: number; leaf: number }[] {
  const out: { rule: Predicate[]; value: readonly number[]; cover: number; leaf: number }[] = []
  const visit = (j: number, rule: Predicate[]) => {
    const node = tree.nodes[j]
    if (node.children.length === 0) {
      out.push({ rule: simplifyRule(rule), value: node.value, cover: node.weight, leaf: j })
      return
    }
    const [left, right] = node.children
    visit(left, [...rule, { feature: node.feature, lower: -Infinity, upper: node.threshold }])
    visit(right, [...rule, { feature: node.feature, lower: node.threshold, upper: Infinity }])
  }
  visit(tree.root, [])
  return out
}

/**
 * Fidelity of a surrogate to a model on the same rows. Throws `ShapeError` when the inputs differ in length or are
 * empty.
 *
 * @param model The model's labels or scores, one per row.
 * @param surrogate The surrogate's, one per row.
 * @param kind `'agreement'`, the fraction of equal labels, or `'r2'`, the $R^2$ of the surrogate's scores against the
 *   model's (1 for an exact match of constant scores, 0 for a mismatch).
 * @returns The fidelity.
 *
 * @example Labels and scores
 * print('agreement:', fidelity([1, 0, 1, 1], [1, 0, 0, 1]))
 * print('r2:', fidelity([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.9], 'r2'))
 */
export function fidelity(model: VectorLike, surrogate: VectorLike, kind: 'agreement' | 'r2' = 'agreement'): number {
  const a = dense.toF64(model, 'fidelity')
  const b = dense.toF64(surrogate, 'fidelity')
  if (a.length !== b.length || a.length === 0)
    throw new ShapeError('fidelity', 'fidelity: need equal, non-empty inputs')
  if (kind === 'agreement') {
    let same = 0
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++
    return same / a.length
  }
  let mean = 0
  for (let i = 0; i < a.length; i++) mean += a[i] / a.length
  let ssr = 0
  let sst = 0
  for (let i = 0; i < a.length; i++) {
    ssr += (a[i] - b[i]) ** 2
    sst += (a[i] - mean) ** 2
  }
  return sst > 0 ? 1 - ssr / sst : ssr === 0 ? 1 : 0
}
