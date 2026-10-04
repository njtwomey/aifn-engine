/**
 * Rule-based explanations. A rule is a conjunction of interval predicates lower < xⱼ ≤ upper.
 *
 * - `anchor` (Ribeiro, Singh and Guestrin, 2018): a rule that holds at x and "anchors" the prediction: with the rule's
 *   features drawn within its intervals and the rest from the data, the model keeps x's label with precision ≥ τ (with
 *   confidence 1 − δ). Among such rules it seeks the one covering most of the data. Predicates are x's quantile bin
 *   per feature; rules grow one predicate at a time by beam search, the best `beam` candidates of each size chosen by
 *   KL-LUCB (Kaufmann and Kalyanakrishnan, 2013), a best-arm identification bandit with KL confidence bounds.
 * - `ruleList`: a decision list learned by sequential covering (Rivest, 1987; Clark and Niblett, 1989): repeatedly the
 *   conjunction (by beam search over threshold predicates) whose covered rows are most purely one class (Laplace
 *   estimate), then remove the rows it covers; a default class ends the list. Fitted to a black box's predictions it is
 *   a global surrogate.
 * - `treeRules`: the root-to-leaf rules of a tree; `fidelity`: how closely a surrogate reproduces a model (agreement
 *   for labels, R² for scores) (Craven and Shavlik, 1996).
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { child, integers, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { quantile } from 'aifn-compute/probability/stats'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { ShapTree } from './tree'

/** lower < x[feature] ≤ upper (either end may be infinite). */
export type Predicate = { feature: Size; lower: number; upper: number }

/** True when the row meets every predicate. */
export function satisfies(row: ArrayLike<number>, rule: readonly Predicate[]): boolean {
  for (const p of rule) {
    const v = row[p.feature]
    if (!(v > p.lower && v <= p.upper)) return false
  }
  return true
}

/** The fraction of rows of X [n, d] that meet the rule. */
export function ruleCoverage(X: MatrixLike, rule: readonly Predicate[]): number {
  const { data, m, n } = dense.toMatrixF64(X, 'ruleCoverage')
  let c = 0
  for (let i = 0; i < m; i++) if (satisfies(data.subarray(i * n, (i + 1) * n), rule)) c++
  return m > 0 ? c / m : 0
}

/** A rule with intervals on the same feature intersected, sorted by feature. */
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

/** Inner bin edges per feature: the distinct j/bins quantiles (j = 1 … bins − 1) of each column of X [n, d]. */
export function quantileEdges(X: MatrixLike, bins = 4): Float64Array[] {
  if (!(Number.isInteger(bins) && bins >= 2)) throw new DomainError('quantileEdges', 'quantileEdges: bins must be ≥ 2')
  const { data, m, n } = dense.toMatrixF64(X, 'quantileEdges')
  return Array.from({ length: n }, (_, j) => {
    const col = Array.from({ length: m }, (_, i) => data[i * n + j])
    const qs = Array.from({ length: bins - 1 }, (_, k) => quantile(col, (k + 1) / bins))
    return Float64Array.from(new Set(qs))
  })
}

/** x's bin on each feature as a predicate, from inner edges per feature. */
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

const clampP = (p: number) => Math.min(1 - 1e-16, Math.max(1e-7, p))

/** The Bernoulli Kullback–Leibler divergence KL(p ‖ q) (arguments clipped away from 0 and 1). */
export function bernoulliKl(p: number, q: number): number {
  const a = clampP(p)
  const b = clampP(q)
  return a * Math.log(a / b) + (1 - a) * Math.log((1 - a) / (1 - b))
}

/** The KL confidence interval of a Bernoulli mean p̂: the q below and above p̂ with KL(p̂ ‖ q) = level (bisection). */
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

/** The exploration rate β(t, δ) = log(k₁ K t^α / δ) + log log(…), k₁ = 405.5, α = 1.1 (as the anchor package). */
const explorationRate = (arms: Size, t: number, delta: number) => {
  const v = Math.log((405.5 * arms * t ** 1.1) / delta)
  return v + Math.log(v)
}

/** Running counts of a Bernoulli arm. */
export type ArmStats = { draws: number; successes: number }

/**
 * KL-LUCB: identify the `top` arms of largest mean among Bernoulli arms, sampling `batch` draws at a time from the two
 * arms whose KL bounds most overlap the boundary between the top set and the rest, until the gap between the best
 * upper bound outside and the worst lower bound inside is ≤ ε (or `budget` draws). `draw(arm, count)` returns the
 * successes in `count` new draws; `stats` (one per arm) is updated in place, so draws carry across calls.
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
  /** Precision τ the anchor must reach (default 0.95). */
  threshold?: number
  /** Confidence parameter δ (default 0.1) and the KL-LUCB tolerance ε (default 0.1). */
  delta?: number
  epsilon?: number
  /** Candidates kept per size (default 2). */
  beam?: Size
  /** Draws per sampling batch (default 10). */
  batch?: Size
  /** Largest anchor size (default all features). */
  maxSize?: Size
  /** Quantile bins per feature (default 4), or explicit inner edges per feature. */
  bins?: Size
  edges?: readonly ArrayLike<number>[]
}

/** A candidate rule and what is known of it. */
export type AnchorCandidate = {
  rule: Predicate[]
  precision: number
  lower: number
  upper: number
  coverage: number
  draws: Size
}

/** The result of `anchor`: the rule, its estimated precision and coverage, and the beam per size. */
export type AnchorResult = {
  rule: Predicate[]
  precision: number
  coverage: number
  /** Whether the precision constraint was met (otherwise the best-precision rule seen). */
  valid: boolean
  label: number
  /** The beam after each size: the candidates kept, with their statistics. */
  beams: AnchorCandidate[][]
  /** Model evaluations used. */
  evaluations: Size
}

/**
 * An anchor for the classifier `predict` (a label per row of [m, d]) at x [d], with the perturbation distribution
 * built from `background` rows [b, d]: a draw takes a random background row and, for each predicate of the rule it
 * breaks, replaces that feature with the value of a random background row meeting the predicate. Coverage is the
 * fraction of background rows meeting the rule.
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

/** One rule of a decision list: its conditions, the label it gives, and its support and precision on the rows it took. */
export type ListRule = { rule: Predicate[]; label: number; support: Size; precision: number }

/** A decision list: the first rule a row meets gives its label, else the default. */
export type RuleList = { rules: ListRule[]; defaultLabel: number }

/** Options of `ruleList`. */
export type RuleListOptions = {
  /** Threshold candidates per feature: the inner edges of this many quantile bins (default 8). */
  bins?: Size
  /** At most this many rules (default 8) and conditions per rule (default 3). */
  maxRules?: Size
  maxConditions?: Size
  /** A rule must cover at least this many remaining rows (default 5). */
  minSupport?: Size
  /** Candidates kept per refinement (default 5). */
  beam?: Size
}

/**
 * Learn a decision list from rows X [n, d] and integer labels y [n] by sequential covering (see the module comment).
 * Each rule maximises the Laplace estimate (correct + 1)/(covered + K) of its majority class on the rows not yet
 * covered, over conjunctions of predicates xⱼ ≤ q and xⱼ > q at quantile thresholds q; learning stops when the best
 * rule is no purer than the remaining rows' majority.
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

/** The label a decision list gives each row of X [m, d] and the index of the rule that fired (−1 for the default). */
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

/** The rules of a tree (`ShapTree` layout): per leaf, its path's predicates (merged per feature), value and cover. */
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
 * Fidelity of a surrogate to a model on the same rows: the fraction of equal labels (`agreement`, default) or the R²
 * of the surrogate's scores against the model's (`r2`).
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
