/**
 * The candidate conditions a rule learner refines with, and the bookkeeping every learner of this module shares.
 *
 * A condition is the interval `Predicate` $\text{lower} < x_j \le \text{upper}$ of `aifn-compute/learning/explain`.
 * Numeric features give the threshold pairs $x_j \le q$ and $x_j > q$ at the inner quantile edges $q$ of each feature
 * (`quantileEdges`, as `ruleList` uses them). Nominal features coded $0, \dots, L - 1$ give the equality tests
 * $x_j = v$, written as the interval $v - \frac12 < x_j \le v + \frac12$, so one predicate type serves both and
 * `satisfies` and `simplifyRule` apply unchanged. A rule's cover is a mask (a `Uint8Array` of 0 and 1, one entry per
 * row), evaluated once by `coverMask` and combined by `andMask`; `ruleCounts` reads the positive and negative counts of
 * a mask within an active subset of the rows, which is what a covering loop needs after rows have been removed.
 * Antecedents (frequent conjunctions) are mined level-wise by `mineAntecedents`, as in Apriori (Agrawal and Srikant,
 * 1994), and shared by Bayesian rule lists, associative classification and CORELS. A list of rules is applied by
 * `applyClassRules`: the first rule a row meets gives its class.
 */

import { quantileEdges, satisfies, type Predicate } from 'aifn-compute/learning/explain'

/** Rows of numbers, one per training example, each indexed by feature. */
export type RuleRows = readonly ArrayLike<number>[]

/** A classification rule: a conjunction of conditions and the class it predicts. */
export type ClassRule = {
  /** The conditions, all of which a row must meet. */
  conditions: Predicate[]
  /** The class predicted for a row that meets them. */
  label: number
}

/**
 * Threshold conditions $x_j \le q$ and $x_j > q$ at each feature's inner quantile edges (linear interpolation, as
 * `quantileEdges`). Throws `DomainError` when `bins` is not an integer of at least 2.
 *
 * @param rows The data: $n$ rows of $d$ values.
 * @param bins The quantile bins per feature, so at most $2(\text{bins} - 1)$ conditions per feature.
 * @returns The conditions, feature by feature, each $x_j \le q$ before its $x_j > q$.
 *
 * @example Two conditions per inner quartile edge
 * const rows = [[1, 0], [2, 0], [3, 1], [4, 1]]
 * for (const c of thresholdConditions(rows, 2)) print(ruleText([c]))
 */
export function thresholdConditions(rows: RuleRows, bins = 8): Predicate[] {
  const edges = quantileEdges(
    rows.map((r) => Array.from(r)),
    bins,
  )
  const out: Predicate[] = []
  edges.forEach((qs, feature) => {
    for (const q of qs) {
      out.push({ feature, lower: -Infinity, upper: q })
      out.push({ feature, lower: q, upper: Infinity })
    }
  })
  return out
}

/**
 * Equality conditions $x_j = v$ for nominal features coded as integers, each written as the interval
 * $(v - \frac12, v + \frac12]$.
 *
 * @param levels The number of levels $L_j$ of each nominal feature (feature $j$ takes the values
 *   $0, \dots, L_j - 1$).
 * @param features The feature index of each entry of `levels`; left out, entry $k$ is feature $k$.
 * @returns One condition per feature and level, feature by feature.
 *
 * @example The Titanic coding: class (4 levels), sex (2), age (2)
 * const conditions = nominalConditions([4, 2, 2])
 * print('conditions:', conditions.length)
 * print(ruleText([conditions[4]], { names: ['class', 'sex', 'age'], levels: { 1: ['male', 'female'] } }))
 */
export function nominalConditions(levels: readonly number[], features?: readonly number[]): Predicate[] {
  const out: Predicate[] = []
  levels.forEach((L, k) => {
    const feature = features?.[k] ?? k
    for (let v = 0; v < L; v++) out.push({ feature, lower: v - 0.5, upper: v + 0.5 })
  })
  return out
}

/**
 * The rows that meet a conjunction, as a mask: 1 where every condition holds (the empty rule covers every row).
 *
 * @param rows The rows to test.
 * @param rule The conditions.
 * @returns One entry per row, 1 when the row meets the rule and 0 otherwise.
 *
 * @example The rows with x > 2
 * print(coverMask([[1], [2], [3], [4]], [{ feature: 0, lower: 2, upper: Infinity }]))
 */
export function coverMask(rows: RuleRows, rule: readonly Predicate[]): Uint8Array {
  const m = new Uint8Array(rows.length)
  for (let i = 0; i < rows.length; i++) m[i] = satisfies(rows[i], rule) ? 1 : 0
  return m
}

/**
 * The elementwise AND of two masks of equal length: the cover of the conjunction of their rules.
 *
 * @param a The first mask.
 * @param b The second mask, as long as `a`.
 * @returns A new mask, 1 where both are 1.
 *
 * @example Two covers intersected
 * print(andMask(Uint8Array.from([1, 1, 0, 1]), Uint8Array.from([0, 1, 1, 1])))
 */
export function andMask(a: Uint8Array, b: Uint8Array): Uint8Array {
  const m = new Uint8Array(a.length)
  for (let i = 0; i < a.length; i++) m[i] = a[i] & b[i]
  return m
}

/**
 * The counts a rule is scored by, restricted to the active rows: $p$ and $n$, the positive and negative rows it
 * covers, and the totals $P$ and $N$.
 *
 * @param mask Which rows the rule covers.
 * @param positive Which rows are positive (of the class the rule predicts), truthy for positive.
 * @param active Which rows are still in play; left out, every row.
 * @returns `p` and `n` covered, `P` and `N` in all, among the active rows.
 *
 * @example A rule covering three of four active rows
 * const mask = Uint8Array.from([1, 1, 1, 0, 1])
 * print(ruleCounts(mask, [1, 1, 0, 0, 1], Uint8Array.from([1, 1, 1, 1, 0])))
 */
export function ruleCounts(
  mask: Uint8Array,
  positive: ArrayLike<boolean | number>,
  active?: Uint8Array,
): { p: number; n: number; P: number; N: number } {
  let p = 0
  let n = 0
  let P = 0
  let N = 0
  for (let i = 0; i < mask.length; i++) {
    if (active && !active[i]) continue
    const pos = Boolean(positive[i])
    if (pos) P++
    else N++
    if (mask[i]) {
      if (pos) p++
      else n++
    }
  }
  return { p, n, P, N }
}

/**
 * A conjunction as text, such as `x1 ≤ 0.50 ∧ x2 > 0.10`, with feature names and the names of nominal levels when
 * given. An interval on a feature with level names prints as `name = level` (or `name ∈ {…}` when it spans several
 * levels).
 *
 * @param rule The conditions.
 * @param options `names`, one per feature (default `x1`, `x2`, ...); `levels`, the level names of nominal features,
 *   keyed by feature; `digits`, the decimals of thresholds (default 2).
 * @returns The text; `true` for the empty rule.
 *
 * @example Thresholds and a named level
 * const rule = [{ feature: 0, lower: -Infinity, upper: 1.5 }, { feature: 1, lower: 0.5, upper: 1.5 }]
 * print(ruleText(rule))
 * print(ruleText(rule, { names: ['class', 'sex'], levels: { 1: ['male', 'female'] } }))
 */
export function ruleText(
  rule: readonly Predicate[],
  options: { names?: readonly string[]; levels?: Readonly<Record<number, readonly string[]>>; digits?: number } = {},
): string {
  if (rule.length === 0) return 'true'
  const digits = options.digits ?? 2
  const f = (v: number) => v.toFixed(digits)
  return rule
    .map((c) => {
      const name = options.names?.[c.feature] ?? `x${c.feature + 1}`
      const lv = options.levels?.[c.feature]
      if (lv && Number.isFinite(c.lower) && Number.isFinite(c.upper)) {
        const values = lv.filter((_, v) => v > c.lower && v <= c.upper)
        return values.length === 1 ? `${name} = ${values[0]}` : `${name} ∈ {${values.join(', ')}}`
      }
      if (c.lower === -Infinity) return `${name} ≤ ${f(c.upper)}`
      if (c.upper === Infinity) return `${name} > ${f(c.lower)}`
      return `${f(c.lower)} < ${name} ≤ ${f(c.upper)}`
    })
    .join(' ∧ ')
}

/**
 * Apply an ordered list of rules: the first rule a row meets gives its class, and a row no rule meets gets the
 * default. Every learner of the module that returns a decision list is applied by it.
 *
 * @param rules The rules, in order.
 * @param defaultLabel The class of a row no rule meets.
 * @param rows The rows to classify.
 * @returns One class per row.
 *
 * @example A two-rule list
 * const rules = [
 *   { conditions: [{ feature: 0, lower: 2, upper: Infinity }], label: 1 },
 *   { conditions: [{ feature: 1, lower: 0.5, upper: Infinity }], label: 2 },
 * ]
 * print(applyClassRules(rules, 0, [[3, 0], [1, 1], [1, 0]]))
 */
export function applyClassRules(rules: readonly ClassRule[], defaultLabel: number, rows: RuleRows): Int32Array {
  const masks = rules.map((r) => coverMask(rows, r.conditions))
  const out = new Int32Array(rows.length)
  for (let i = 0; i < rows.length; i++) {
    const k = masks.findIndex((m) => m[i])
    out[i] = k >= 0 ? rules[k].label : defaultLabel
  }
  return out
}

/** A pre-mined antecedent: a frequent conjunction, the rows it covers and its support. */
export type Antecedent = {
  /** The conditions of the conjunction, in the order of the items they came from. */
  conditions: Predicate[]
  /** The rows the conjunction covers, one entry per row. */
  mask: Uint8Array
  /** The fraction of rows it covers. */
  support: number
}

/**
 * Every conjunction of at most `maxLength` single conditions whose support is at least `minSupport`, level by level
 * as in Apriori: a conjunction of length $k$ is tried only when each of its subsets of length $k - 1$ is frequent. Two
 * conditions on one feature are never combined.
 *
 * @param rows The training rows.
 * @param items The single conditions (nominal equalities or thresholds).
 * @param options `minSupport`, the least support as a fraction of the rows (default 0.01); `maxLength`, the most
 *   conditions in a conjunction (default 2).
 * @returns The antecedents, shortest first, then in the order of `items`.
 *
 * @example Frequent pairs of three binary features
 * const rows = [[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]]
 * const found = mineAntecedents(rows, nominalConditions([2, 2, 2]), { minSupport: 0.4 })
 * for (const a of found) print(ruleText(a.conditions), ' support', a.support)
 */
export function mineAntecedents(
  rows: RuleRows,
  items: readonly Predicate[],
  options: { minSupport?: number; maxLength?: number } = {},
): Antecedent[] {
  const minSupport = options.minSupport ?? 0.01
  const maxLength = options.maxLength ?? 2
  const n = rows.length
  const masks = items.map((c) => coverMask(rows, [c]))
  const support = (m: Uint8Array) => {
    let s = 0
    for (let i = 0; i < n; i++) s += m[i]
    return s / n
  }
  let level: { idx: number[]; mask: Uint8Array }[] = []
  items.forEach((_, j) => {
    if (support(masks[j]) >= minSupport) level.push({ idx: [j], mask: masks[j] })
  })
  const out: Antecedent[] = level.map((e) => ({
    conditions: e.idx.map((j) => items[j]),
    mask: e.mask,
    support: support(e.mask),
  }))
  for (let len = 2; len <= maxLength; len++) {
    const next: { idx: number[]; mask: Uint8Array }[] = []
    const frequent = new Set(level.map((e) => e.idx.join(',')))
    for (const e of level) {
      const last = e.idx[e.idx.length - 1]
      for (let j = last + 1; j < items.length; j++) {
        if (e.idx.some((k) => items[k].feature === items[j].feature)) continue
        const idx = [...e.idx, j]
        // Apriori: every subset one shorter must be frequent.
        if (!idx.every((_, drop) => frequent.has(idx.filter((__, k) => k !== drop).join(',')))) continue
        const mask = andMask(e.mask, masks[j])
        if (support(mask) >= minSupport) next.push({ idx, mask })
      }
    }
    for (const e of next) out.push({ conditions: e.idx.map((j) => items[j]), mask: e.mask, support: support(e.mask) })
    level = next
  }
  return out
}

/**
 * A canonical key of a conjunction, independent of the order of its conditions: used to find repeated rules.
 *
 * @param conditions The conditions.
 * @returns The sorted `feature:lower:upper` strings joined by `|`.
 */
export function conditionsKey(conditions: readonly Predicate[]): string {
  return conditions
    .map((c) => `${c.feature}:${c.lower}:${c.upper}`)
    .sort()
    .join('|')
}
