/**
 * C4.5rules: turn a decision tree into an ordered rule set (Quinlan, 1987, "Generating production rules from decision
 * trees", IJCAI; Quinlan, 1993, "C4.5: Programs for Machine Learning", chapter 5), with a trace of every step.
 *
 * Every root-to-leaf path is a conjunctive rule predicting its leaf's class (`treeRules` of
 * `aifn-compute/learning/explain`). Each rule is generalised by deleting conditions: a rule covering $N$ training
 * rows with $E$ errors has the pessimistic error $U_{CF}(E, N)$, the upper end of a one-sided binomial confidence
 * interval at level $CF$ (25% by default), the $q$ with $\Pr(\text{Binomial}(N, q) \le E) = CF$. Greedily, the
 * condition whose deletion gives the smallest pessimistic error is deleted, while that error is no larger than the
 * rule's current one. Duplicate rules are merged. For each class, the subset of its rules that minimises a description
 * length (bits for the rules plus bits for the false positives and false negatives the subset makes on the training
 * rows) is kept: exhaustively for up to 12 rules, by greedy deletion beyond. The classes' subsets are ordered by their
 * false positives, fewest first, and the default class is the most frequent class among the training rows no rule
 * covers.
 *
 * Differences from C4.5: the pessimistic bound is the exact binomial quantile, the Clopper-Pearson upper limit
 * $I^{-1}_{1 - CF}(E + 1, N - E)$ (C4.5's code uses a normal approximation for $E > 0$); the description length uses
 * $\log_2$ of the number of candidate conditions per condition, minus $\log_2 k!$ for the order of $k$ conditions and
 * of the rules, with the exceptions coded as $\log_2\binom{C}{f_p} + \log_2\binom{U}{f_n}$, without C4.5's weighting
 * of the theory bits; rules are not dropped against the default class afterwards.
 */

import type { Predicate } from 'aifn-compute/learning/explain'
import { logChoose, logFactorial, regularisedBetaInverse } from 'aifn-compute/numerics/special'
import { coverMask, conditionsKey, type ClassRule, type RuleRows } from './conditions'

/**
 * The pessimistic error $U_{CF}(E, N)$: the $q$ at which $E$ or fewer errors in $N$ trials have probability $CF$,
 * i.e. the Clopper-Pearson upper limit $I^{-1}_{1 - CF}(E + 1, N - E)$ (`regularisedBetaInverse`), and
 * $1 - CF^{1/N}$ when $E = 0$.
 *
 * @param E The errors, $0 \le E \le N$.
 * @param N The covered rows ($N = 0$ gives 1).
 * @param cf The confidence level $CF \in (0, 1)$.
 * @returns The bound, in $[E/N, 1]$ (1 when $E \ge N$).
 *
 * @example Quinlan's examples (C4.5, chapter 4)
 * print('U(0, 6) =', pessimisticError(0, 6))
 * print('U(0, 1) =', pessimisticError(0, 1))
 * print('U(1, 16) =', pessimisticError(1, 16))
 */
export function pessimisticError(E: number, N: number, cf = 0.25): number {
  if (N <= 0) return 1
  if (E >= N) return 1
  if (E === 0) return 1 - Math.pow(cf, 1 / N)
  return regularisedBetaInverse(E + 1, N - E, 1 - cf)
}

/** The covered rows $N$, the errors $E$ among them and the pessimistic error $U$ of a rule. */
export type RuleError = {
  /** The rows the rule covers. */
  N: number
  /** The covered rows not of the rule's class. */
  E: number
  /** The pessimistic error $U_{CF}(E, N)$. */
  U: number
}

/**
 * The counts and pessimistic error of a rule on the rows.
 *
 * @param rows The rows.
 * @param y The class of each row.
 * @param rule The rule.
 * @param cf The confidence level $CF$.
 * @returns $N$, $E$ and $U$.
 *
 * @example A rule with one error in five rows
 * const rows = [[0], [1], [1], [1], [1], [1]]
 * const rule = { conditions: [{ feature: 0, lower: 0.5, upper: Infinity }], label: 1 }
 * print(ruleError(rows, [0, 1, 1, 1, 1, 0], rule))
 */
export function ruleError(rows: RuleRows, y: ArrayLike<number>, rule: ClassRule, cf = 0.25): RuleError {
  const m = coverMask(rows, rule.conditions)
  let N = 0
  let E = 0
  for (let i = 0; i < rows.length; i++)
    if (m[i]) {
      N++
      if (y[i] !== rule.label) E++
    }
  return { N, E, U: pessimisticError(E, N, cf) }
}

/** One deletion while generalising a rule. */
export type Deletion = {
  /** The condition deleted. */
  dropped: Predicate
  /** The rule's counts and pessimistic error before the deletion. */
  before: RuleError
  /** The same after it. */
  after: RuleError
  /** The conditions left. */
  conditions: Predicate[]
}

/**
 * Generalise a rule by deleting, one at a time, the condition whose deletion gives the smallest pessimistic error,
 * while that error does not rise.
 *
 * @param rows The training rows.
 * @param y Their classes.
 * @param rule The rule; not modified.
 * @param cf The confidence level $CF$.
 * @returns The generalised rule and its deletions in order.
 *
 * @example A redundant condition is dropped
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 0]]
 * const y = [0, 0, 1, 1, 1, 1]
 * const gt = (feature) => ({ feature, lower: 0.5, upper: Infinity })
 * const rule = { conditions: [gt(0), gt(1)], label: 1 }
 * print(ruleText(generaliseRule(rows, y, rule).rule.conditions))
 */
export function generaliseRule(
  rows: RuleRows,
  y: ArrayLike<number>,
  rule: ClassRule,
  cf = 0.25,
): { rule: ClassRule; deletions: Deletion[] } {
  let conds = rule.conditions.slice()
  let cur = ruleError(rows, y, { conditions: conds, label: rule.label }, cf)
  const deletions: Deletion[] = []
  while (conds.length > 0) {
    let best = -1
    let bestErr: RuleError | null = null
    for (let j = 0; j < conds.length; j++) {
      const without = conds.filter((_, k) => k !== j)
      const e = ruleError(rows, y, { conditions: without, label: rule.label }, cf)
      if (!bestErr || e.U < bestErr.U) {
        best = j
        bestErr = e
      }
    }
    if (best < 0 || !bestErr || bestErr.U > cur.U + 1e-12) break
    const dropped = conds[best]
    conds = conds.filter((_, k) => k !== best)
    deletions.push({ dropped, before: cur, after: bestErr, conditions: conds.slice() })
    cur = bestErr
  }
  return { rule: { conditions: conds, label: rule.label }, deletions }
}

/** A step of `c45Rules`. */
export type TreeRulesStep =
  | {
      /** The rules read off the tree's paths. */
      kind: 'paths'
      /** The rules. */
      rules: ClassRule[]
    }
  | {
      /** One condition deleted from one rule. */
      kind: 'generalise'
      /** The index of the rule. */
      index: number
      /** The deletion. */
      deletion: Deletion
      /** Every rule after it. */
      rules: ClassRule[]
    }
  | {
      /** Duplicate rules merged. */
      kind: 'merge'
      /** How many were removed. */
      removed: number
      /** The rules left. */
      rules: ClassRule[]
    }
  | {
      /** One class's rule subset chosen. */
      kind: 'select'
      /** The class. */
      label: number
      /** Indices into `rules` of the class's rules kept. */
      kept: number[]
      /** Indices into `rules` of the class's rules dropped. */
      dropped: number[]
      /** The description length of the kept subset, in bits. */
      bits: number
      /** The merged rules the indices refer to. */
      rules: ClassRule[]
    }
  | {
      /** The final order. */
      kind: 'order'
      /** The rules in order. */
      rules: ClassRule[]
      /** The default class. */
      defaultLabel: number
      /** The false positives of each class's subset, by class. */
      falsePositives: Record<number, number>
    }

/** What `c45Rules` returns. */
export type TreeRulesResult = {
  /** The rules in order. */
  rules: ClassRule[]
  /** The class of a row no rule covers. */
  defaultLabel: number
  /** Every step. */
  trace: TreeRulesStep[]
}

/**
 * C4.5rules (see the file comment): generalise each path rule, merge duplicates, choose each class's subset by
 * description length and order the classes by false positives. Apply the result with `applyClassRules`.
 *
 * @param rows The training rows.
 * @param y Their classes, $0, \dots, K - 1$.
 * @param paths The tree's rules, one per leaf (from `treeRules`, each labelled by its leaf's majority class).
 * @param options `cf`, the confidence level (default 0.25); `candidates`, the number of candidate conditions per
 *   condition for the description length (default the distinct conditions in `paths`, at least 2).
 * @returns The ordered rules, the default class and the trace.
 *
 * @example The four leaves of an AND tree become two rules
 * const rows = []
 * for (const a of [0, 1]) for (const b of [0, 1]) for (let k = 0; k < 5; k++) rows.push([a, b])
 * const y = rows.map(([a, b]) => a & b)
 * const le = (f) => ({ feature: f, lower: -Infinity, upper: 0.5 })
 * const gt = (f) => ({ feature: f, lower: 0.5, upper: Infinity })
 * const paths = [
 *   { conditions: [le(0)], label: 0 },
 *   { conditions: [gt(0), le(1)], label: 0 },
 *   { conditions: [gt(0), gt(1)], label: 1 },
 * ]
 * const result = c45Rules(rows, y, paths)
 * for (const r of result.rules) print(ruleText(r.conditions), '->', r.label)
 * print('else', result.defaultLabel)
 */
export function c45Rules(
  rows: RuleRows,
  y: ArrayLike<number>,
  paths: readonly ClassRule[],
  options: { cf?: number; candidates?: number } = {},
): TreeRulesResult {
  const cf = options.cf ?? 0.25
  const trace: TreeRulesStep[] = []
  let rules = paths.map((r) => ({ conditions: r.conditions.slice(), label: r.label }))
  trace.push({ kind: 'paths', rules: rules.map(copy) })
  rules.forEach((r, index) => {
    const g = generaliseRule(rows, y, r, cf)
    for (const deletion of g.deletions) {
      rules[index] = { conditions: deletion.conditions, label: r.label }
      trace.push({ kind: 'generalise', index, deletion, rules: rules.map(copy) })
    }
  })
  const seen = new Set<string>()
  const merged: ClassRule[] = []
  for (const r of rules) {
    const k = `${r.label}#${conditionsKey(r.conditions)}`
    if (seen.has(k)) continue
    seen.add(k)
    merged.push(r)
  }
  if (merged.length < rules.length)
    trace.push({ kind: 'merge', removed: rules.length - merged.length, rules: merged.map(copy) })
  rules = merged
  const distinct = new Set(paths.flatMap((r) => r.conditions.map((c) => conditionsKey([c])))).size
  const candidates = Math.max(2, options.candidates ?? distinct)
  let K = 2
  for (let i = 0; i < y.length; i++) K = Math.max(K, y[i] + 1)
  const masks = rules.map((r) => coverMask(rows, r.conditions))
  const chosen: number[] = []
  const falsePositives: Record<number, number> = {}
  for (let label = 0; label < K; label++) {
    const own = rules.map((r, i) => (r.label === label ? i : -1)).filter((i) => i >= 0)
    if (own.length === 0) continue
    const bits = (subset: number[]) =>
      classSubsetBits(
        rows.length,
        y,
        label,
        subset.map((i) => masks[i]),
        subset.map((i) => rules[i].conditions.length),
        candidates,
      )
    let best = own
    let bestBits = bits(own)
    if (own.length <= 12) {
      for (let mask = 0; mask < 1 << own.length; mask++) {
        const subset = own.filter((_, k) => mask & (1 << k))
        const b = bits(subset)
        if (b < bestBits - 1e-12) {
          best = subset
          bestBits = b
        }
      }
    } else {
      let improved = true
      while (improved) {
        improved = false
        for (const i of best) {
          const subset = best.filter((j) => j !== i)
          const b = bits(subset)
          if (b < bestBits) {
            best = subset
            bestBits = b
            improved = true
            break
          }
        }
      }
    }
    const kept = best
    trace.push({
      kind: 'select',
      label,
      kept,
      dropped: own.filter((i) => !kept.includes(i)),
      bits: bestBits,
      rules: rules.map(copy),
    })
    chosen.push(...kept)
    let fp = 0
    for (let i = 0; i < rows.length; i++) if (y[i] !== label && kept.some((k) => masks[k][i])) fp++
    falsePositives[label] = fp
  }
  const classOrder = Object.keys(falsePositives)
    .map(Number)
    .sort((a, b) => falsePositives[a] - falsePositives[b] || a - b)
  const ordered = classOrder.flatMap((label) => chosen.filter((i) => rules[i].label === label).map((i) => rules[i]))
  const orderedMasks = ordered.map((r) => coverMask(rows, r.conditions))
  const left = new Array<number>(K).fill(0)
  const all = new Array<number>(K).fill(0)
  for (let i = 0; i < rows.length; i++) {
    all[y[i]]++
    if (!orderedMasks.some((m) => m[i])) left[y[i]]++
  }
  const defaultLabel = argmaxTie(left, all)
  trace.push({ kind: 'order', rules: ordered.map(copy), defaultLabel, falsePositives })
  return { rules: ordered, defaultLabel, trace }
}

/**
 * The bits of one class's rule subset: its rules plus its exceptions on all rows (see the file comment).
 *
 * @param n The number of rows.
 * @param y The class of each row.
 * @param label The class of the subset.
 * @param masks The cover of each rule of the subset.
 * @param lengths The number of conditions of each rule of the subset.
 * @param candidates The number of candidate conditions per condition.
 * @returns The description length in bits.
 */
function classSubsetBits(
  n: number,
  y: ArrayLike<number>,
  label: number,
  masks: Uint8Array[],
  lengths: number[],
  candidates: number,
): number {
  let C = 0
  let fp = 0
  let fn = 0
  for (let i = 0; i < n; i++) {
    const covered = masks.some((m) => m[i])
    if (covered) {
      C++
      if (y[i] !== label) fp++
    } else if (y[i] === label) fn++
  }
  const U = n - C
  const theory =
    lengths.reduce((a, k) => a + k * Math.log2(candidates) - log2Factorial(k), 0) - log2Factorial(lengths.length)
  return theory + log2Choose(C, fp) + log2Choose(U, fn)
}

/**
 * $\log_2 k!$.
 *
 * @param k A count.
 * @returns The bits.
 */
function log2Factorial(k: number): number {
  return logFactorial(k) / Math.LN2
}

/**
 * $\log_2 \binom{n}{k}$.
 *
 * @param n The number of items.
 * @param k The number chosen.
 * @returns The bits.
 */
function log2Choose(n: number, k: number): number {
  return logChoose(n, k) / Math.LN2
}

/**
 * The index of the largest entry of `primary`, ties broken by the larger `secondary`, then by the smaller index.
 *
 * @param primary The values compared first.
 * @param secondary The values that break ties.
 * @returns The index.
 */
function argmaxTie(primary: number[], secondary: number[]): number {
  let best = 0
  for (let k = 1; k < primary.length; k++)
    if (primary[k] > primary[best] || (primary[k] === primary[best] && secondary[k] > secondary[best])) best = k
  return best
}

/**
 * A copy of a rule with its own array of conditions.
 *
 * @param r The rule.
 * @returns The copy.
 */
function copy(r: ClassRule): ClassRule {
  return { conditions: r.conditions.slice(), label: r.label }
}
