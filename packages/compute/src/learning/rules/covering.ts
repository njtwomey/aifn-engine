/**
 * Sequential covering (separate and conquer) with a beam search for each rule, as in CN2 (Clark and Niblett, 1989,
 * "The CN2 induction algorithm", Machine Learning 3, §3; Clark and Boswell, 1991, for the Laplace estimate and the
 * unordered mode), with a full trace of every step for a figure to step through.
 *
 * Each rule starts empty and is specialised one condition at a time: every member of the beam is extended by every
 * candidate condition, each extension is scored on the rows still in play by a rule quality measure (`ruleQuality`),
 * and the best `beam` extensions are kept. The best rule seen at any depth is the one learned, provided it beats the
 * empty rule and, when `significance` is set, its likelihood-ratio statistic exceeds that threshold (CN2's
 * significance test). In `ordered` mode (a decision list) a rule predicts the majority class of the rows it covers and
 * every covered row is removed; in `target` mode (a rule set for one class, CN2's unordered mode run once per class)
 * rules predict the target class and only the covered positives are removed, so later rules are still judged against
 * every negative. The loop stops when no rule beats the empty rule, when fewer than `minSupport` rows (ordered) or
 * positives (target) remain, or after `maxRules` rules; the default class is the majority of the rows left.
 *
 * Differences from CN2: the beam search stops at the first depth that does not improve on the best rule so far (CN2
 * runs until the beam is empty), and the conditions come from a given list (thresholds at quantiles, or nominal
 * equalities). `ruleList` of `aifn-compute/learning/explain` is the special case with the Laplace estimate, without a
 * trace and with a search that runs to `maxConditions`.
 */

import { simplifyRule, type Predicate } from 'aifn-compute/learning/explain'
import { andMask, applyClassRules, conditionsKey, coverMask, type RuleRows } from './conditions'
import { likelihoodRatio, ruleQuality, type RuleMeasure, type RuleMeasureOptions } from './quality'

/** Options of `sequentialCovering`. */
export type CoveringOptions = {
  /** `ordered`: a decision list over all classes (default). `target`: rules for the class `target` only. */
  mode?: 'ordered' | 'target'
  /** The class the rules of `target` mode predict (default 1). */
  target?: number
  /** The quality measure that ranks candidate rules (default `laplace`). */
  measure?: RuleMeasure
  /**
   * Parameters of the measure ($m$ of the m-estimate); the Laplace class count defaults to the number of classes.
   * FOIL gain's parent is always the rule being extended.
   */
  measureOptions?: RuleMeasureOptions
  /** Rules kept per depth of the beam search (default 3; 1 is hill climbing). */
  beam?: number
  /** At most this many conditions per rule (default 3). */
  maxConditions?: number
  /** A rule must cover at least this many rows in play (default 2). */
  minSupport?: number
  /** At most this many rules (default 10). */
  maxRules?: number
  /**
   * CN2's significance threshold on the likelihood-ratio statistic (default 0, off; 3.84 is the 5% level for two
   * classes).
   */
  significance?: number
}

/** A candidate rule during the search, scored on the rows in play. */
export type CandidateRule = {
  /** The rule's conditions, one per feature after `simplifyRule`. */
  conditions: Predicate[]
  /** The class it predicts: the majority of its covered rows (ordered mode) or the target. */
  label: number
  /** Covered rows in play of the rule's class. */
  p: number
  /** Covered rows in play of the other classes. */
  n: number
  /** Covered rows in play of every class, indexed by class. */
  counts: number[]
  /** The rule's quality under the measure. */
  score: number
}

/** A learned rule: a candidate with the rows it covered. */
export type CoveredRule = CandidateRule & {
  /** The rows (of all) the rule covered among those in play when it was learned. */
  covered: Uint8Array
}

/** One step of the trace. `active` marks the rows in play before the step's removal. */
export type CoveringStep =
  | {
      /** One depth of the beam search for a rule. */
      kind: 'grow'
      /** The index of the rule being learned. */
      rule: number
      /** Depth of the search: the number of conditions of the rules in this beam. */
      depth: number
      /** The best rule of this depth. */
      best: CandidateRule
      /** The rows in play the best rule of this depth covers. */
      covered: Uint8Array
      /** The whole beam at this depth, best first. */
      beam: CandidateRule[]
      /** The rows in play. */
      active: Uint8Array
    }
  | {
      /** A rule is learned. */
      kind: 'accept'
      /** The index of the rule. */
      rule: number
      /** The rule. */
      best: CandidateRule
      /** The rows in play it covers. */
      covered: Uint8Array
      /** The rows in play. */
      active: Uint8Array
    }
  | {
      /** The covered rows leave play. */
      kind: 'remove'
      /** The index of the rule. */
      rule: number
      /** How many rows left play. */
      removed: number
      /** The rows in play after the removal. */
      active: Uint8Array
    }
  | {
      /** The loop ends. */
      kind: 'stop'
      /** Why it stopped. */
      reason: string
      /** The default class. */
      defaultLabel: number
      /** The rows still in play. */
      active: Uint8Array
    }

/** What `sequentialCovering` returns. */
export type CoveringResult = {
  /** The rules, in the order learned. */
  rules: CoveredRule[]
  /** The class of a row no rule covers: the majority of the rows left (of all rows when none is left). */
  defaultLabel: number
  /** The number of classes $K$ (at least 2). */
  classes: number
  /** Every step, for a figure to step through. */
  trace: CoveringStep[]
}

/**
 * Learn rules by sequential covering with a beam search per rule (see the file comment). A plain function of plain
 * data, so it can run in a worker.
 *
 * @param rows The training rows.
 * @param y The class of each row, integers $0, \dots, K - 1$.
 * @param conditions The candidate conditions (`thresholdConditions`, `nominalConditions`).
 * @param options The mode, the measure, the beam, the limits and the significance threshold.
 * @returns The rules in order, the default class, the number of classes and the trace.
 *
 * @example A decision list for an AND of two features
 * const rows = []
 * for (const a of [0, 1]) for (const b of [0, 1]) for (let k = 0; k < 3; k++) rows.push([a, b])
 * const y = rows.map(([a, b]) => a & b)
 * const result = sequentialCovering(rows, y, nominalConditions([2, 2]))
 * for (const r of result.rules) print(ruleText(r.conditions), '->', r.label, ` (p = ${r.p}, n = ${r.n})`)
 * print('else', result.defaultLabel)
 *
 * @example Target mode learns rules for one class only
 * const rows = [[0], [1], [2], [3], [4], [5]]
 * const y = [0, 1, 0, 1, 1, 1]
 * const result = sequentialCovering(rows, y, nominalConditions([6]), { mode: 'target', target: 1, minSupport: 1 })
 * print(result.rules.map((r) => ruleText(r.conditions)))
 */
export function sequentialCovering(
  rows: RuleRows,
  y: ArrayLike<number>,
  conditions: readonly Predicate[],
  options: CoveringOptions = {},
): CoveringResult {
  const nRows = rows.length
  let K = 2
  for (let i = 0; i < nRows; i++) K = Math.max(K, y[i] + 1)
  const mode = options.mode ?? 'ordered'
  const target = options.target ?? 1
  const measure = options.measure ?? 'laplace'
  const beamWidth = Math.max(1, options.beam ?? 3)
  const maxConditions = options.maxConditions ?? 3
  const minSupport = Math.max(1, options.minSupport ?? 2)
  const maxRules = options.maxRules ?? 10
  const significance = options.significance ?? 0
  const masks = conditions.map((c) => coverMask(rows, [c]))
  const active = new Uint8Array(nRows).fill(1)
  const rules: CoveredRule[] = []
  const trace: CoveringStep[] = []

  // Class counts of the covered rows in play.
  const classCounts = (mask: Uint8Array) => {
    const c = new Array<number>(K).fill(0)
    for (let i = 0; i < nRows; i++) if (mask[i] && active[i]) c[y[i]]++
    return c
  }
  const totals = () => classCounts(active)

  // Score a cover: its label, counts and quality against the rows in play.
  const score = (
    conds: Predicate[],
    mask: Uint8Array,
    tot: number[],
    parent: { p: number; n: number } | undefined,
  ): CandidateRule => {
    const c = classCounts(mask)
    const covered = sum(c)
    const label = mode === 'target' ? target : c.indexOf(Math.max(...c))
    const p = c[label]
    const n = covered - p
    const P = tot[label]
    const N = sum(tot) - P
    const opts: RuleMeasureOptions = { classes: K, ...options.measureOptions }
    if (measure === 'foil-gain' && parent) opts.parent = parent
    return { conditions: conds, label, p, n, counts: c, score: ruleQuality(measure, { p, n, P, N }, opts) }
  }

  const remaining = () => {
    let a = 0
    let pos = 0
    for (let i = 0; i < nRows; i++)
      if (active[i]) {
        a++
        if (y[i] === target) pos++
      }
    return mode === 'target' ? pos : a
  }

  let reason = 'rule limit reached'
  while (rules.length < maxRules) {
    if (remaining() < minSupport) {
      reason = mode === 'target' ? 'too few positives left' : 'too few rows left'
      break
    }
    const tot = totals()
    const all = new Uint8Array(nRows).fill(1)
    const empty = score([], all, tot, undefined)
    let best: { cand: CandidateRule; mask: Uint8Array } | null = null
    let beam: { cand: CandidateRule; mask: Uint8Array }[] = [{ cand: empty, mask: all }]
    const ruleIndex = rules.length
    for (let depth = 1; depth <= maxConditions; depth++) {
      const seen = new Set<string>()
      const next: { cand: CandidateRule; mask: Uint8Array }[] = []
      for (const b of beam) {
        conditions.forEach((cond, j) => {
          const conds = simplifyRule([...b.cand.conditions, cond])
          const k = conditionsKey(conds)
          if (conds.length === b.cand.conditions.length && k === conditionsKey(b.cand.conditions)) return
          if (seen.has(k)) return
          seen.add(k)
          // An empty interval covers no row: skip it before scoring.
          if (conds.some((c) => !(c.lower < c.upper))) return
          const mask = andMask(b.mask, masks[j])
          const cand = score(conds, mask, tot, { p: b.cand.p, n: b.cand.n })
          if (cand.p + cand.n < minSupport) return
          if (mode === 'target' && cand.p === 0) return
          next.push({ cand, mask })
        })
      }
      if (next.length === 0) break
      next.sort(
        (a, b) =>
          b.cand.score - a.cand.score ||
          b.cand.p + b.cand.n - (a.cand.p + a.cand.n) ||
          a.cand.conditions.length - b.cand.conditions.length,
      )
      beam = next.slice(0, beamWidth)
      let improved = false
      for (const b of beam) {
        const P = tot[b.cand.label]
        const passes =
          significance <= 0 || likelihoodRatio({ p: b.cand.p, n: b.cand.n, P, N: sum(tot) - P }) > significance
        if (passes && (!best || b.cand.score > best.cand.score)) {
          best = b
          improved = true
        }
      }
      trace.push({
        kind: 'grow',
        rule: ruleIndex,
        depth,
        best: beam[0].cand,
        covered: andMask(beam[0].mask, active),
        beam: beam.map((b) => b.cand),
        active: active.slice(),
      })
      if (!improved) break
    }
    const baseline = measure === 'foil-gain' || measure === 'wracc' ? 0 : empty.score
    if (!best || !(best.cand.score > baseline)) {
      reason = 'no rule beats the empty rule'
      break
    }
    const covered = andMask(best.mask, active)
    rules.push({ ...best.cand, covered })
    trace.push({ kind: 'accept', rule: ruleIndex, best: best.cand, covered, active: active.slice() })
    let removed = 0
    for (let i = 0; i < nRows; i++) {
      if (!covered[i]) continue
      if (mode === 'target' && y[i] !== target) continue
      active[i] = 0
      removed++
    }
    trace.push({ kind: 'remove', rule: ruleIndex, removed, active: active.slice() })
  }
  const left = totals()
  const defaultLabel = sum(left) > 0 ? left.indexOf(Math.max(...left)) : majority(y, K)
  trace.push({ kind: 'stop', reason, defaultLabel, active: active.slice() })
  return { rules, defaultLabel, classes: K, trace }
}

/**
 * Apply rules learned by `sequentialCovering`: as a decision list (the first covering rule gives the class), or as an
 * unordered set (each covering rule adds its covered class counts, and the largest total wins: CN2's unordered vote).
 * A row no rule covers gets the default class.
 *
 * @param result The rules (with their class counts and labels) and the default class.
 * @param rows The rows to classify.
 * @param how `list` (default) or `vote`.
 * @returns One class per row.
 *
 * @example The list on new rows
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]]
 * const result = sequentialCovering(rows, [0, 0, 0, 1, 1, 1], nominalConditions([2, 2]))
 * print(applyCoveringRules(result, [[1, 1], [0, 1]]))
 */
export function applyCoveringRules(result: CoveringResult, rows: RuleRows, how: 'list' | 'vote' = 'list'): Int32Array {
  if (how === 'list') return applyClassRules(result.rules, result.defaultLabel, rows)
  const masks = result.rules.map((r) => coverMask(rows, r.conditions))
  const out = new Int32Array(rows.length)
  for (let i = 0; i < rows.length; i++) {
    out[i] = result.defaultLabel
    const votes = new Array<number>(result.classes).fill(0)
    let any = false
    result.rules.forEach((r, k) => {
      if (!masks[k][i]) return
      any = true
      for (let c = 0; c < result.classes; c++) votes[c] += r.counts[c]
    })
    if (any) out[i] = votes.indexOf(Math.max(...votes))
  }
  return out
}

/**
 * The sum of an array.
 *
 * @param v The values.
 * @returns Their sum (0 for none).
 */
function sum(v: readonly number[]): number {
  return v.reduce((a, b) => a + b, 0)
}

/**
 * The most frequent class among all rows (the first on a tie).
 *
 * @param y The class of each row.
 * @param K The number of classes.
 * @returns The majority class.
 */
function majority(y: ArrayLike<number>, K: number): number {
  const c = new Array<number>(K).fill(0)
  for (let i = 0; i < y.length; i++) c[y[i]]++
  return c.indexOf(Math.max(...c))
}
