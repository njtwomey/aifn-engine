/**
 * Associative classification: class association rules, CBA's classifier builder and CMAR's pruning and weighted
 * $\chi^2$ vote.
 *
 * A class association rule (CAR) $X \to c$ pairs a frequent condition set $X$ with a class $c$. CBA (Liu, Hsu and Ma,
 * 1998, "Integrating classification and association rule mining", KDD): the support of $X \to c$ is the fraction of
 * rows that meet $X$ and have class $c$, its confidence the fraction of rows meeting $X$ that have class $c$; for each
 * condition set only the class of highest confidence is kept (the "possible rule"), and it is a CAR when its support
 * is at least `minSupport` and its confidence at least `minConfidence`. Condition sets are pre-mined level-wise
 * (`mineAntecedents`) with the same support threshold, which loses no CAR, since a condition set is at least as
 * frequent as any rule built on it. Rules are ranked by confidence, then support, then fewer conditions, then mining
 * order. The classifier builder is CBA-CB's M1: walk the ranked rules; a rule that covers at least one remaining row
 * correctly is appended and every remaining row it covers is removed; after each appended rule the default class is
 * the majority of the rows left and the total errors are the rules' errors plus the default's; the list is cut after
 * the first rule with the fewest total errors.
 *
 * CMAR (Li, Han and Pei, 2001, "CMAR: accurate and efficient classification based on multiple class-association
 * rules", ICDM, §3.3 and §4): rules are pruned when a more general rule (a subset of its conditions) ranks higher,
 * when the rule's $\chi^2$ against its class is below a threshold or its confidence is not above the class's share,
 * and by database coverage, where a row leaves only after $\delta$ selected rules have covered it; a row is classified
 * by grouping the matching rules by class and choosing the group with the largest weighted $\chi^2$,
 * $\sum \chi^2 \cdot \chi^2 / \max\chi^2$, where $\max\chi^2$ is the largest $\chi^2$ the rule's margins allow.
 * General-rule pruning compares a rule only with the rules already accepted, so a rule pruned by the $\chi^2$ test
 * cannot prune its specialisations (one reading of the paper); it ignores class, as in the paper.
 *
 * Differences: CBA's optional pessimistic-error pruning of CARs and CMAR's FP-growth miner and CR-tree are left out
 * (they change speed, not the rules). The predictors take a `covers(rule, i)` test, so the same classifier serves the
 * training rows (through the rules' masks) and new rows (through `satisfies` on their conditions).
 */

import type { Predicate } from 'aifn-compute/learning/explain'
import { conditionsKey, type Antecedent } from './conditions'

/** A class association rule. */
export type ClassAssociationRule = {
  /** The index of the condition set among the mined antecedents. */
  antecedent: number
  /** The conditions of the condition set. */
  conditions: Predicate[]
  /** The training rows the condition set covers. */
  mask: Uint8Array
  /** The class predicted. */
  label: number
  /** The rows meeting the conditions. */
  condCount: number
  /** The rows meeting the conditions with class `label`. */
  ruleCount: number
  /** `ruleCount` over the number of rows. */
  support: number
  /** `ruleCount` over `condCount`. */
  confidence: number
}

/**
 * The class association rules of pre-mined condition sets, ranked by CBA's precedence (confidence, then support,
 * then fewer conditions, then mining order).
 *
 * @param antecedents The condition sets, from `mineAntecedents` with a support threshold at most `minSupport`.
 * @param y The class of each row, $0, \dots, K - 1$.
 * @param classes The number of classes $K$.
 * @param options `minSupport` (default 0.01) and `minConfidence` (default 0.5), as fractions.
 * @returns The CARs, highest precedence first.
 *
 * @example The rules of a toy table
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0]]
 * const y = [0, 0, 1, 1, 1, 0]
 * const cars = classAssociationRules(mineAntecedents(rows, nominalConditions([2, 2])), y, 2, { minSupport: 0.2 })
 * for (const r of cars) print(ruleText(r.conditions), '->', r.label, ' confidence', r.confidence)
 */
export function classAssociationRules(
  antecedents: readonly Antecedent[],
  y: ArrayLike<number>,
  classes: number,
  options: { minSupport?: number; minConfidence?: number } = {},
): ClassAssociationRule[] {
  const minSupport = options.minSupport ?? 0.01
  const minConfidence = options.minConfidence ?? 0.5
  const n = y.length
  const out: (ClassAssociationRule & { order: number })[] = []
  antecedents.forEach((a, j) => {
    const counts = new Array<number>(classes).fill(0)
    let cond = 0
    for (let i = 0; i < n; i++) {
      if (a.mask[i]) {
        cond++
        counts[y[i]]++
      }
    }
    if (cond === 0) return
    let label = 0
    for (let c = 1; c < classes; c++) if (counts[c] > counts[label]) label = c
    const support = counts[label] / n
    const confidence = counts[label] / cond
    if (support + 1e-12 < minSupport || confidence + 1e-12 < minConfidence) return
    out.push({
      antecedent: j,
      conditions: a.conditions,
      mask: a.mask,
      label,
      condCount: cond,
      ruleCount: counts[label],
      support,
      confidence,
      order: j,
    })
  })
  out.sort(
    (r, s) =>
      s.confidence - r.confidence ||
      s.support - r.support ||
      r.conditions.length - s.conditions.length ||
      r.order - s.order,
  )
  return out.map(({ order: _order, ...rule }) => rule)
}

/** What happened to one ranked rule in CBA-CB's M1 pass. */
export type CbaStep = {
  /** The index of the rule in the ranked list. */
  rule: number
  /** The remaining rows the rule covers. */
  covered: number
  /** How many of them it classifies correctly. */
  correct: number
  /** Whether the rule was appended (it classified at least one remaining row correctly). */
  kept: boolean
  /** The rows still unclassified after this rule. */
  remaining: number
  /** For a kept rule, the default class of the list cut after it ($-1$ otherwise). */
  defaultLabel: number
  /** For a kept rule, the training errors of the list cut after it (NaN otherwise). */
  totalErrors: number
}

/** A CBA classifier and the trace of its construction. */
export type CbaClassifier = {
  /** Indices into the ranked rules of the rules of the final list, in order. */
  list: number[]
  /** The class of a row no rule of the list covers. */
  defaultLabel: number
  /** The training errors of the final classifier. */
  errors: number
  /** One entry per ranked rule examined, in order (the pass stops when no row is left). */
  trace: CbaStep[]
  /** The index into `trace` of the cut-off rule ($-1$ when no rule is kept). */
  cut: number
}

/**
 * The majority class of the active rows and the number of active rows outside it.
 *
 * @param y The class of each row.
 * @param active Which rows count.
 * @param classes The number of classes.
 * @returns The majority `label` (the first on a tie) and its `errors`.
 */
function majority(y: ArrayLike<number>, active: Uint8Array, classes: number): { label: number; errors: number } {
  const counts = new Array<number>(classes).fill(0)
  let total = 0
  for (let i = 0; i < y.length; i++) {
    if (active[i]) {
      counts[y[i]]++
      total++
    }
  }
  let label = 0
  for (let c = 1; c < classes; c++) if (counts[c] > counts[label]) label = c
  return { label, errors: total - counts[label] }
}

/**
 * CBA-CB's M1: select rules by database coverage and cut the list where the training errors are fewest.
 *
 * @param rules The ranked CARs.
 * @param y The class of each row.
 * @param classes The number of classes $K$.
 * @returns The classifier and the trace of every rule examined.
 *
 * @example A two-rule classifier
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0]]
 * const y = [0, 0, 1, 1, 1, 0]
 * const cars = classAssociationRules(mineAntecedents(rows, nominalConditions([2, 2])), y, 2, { minSupport: 0.2 })
 * const cba = cbaBuild(cars, y, 2)
 * print('list:', cba.list.map((r) => ruleText(cars[r].conditions)))
 * print('default:', cba.defaultLabel, ' errors:', cba.errors)
 */
export function cbaBuild(rules: readonly ClassAssociationRule[], y: ArrayLike<number>, classes: number): CbaClassifier {
  const n = y.length
  const active = new Uint8Array(n).fill(1)
  let remaining = n
  let ruleErrors = 0
  const trace: CbaStep[] = []
  const kept: number[] = []
  let best = { at: -1, errors: Infinity, defaultLabel: majority(y, active, classes).label, length: 0 }
  for (let r = 0; r < rules.length && remaining > 0; r++) {
    const rule = rules[r]
    let covered = 0
    let correct = 0
    for (let i = 0; i < n; i++) {
      if (active[i] && rule.mask[i]) {
        covered++
        if (y[i] === rule.label) correct++
      }
    }
    if (correct === 0) {
      trace.push({ rule: r, covered, correct, kept: false, remaining, defaultLabel: -1, totalErrors: NaN })
      continue
    }
    for (let i = 0; i < n; i++) {
      if (active[i] && rule.mask[i]) {
        active[i] = 0
        remaining--
      }
    }
    ruleErrors += covered - correct
    kept.push(r)
    const d = majority(y, active, classes)
    const total = ruleErrors + d.errors
    trace.push({ rule: r, covered, correct, kept: true, remaining, defaultLabel: d.label, totalErrors: total })
    if (total < best.errors) best = { at: trace.length - 1, errors: total, defaultLabel: d.label, length: kept.length }
  }
  if (best.at < 0) {
    const d = majority(y, new Uint8Array(n).fill(1), classes)
    return { list: [], defaultLabel: d.label, errors: d.errors, trace, cut: -1 }
  }
  return { list: kept.slice(0, best.length), defaultLabel: best.defaultLabel, errors: best.errors, trace, cut: best.at }
}

/**
 * Predictions of a CBA classifier: the class of the first rule of its list that covers the row, else the default.
 *
 * @param classifier The classifier.
 * @param rules The ranked CARs it indexes.
 * @param covers Whether rule `rule` (an index into `rules`) covers row `i`: `rules[rule].mask[i] === 1` for the
 *   training rows, `satisfies(rows[i], rules[rule].conditions)` for new ones.
 * @param n The number of rows.
 * @returns One class per row.
 *
 * @example Predict the training rows through the rules' masks
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0]]
 * const y = [0, 0, 1, 1, 1, 0]
 * const cars = classAssociationRules(mineAntecedents(rows, nominalConditions([2, 2])), y, 2, { minSupport: 0.2 })
 * const cba = cbaBuild(cars, y, 2)
 * print(cbaPredict(cba, cars, (r, i) => cars[r].mask[i] === 1, rows.length))
 */
export function cbaPredict(
  classifier: CbaClassifier,
  rules: readonly ClassAssociationRule[],
  covers: (rule: number, i: number) => boolean,
  n: number,
): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const hit = classifier.list.find((r) => covers(r, i))
    out.push(hit === undefined ? classifier.defaultLabel : rules[hit].label)
  }
  return out
}

/**
 * The $\chi^2$ statistic of the $2 \times 2$ table (meets $X$ or not) $\times$ (class $c$ or not), and the largest
 * value its margins allow (Li, Han and Pei, 2001, §4).
 *
 * @param condCount The rows meeting $X$.
 * @param ruleCount The rows meeting $X$ with class $c$.
 * @param classCount The rows of class $c$.
 * @param n The number of rows.
 * @returns `chi2` and `maxChi2`.
 *
 * @example A rule that matches its class exactly reaches the maximum
 * print(ruleChiSquare(500, 500, 500, 2000))
 * print(ruleChiSquare(1731, 1364, 1490, 2201))
 */
export function ruleChiSquare(
  condCount: number,
  ruleCount: number,
  classCount: number,
  n: number,
): { chi2: number; maxChi2: number } {
  const a = ruleCount
  const b = condCount - ruleCount
  const c = classCount - ruleCount
  const d = n - condCount - classCount + ruleCount
  const cells = [
    [a, condCount, classCount],
    [b, condCount, n - classCount],
    [c, n - condCount, classCount],
    [d, n - condCount, n - classCount],
  ]
  let chi2 = 0
  for (const [o, r, k] of cells) {
    const e = (r * k) / n
    if (e > 0) chi2 += ((o - e) * (o - e)) / e
  }
  const sp = condCount
  const sc = classCount
  const inv = (v: number) => (v > 0 ? 1 / v : 0)
  const e = inv(sp * sc) + inv(sp * (n - sc)) + inv((n - sp) * sc) + inv((n - sp) * (n - sc))
  const maxChi2 = (Math.min(sp, sc) - (sp * sc) / n) ** 2 * n * e
  return { chi2, maxChi2 }
}

/** A CMAR classifier: the selected rules with their $\chi^2$ statistics, and the default class. */
export type CmarClassifier = {
  /** Indices into the ranked rules, in rank order. */
  rules: number[]
  /** The $\chi^2$ of each selected rule. */
  chi2: number[]
  /** The largest $\chi^2$ each selected rule's margins allow. */
  maxChi2: number[]
  /** The most frequent class, for a row no rule covers. */
  defaultLabel: number
  /** Rules removed because a more general rule ranks higher. */
  prunedGeneral: number
  /** Rules removed by the $\chi^2$ test or for a confidence not above the class share. */
  prunedCorrelation: number
  /** Rules removed by database coverage. */
  prunedCoverage: number
}

/**
 * CMAR's three pruning steps on ranked CARs (see the file comment). A row's coverage count is an `Int32Array`, so any
 * $\delta$ works.
 *
 * @param rules The ranked CARs (CMAR's rank is CBA's precedence).
 * @param y The class of each row.
 * @param classes The number of classes $K$.
 * @param options `coverage`, the $\delta$ of database coverage (default 4), and `chiSquare`, the threshold of the
 *   correlation test (default 3.841, the 95% point of $\chi^2_1$).
 * @returns The selected rules and the counts pruned by each step.
 *
 * @example Prune the rules of a toy table
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0], [1, 1], [0, 1]]
 * const y = [0, 0, 1, 1, 1, 0, 1, 0]
 * const cars = classAssociationRules(mineAntecedents(rows, nominalConditions([2, 2])), y, 2, { minSupport: 0.2 })
 * const cmar = cmarBuild(cars, y, 2, { chiSquare: 1 })
 * print('kept:', cmar.rules.map((r) => ruleText(cars[r].conditions)), ' chi2:', cmar.chi2)
 */
export function cmarBuild(
  rules: readonly ClassAssociationRule[],
  y: ArrayLike<number>,
  classes: number,
  options: { coverage?: number; chiSquare?: number } = {},
): CmarClassifier {
  const delta = options.coverage ?? 4
  const threshold = options.chiSquare ?? 3.841
  const n = y.length
  const classCount = new Array<number>(classes).fill(0)
  for (let i = 0; i < n; i++) classCount[y[i]]++
  const sets = rules.map((r) => new Set(r.conditions.map((c) => conditionsKey([c]))))
  let prunedGeneral = 0
  let prunedCorrelation = 0
  const stage: number[] = []
  const stats = new Map<number, { chi2: number; maxChi2: number }>()
  for (let r = 0; r < rules.length; r++) {
    const general = stage.some((g) => [...sets[g]].every((k) => sets[r].has(k)))
    if (general) {
      prunedGeneral++
      continue
    }
    const rule = rules[r]
    const s = ruleChiSquare(rule.condCount, rule.ruleCount, classCount[rule.label], n)
    if (s.chi2 < threshold || rule.confidence <= classCount[rule.label] / n) {
      prunedCorrelation++
      continue
    }
    stats.set(r, s)
    stage.push(r)
  }
  const count = new Int32Array(n)
  const active = new Uint8Array(n).fill(1)
  let left = n
  const selected: number[] = []
  for (const r of stage) {
    if (left === 0) break
    const rule = rules[r]
    let correct = false
    for (let i = 0; i < n; i++) if (active[i] && rule.mask[i] && y[i] === rule.label) correct = true
    if (!correct) continue
    selected.push(r)
    for (let i = 0; i < n; i++) {
      if (active[i] && rule.mask[i]) {
        count[i]++
        if (count[i] >= delta) {
          active[i] = 0
          left--
        }
      }
    }
  }
  let defaultLabel = 0
  for (let c = 1; c < classes; c++) if (classCount[c] > classCount[defaultLabel]) defaultLabel = c
  return {
    rules: selected,
    chi2: selected.map((r) => (stats.get(r) as { chi2: number }).chi2),
    maxChi2: selected.map((r) => (stats.get(r) as { maxChi2: number }).maxChi2),
    defaultLabel,
    prunedGeneral,
    prunedCorrelation,
    prunedCoverage: stage.length - selected.length,
  }
}

/**
 * CMAR's prediction: the class whose matching rules have the largest weighted $\chi^2$,
 * $\sum \chi^2 \cdot \chi^2 / \max\chi^2$; the default class when no rule matches.
 *
 * @param classifier The CMAR rules.
 * @param rules The ranked CARs it indexes.
 * @param covers Whether rule `rule` (an index into `rules`) covers row `i`.
 * @param n The number of rows.
 * @param classes The number of classes $K$.
 * @returns One class per row.
 *
 * @example Vote on the training rows
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0], [1, 1], [0, 1]]
 * const y = [0, 0, 1, 1, 1, 0, 1, 0]
 * const cars = classAssociationRules(mineAntecedents(rows, nominalConditions([2, 2])), y, 2, { minSupport: 0.2 })
 * const cmar = cmarBuild(cars, y, 2, { chiSquare: 1 })
 * print(cmarPredict(cmar, cars, (r, i) => cars[r].mask[i] === 1, rows.length, 2))
 */
export function cmarPredict(
  classifier: CmarClassifier,
  rules: readonly ClassAssociationRule[],
  covers: (rule: number, i: number) => boolean,
  n: number,
  classes: number,
): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const score = new Array<number>(classes).fill(0)
    let any = false
    classifier.rules.forEach((r, k) => {
      if (!covers(r, i)) return
      any = true
      const m = classifier.maxChi2[k]
      score[rules[r].label] += m > 0 ? (classifier.chi2[k] * classifier.chi2[k]) / m : 0
    })
    if (!any) {
      out.push(classifier.defaultLabel)
      continue
    }
    let best = 0
    for (let c = 1; c < classes; c++) if (score[c] > score[best]) best = c
    out.push(best)
  }
  return out
}
