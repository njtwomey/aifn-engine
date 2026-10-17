/**
 * RIPPER (Cohen, 1995, "Fast effective rule induction", ICML, §3 and §4), with its description length as Weka's JRip
 * encodes Cohen's program, and a trace of every phase.
 *
 * For a two-class problem with a positive class, IREP* learns a rule set (a disjunction of conjunctions that predicts
 * positive). The rows in play are split at random into a growing set (by default $2/3$) and a pruning set. A rule is
 * grown on the growing set from the empty rule by adding, at each step, the condition of largest FOIL gain against the
 * rule so far, until it covers no negative growing row or no condition gains; conditions keep the order they were
 * added in (two on one feature are not merged). It is then pruned on the pruning set: of the rule and every prefix of
 * it (deleting any final sequence of conditions), the one of largest $(p - n)/(p + n)$ is kept, the shorter on a tie.
 * The rule is accepted unless its error rate on the pruning set reaches 50%, and every row it covers leaves play.
 * After each rule the description length (DL) of the rule set and its exceptions is computed on the rows of the
 * problem: learning stops when no positive is left, or when the DL exceeds the smallest DL seen so far by `slack` bits
 * (64). Rules are then deleted, last first, while a deletion lowers the DL. The optimisation pass (RIPPER$k$ repeats it
 * $k$ times) revisits each rule $R_i$ on a fresh split: a replacement is grown from the empty rule, a revision from
 * $R_i$, and each is pruned to maximise the accuracy of the whole rule set (with it in place of $R_i$) on the pruning
 * set; the variant of smallest DL is kept. Positives still uncovered get new rules by IREP*, and the DL deletion runs
 * again. Multiclass problems are learned one class at a time, from the least to the most frequent; the rows a class's
 * rules cover leave play, and the most frequent class is the default.
 *
 * Description length, in bits (Quinlan, 1995, and Cohen, 1995, as in JRip): a rule of $k$ conditions out of $c$
 * candidates costs $\frac12\,(\lVert k \rVert + S(c, k, k/c))$, with $\lVert k \rVert = \log_2 k + 2\log_2\log_2 k$ for
 * $k > 1$ ($\log_2 k$ for $k = 1$) and $S(t, k, p) = -k\log_2 p - (t - k)\log_2(1 - p)$ the bits to name a subset of
 * $k$ of $t$ items at rate $p$. The exceptions of a rule set covering $C$ rows with $f_p$ false positives, and leaving
 * $U$ rows uncovered with $f_n$ false negatives, cost $\log_2(C + U + 1) + S(C, f_p, e/(2C)) + S(U, f_n, f_n/U)$ when
 * $C > U$ (else $S(C, f_p, f_p/C) + S(U, f_n, e/(2U))$), with $e = f_p + f_n$.
 *
 * Every split draws from its own child stream, so a trace is reproducible from the seed. The splits are not
 * stratified (JRip stratifies them).
 */

import { child, integers, type Stream } from 'aifn-compute/foundation/random'
import type { Predicate } from 'aifn-compute/learning/explain'
import { foilGain } from 'aifn-compute/logic/induction'
import { andMask, applyClassRules, coverMask, type ClassRule, type RuleRows } from './conditions'
import { ripperPruneValue } from './quality'

/** Options of `ripper`. */
export type RipperOptions = {
  /** Share of the rows in play used for growing (default $2/3$). */
  growFraction?: number
  /** At most this many conditions per rule (default 6). */
  maxConditions?: number
  /** Stop when the DL exceeds the smallest DL so far by this many bits (default 64). */
  slack?: number
  /** Optimisation passes $k$ of RIPPER$k$ (default 1; 0 is IREP* alone). */
  optimisations?: number
  /** The class to learn rules for in a two-class problem (default the less frequent). Multiclass ignores it. */
  positive?: number
}

/** A rule of the result: its conditions and the class it predicts. */
export type RipperRule = ClassRule

/** One step of the trace of one class's rule set. Masks run over all rows. */
export type RipperStep =
  | {
      /** A condition was added while growing a rule (or the growth started). */
      phase: 'grow'
      /** The class whose rules are being learned. */
      label: number
      /** The index of the rule in the class's set. */
      rule: number
      /** The rule so far. */
      conditions: Predicate[]
      /** The split: 1 for a growing row, 2 for a pruning row, 0 for a row out of play. */
      split: Uint8Array
      /** Positives and negatives covered on the growing set. */
      grow: { p: number; n: number }
      /** Positives and negatives covered on the pruning set. */
      prune: { p: number; n: number }
    }
  | {
      /** The grown rule was pruned. */
      phase: 'prune'
      /** The class whose rules are being learned. */
      label: number
      /** The index of the rule in the class's set. */
      rule: number
      /** The grown rule and each of its non-empty prefixes, longest first, with their pruning-set counts and value. */
      candidates: { conditions: Predicate[]; p: number; n: number; value: number }[]
      /** The index into `candidates` of the kept rule. */
      kept: number
      /** The split: 1 growing, 2 pruning, 0 out of play. */
      split: Uint8Array
    }
  | {
      /** The rule was accepted or rejected. */
      phase: 'accept' | 'reject'
      /** The class whose rules are being learned. */
      label: number
      /** The index of the rule in the class's set. */
      rule: number
      /** The rule. */
      conditions: Predicate[]
      /** Description length of the rule set after the decision. */
      dl: number
      /** The smallest description length so far. */
      minDl: number
      /** The rows still in play after the rule's covered rows leave. */
      active: Uint8Array
      /** Why learning stops here, when it does. */
      reason?: string
    }
  | {
      /** A rule was deleted because the set's DL is shorter without it. */
      phase: 'delete'
      /** The class whose rules are being learned. */
      label: number
      /** The deleted rule. */
      conditions: Predicate[]
      /** The DL without it. */
      dl: number
    }
  | {
      /** One rule of the optimisation pass. */
      phase: 'optimise'
      /** The class whose rules are being learned. */
      label: number
      /** The index of the rule revisited. */
      rule: number
      /** The rule as it was. */
      original: Predicate[]
      /** The rule grown afresh from the empty rule and pruned for the whole set. */
      replacement: Predicate[]
      /** The rule grown on from the original and pruned for the whole set. */
      revision: Predicate[]
      /** The DL of the rule set with each variant; the smallest is kept. */
      dls: { original: number; replacement: number; revision: number }
      /** The variant kept. */
      chosen: 'original' | 'replacement' | 'revision'
    }
  | {
      /** The class's rule set is final. */
      phase: 'done'
      /** The class. */
      label: number
      /** Its rules. */
      rules: Predicate[][]
      /** Their DL. */
      dl: number
    }

/** What `ripper` returns. */
export type RipperResult = {
  /** The rules in order: a row takes the label of the first rule it meets, else the default. */
  rules: RipperRule[]
  /** The most frequent class (or the other class of a two-class problem with `positive` set). */
  defaultLabel: number
  /** Every phase of every class, in order. */
  trace: RipperStep[]
}

/**
 * $S(t, k, p) = -k\log_2 p - (t - k)\log_2(1 - p)$, the bits to name a subset of $k$ of $t$ items at rate $p$, with
 * $0 \log 0 = 0$.
 *
 * @param t The number of items $t$.
 * @param k The size of the subset $k$.
 * @param p The rate $p$ at which items are taken.
 * @returns The bits.
 *
 * @example Two of four items at rate one half cost four bits
 * print(subsetBits(4, 2, 0.5))
 */
export function subsetBits(t: number, k: number, p: number): number {
  const a = k > 0 ? -k * Math.log2(p) : 0
  const b = t - k > 0 ? -(t - k) * Math.log2(1 - p) : 0
  return a + b
}

/**
 * The bits for one rule of $k$ conditions out of $c$ candidates, $\frac12\,(\lVert k \rVert + S(c, k, k/c))$, halved
 * for redundancy as in JRip.
 *
 * @param k The number of conditions $k$ (0 costs nothing).
 * @param c The number of candidate conditions $c$.
 * @returns The bits.
 *
 * @example Longer rules cost more
 * for (const k of [1, 2, 3]) print(k, 'conditions of 8:', ruleBits(k, 8))
 */
export function ruleBits(k: number, c: number): number {
  if (k === 0) return 0
  let bits = Math.log2(k)
  if (k > 1) bits += 2 * Math.log2(bits)
  return 0.5 * (bits + subsetBits(c, k, k / c))
}

/**
 * The bits for the exceptions of a rule set: the false positives among the $C$ rows it covers and the false negatives
 * among the $U$ rows it leaves (see the file comment).
 *
 * @param C The rows the set covers.
 * @param U The rows it leaves uncovered.
 * @param fp The false positives $f_p$ among the covered rows.
 * @param fn The false negatives $f_n$ among the uncovered rows.
 * @returns The bits.
 *
 * @example A perfect rule set costs only the count, log2(C + U + 1)
 * print(exceptionBits(10, 10, 0, 0), Math.log2(21))
 */
export function exceptionBits(C: number, U: number, fp: number, fn: number): number {
  const e = fp + fn
  const total = Math.log2(C + U + 1)
  if (C > U) return total + subsetBits(C, fp, e / (2 * C)) + (U > 0 ? subsetBits(U, fn, fn / U) : 0)
  return total + (C > 0 ? subsetBits(C, fp, fp / C) : 0) + subsetBits(U, fn, U > 0 ? e / (2 * U) : 0)
}

/**
 * The description length of a two-class rule set on the given rows: the rules' bits plus the exceptions' bits. A row
 * is predicted positive when any rule covers it.
 *
 * @param rules The rule set, each a conjunction.
 * @param rows The rows.
 * @param positive Whether each row is positive.
 * @param inPlay Which rows count; `undefined` for every row.
 * @param candidates The number of candidate conditions $c$.
 * @returns The DL in bits.
 *
 * @example The empty set against one rule on a toy table
 * const rows = [[0], [0], [1], [1], [1]]
 * const positive = [false, false, true, true, false]
 * const rule = [{ feature: 0, lower: 0.5, upper: 1.5 }]
 * print('empty:', ripperDescriptionLength([], rows, positive, undefined, 2))
 * print('x = 1:', ripperDescriptionLength([rule], rows, positive, undefined, 2))
 */
export function ripperDescriptionLength(
  rules: readonly (readonly Predicate[])[],
  rows: RuleRows,
  positive: ArrayLike<boolean>,
  inPlay: Uint8Array | undefined,
  candidates: number,
): number {
  const masks = rules.map((r) => coverMask(rows, r))
  let C = 0
  let U = 0
  let fp = 0
  let fn = 0
  for (let i = 0; i < rows.length; i++) {
    if (inPlay && !inPlay[i]) continue
    const covered = masks.some((m) => m[i])
    if (covered) {
      C++
      if (!positive[i]) fp++
    } else {
      U++
      if (positive[i]) fn++
    }
  }
  return rules.reduce((a, r) => a + ruleBits(r.length, candidates), 0) + exceptionBits(C, U, fp, fn)
}

/**
 * RIPPER$k$ (see the file comment): IREP* then `optimisations` passes, class by class from the least frequent. A
 * plain function of plain data and a stream, so it can run in a worker.
 *
 * @param rows The training rows.
 * @param y The class of each row, integers $0, \dots, K - 1$.
 * @param conditions The candidate conditions.
 * @param s The random stream for the grow and prune splits (each split draws from its own child stream).
 * @param options The split share, rule length, slack, optimisation passes and positive class.
 * @returns The ordered rules, the default class and the trace.
 *
 * @example Rules for a band on one feature
 * const rows = Array.from({ length: 40 }, (_, i) => [i % 10, Math.floor(i / 10)])
 * const y = rows.map(([a]) => (a >= 3 && a <= 6 ? 1 : 0))
 * const result = ripper(rows, y, thresholdConditions(rows, 10), stream(1))
 * for (const r of result.rules) print(ruleText(r.conditions), '->', r.label)
 * print('else', result.defaultLabel)
 */
export function ripper(
  rows: RuleRows,
  y: ArrayLike<number>,
  conditions: readonly Predicate[],
  s: Stream,
  options: RipperOptions = {},
): RipperResult {
  const n = rows.length
  let K = 2
  for (let i = 0; i < n; i++) K = Math.max(K, y[i] + 1)
  const freq = new Array<number>(K).fill(0)
  for (let i = 0; i < n; i++) freq[y[i]]++
  let order = Array.from({ length: K }, (_, k) => k).sort((a, b) => freq[a] - freq[b] || a - b)
  if (K === 2 && options.positive !== undefined) order = [options.positive, 1 - options.positive]
  const masks = conditions.map((c) => coverMask(rows, [c]))
  const inPlay = new Uint8Array(n).fill(1)
  const rules: RipperRule[] = []
  const trace: RipperStep[] = []
  for (let k = 0; k < order.length - 1; k++) {
    const label = order[k]
    const positive = Array.from({ length: n }, (_, i) => y[i] === label)
    const ctx: Ctx = {
      rows,
      positive,
      conditions,
      masks,
      maxConditions: options.maxConditions ?? 6,
      label,
      trace,
    }
    const set = learnClass(ctx, inPlay.slice(), child(s, 'class', label), options)
    for (const r of set) rules.push({ conditions: r, label })
    const cover = set.map((r) => coverMask(rows, r))
    for (let i = 0; i < n; i++) if (cover.some((m) => m[i])) inPlay[i] = 0
  }
  return { rules, defaultLabel: order[order.length - 1], trace }
}

/**
 * Apply RIPPER's ordered rules: the first rule a row meets gives its class, else the default.
 *
 * @param result The rules and the default class, as `ripper` returns them.
 * @param rows The rows to classify.
 * @returns One class per row.
 *
 * @example Learn, then classify new rows
 * const rows = Array.from({ length: 40 }, (_, i) => [i % 10])
 * const y = rows.map(([a]) => (a >= 5 ? 1 : 0))
 * const result = ripper(rows, y, thresholdConditions(rows, 10), stream(2))
 * print(applyRipper(result, [[1], [7]]))
 */
export function applyRipper(result: RipperResult, rows: RuleRows): Int32Array {
  return applyClassRules(result.rules, result.defaultLabel, rows)
}

/** What the phases of one class share. */
type Ctx = {
  /** The training rows. */
  rows: RuleRows
  /** Whether each row is of the class being learned. */
  positive: boolean[]
  /** The candidate conditions. */
  conditions: readonly Predicate[]
  /** The cover of each candidate condition. */
  masks: Uint8Array[]
  /** The most conditions per rule. */
  maxConditions: number
  /** The class being learned. */
  label: number
  /** The trace, appended to. */
  trace: RipperStep[]
}

/**
 * The rule set of one class: IREP*, deletion by DL, then the optimisation passes.
 *
 * @param ctx The class's rows, conditions and trace.
 * @param problem The rows of the class's problem (those no earlier class's rules cover); not modified.
 * @param s The class's stream.
 * @param options The options of `ripper`.
 * @returns The class's rules.
 */
function learnClass(ctx: Ctx, problem: Uint8Array, s: Stream, options: RipperOptions): Predicate[][] {
  const { rows, positive, label, trace } = ctx
  const growFraction = options.growFraction ?? 2 / 3
  const slack = options.slack ?? 64
  const c = ctx.conditions.length
  const dl = (rs: readonly Predicate[][]) => ripperDescriptionLength(rs, rows, positive, problem, c)
  let set = irepStar(ctx, [], problem.slice(), child(s, 'build'), growFraction, slack, dl)
  set = deleteByDl(ctx, set, dl)
  const passes = options.optimisations ?? 1
  for (let pass = 0; pass < passes; pass++) {
    const ps = child(s, 'optimise', pass)
    for (let i = 0; i < set.length; i++) {
      const split = splitRows(problem, growFraction, child(ps, 'split', i))
      // Grow on the growing rows not covered by the rules before R_i (those rules decide them).
      const before = set.slice(0, i).map((r) => coverMask(rows, r))
      const growRows = new Uint8Array(rows.length)
      for (let j = 0; j < rows.length; j++) if (split[j] === 1 && !before.some((m) => m[j])) growRows[j] = 1
      const others = set.filter((_, j) => j !== i)
      const replacement = pruneForRuleSet(ctx, grow(ctx, [], growRows), others, split)
      const revision = pruneForRuleSet(ctx, grow(ctx, set[i], growRows), others, split)
      const variants = { original: set[i], replacement, revision }
      const dls = {
        original: dl(set),
        replacement: dl(set.map((r, j) => (j === i ? replacement : r))),
        revision: dl(set.map((r, j) => (j === i ? revision : r))),
      }
      const chosen = (['original', 'replacement', 'revision'] as const).reduce((a, b) => (dls[b] < dls[a] ? b : a))
      trace.push({ phase: 'optimise', label, rule: i, original: set[i], replacement, revision, dls, chosen })
      set = set.map((r, j) => (j === i ? variants[chosen] : r))
    }
    // Cover the positives the optimised set leaves uncovered.
    const left = problem.slice()
    const cover = set.map((r) => coverMask(rows, r))
    for (let j = 0; j < rows.length; j++) if (cover.some((m) => m[j])) left[j] = 0
    set = irepStar(ctx, set, left, child(ps, 'build'), growFraction, slack, dl)
    set = deleteByDl(ctx, set, dl)
  }
  trace.push({ phase: 'done', label, rules: set, dl: dl(set) })
  return set
}

/**
 * IREP*: add grown and pruned rules to `start` while positives remain in play and the DL stays within `slack` bits
 * of the smallest so far (at most 100 rules).
 *
 * @param ctx The class's rows, conditions and trace.
 * @param start The rules to extend; not modified.
 * @param active The rows in play; rows a new rule covers are cleared from it.
 * @param s The stream the splits draw from (one child per rule).
 * @param growFraction The share of the rows in play used for growing.
 * @param slack The bits by which the DL may exceed its minimum before learning stops.
 * @param dl The description length of a rule set.
 * @returns The extended rule set.
 */
function irepStar(
  ctx: Ctx,
  start: Predicate[][],
  active: Uint8Array,
  s: Stream,
  growFraction: number,
  slack: number,
  dl: (rs: readonly Predicate[][]) => number,
): Predicate[][] {
  const set = start.slice()
  let minDl = dl(set)
  for (let r = 0; r < 100; r++) {
    let pos = 0
    for (let i = 0; i < active.length; i++) if (active[i] && ctx.positive[i]) pos++
    if (pos === 0) break
    const split = splitRows(active, growFraction, child(s, 'split', r))
    const growRows = split.map((v) => (v === 1 ? 1 : 0))
    const pruneRows = split.map((v) => (v === 2 ? 1 : 0))
    const index = set.length
    const grown = grow(ctx, [], growRows, (conds) =>
      ctx.trace.push({
        phase: 'grow',
        label: ctx.label,
        rule: index,
        conditions: conds,
        split,
        grow: counts(ctx, coverMask(ctx.rows, conds), growRows),
        prune: counts(ctx, coverMask(ctx.rows, conds), pruneRows),
      }),
    )
    if (grown.length === 0) {
      ctx.trace.push({
        phase: 'reject',
        label: ctx.label,
        rule: index,
        conditions: grown,
        dl: dl(set),
        minDl,
        active: active.slice(),
        reason: 'no condition has positive gain',
      })
      break
    }
    const candidates = prefixes(grown)
      .filter((r) => r.length > 0)
      .map((conditions) => {
        const c = counts(ctx, coverMask(ctx.rows, conditions), pruneRows)
        return { conditions, ...c, value: ripperPruneValue({ ...c, P: 0, N: 0 }) }
      })
    let kept = 0
    candidates.forEach((cand, k) => {
      if (cand.value >= candidates[kept].value) kept = k
    })
    ctx.trace.push({ phase: 'prune', label: ctx.label, rule: index, candidates, kept, split })
    const rule = candidates[kept].conditions
    const pc = candidates[kept]
    const errorRate = pc.p + pc.n > 0 ? pc.n / (pc.p + pc.n) : 1
    if (errorRate >= 0.5) {
      ctx.trace.push({
        phase: 'reject',
        label: ctx.label,
        rule: index,
        conditions: rule,
        dl: dl(set),
        minDl,
        active: active.slice(),
        reason: 'error rate on the pruning set ≥ 50%',
      })
      break
    }
    set.push(rule)
    const value = dl(set)
    const cover = coverMask(ctx.rows, rule)
    for (let i = 0; i < active.length; i++) if (cover[i]) active[i] = 0
    const over = value > minDl + slack
    minDl = Math.min(minDl, value)
    ctx.trace.push({
      phase: 'accept',
      label: ctx.label,
      rule: index,
      conditions: rule,
      dl: value,
      minDl,
      active: active.slice(),
      reason: over ? `DL exceeds the smallest by more than ${slack} bits` : undefined,
    })
    if (over) break
  }
  return set
}

/**
 * Delete rules, last first, while a deletion lowers the DL; each deletion is traced.
 *
 * @param ctx The class's trace and label.
 * @param set The rule set; not modified.
 * @param dl The description length of a rule set.
 * @returns The rule set after the deletions.
 */
function deleteByDl(ctx: Ctx, set: Predicate[][], dl: (rs: readonly Predicate[][]) => number): Predicate[][] {
  let out = set.slice()
  for (let i = out.length - 1; i >= 0; i--) {
    const without = out.filter((_, j) => j !== i)
    const value = dl(without)
    if (value < dl(out)) {
      ctx.trace.push({ phase: 'delete', label: ctx.label, conditions: out[i], dl: value })
      out = without
    }
  }
  return out
}

/**
 * Grow a rule from `start` on the marked rows by FOIL gain (`foilGain` of `aifn-compute/logic/induction`, with
 * $t = p$), until it covers no negative, no condition gains, or it has `maxConditions` conditions.
 *
 * @param ctx The class's rows, conditions and masks.
 * @param start The conditions to start from; not modified.
 * @param rowsIn The growing rows.
 * @param onStep Called with the rule at the start (when `start` is empty) and after each added condition.
 * @returns The grown rule's conditions, in the order added.
 */
function grow(
  ctx: Ctx,
  start: readonly Predicate[],
  rowsIn: Uint8Array,
  onStep?: (c: Predicate[]) => void,
): Predicate[] {
  let conds = start.slice()
  let mask = andMask(coverMask(ctx.rows, conds), rowsIn)
  let cur = counts(ctx, mask, rowsIn)
  if (onStep && conds.length === 0) onStep(conds)
  while (cur.n > 0 && conds.length < ctx.maxConditions) {
    let best = -1
    let bestGain = 0
    let bestMask: Uint8Array | null = null
    let bestConds: Predicate[] = conds
    ctx.conditions.forEach((c, j) => {
      // Conditions stay in the order they were added (no merging), so pruning can delete a final sequence of them.
      const m = andMask(mask, ctx.masks[j])
      const k = counts(ctx, m, rowsIn)
      if (k.p === cur.p && k.n === cur.n) return
      const gain = foilGain(cur.p, cur.n, k.p, k.n, k.p)
      if (gain > bestGain + 1e-12) {
        best = j
        bestGain = gain
        bestMask = m
        bestConds = [...conds, c]
      }
    })
    if (best < 0 || !bestMask) break
    conds = bestConds
    mask = bestMask
    cur = counts(ctx, mask, rowsIn)
    onStep?.(conds)
  }
  return conds
}

/**
 * Prune a rule for the optimisation pass: of its non-empty prefixes, the one that maximises the accuracy of the rule
 * set (the other rules and this one) on the pruning rows, the shortest on a tie.
 *
 * @param ctx The class's rows and labels.
 * @param grown The grown rule.
 * @param others The other rules of the set.
 * @param split The split: pruning rows are marked 2.
 * @returns The kept prefix.
 */
function pruneForRuleSet(ctx: Ctx, grown: Predicate[], others: Predicate[][], split: Uint8Array): Predicate[] {
  const otherMasks = others.map((r) => coverMask(ctx.rows, r))
  let best = grown
  let bestAcc = -1
  for (const cand of prefixes(grown)) {
    if (cand.length === 0) continue
    const m = coverMask(ctx.rows, cand)
    let right = 0
    let total = 0
    for (let i = 0; i < ctx.rows.length; i++) {
      if (split[i] !== 2) continue
      total++
      const predicted = m[i] || otherMasks.some((o) => o[i])
      if (Boolean(predicted) === ctx.positive[i]) right++
    }
    const acc = total > 0 ? right / total : 0
    if (acc >= bestAcc) {
      bestAcc = acc
      best = cand
    }
  }
  return best
}

/**
 * The rule and its prefixes, longest first (the empty prefix last).
 *
 * @param rule The conditions, in the order added.
 * @returns The prefixes.
 */
function prefixes(rule: readonly Predicate[]): Predicate[][] {
  const out: Predicate[][] = []
  for (let k = rule.length; k >= 0; k--) out.push(rule.slice(0, k))
  return out
}

/**
 * The positives and negatives a mask covers among the marked rows.
 *
 * @param ctx The class's labels.
 * @param mask The cover.
 * @param rowsIn The rows to count.
 * @returns `p` and `n`.
 */
function counts(ctx: Ctx, mask: Uint8Array, rowsIn: Uint8Array): { p: number; n: number } {
  let p = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) {
    if (!rowsIn[i] || !mask[i]) continue
    if (ctx.positive[i]) p++
    else n++
  }
  return { p, n }
}

/**
 * A random split of the marked rows into growing and pruning rows, by a Fisher-Yates shuffle that draws one integer
 * per swap from the stream (kept as it is, rather than `shuffle`, so that seeded traces stay the same).
 *
 * @param active The rows to split.
 * @param growFraction The share of them that grow.
 * @param s The stream.
 * @returns 1 for a growing row, 2 for a pruning row, 0 for a row out of play.
 */
function splitRows(active: Uint8Array, growFraction: number, s: Stream): Uint8Array {
  const idx: number[] = []
  for (let i = 0; i < active.length; i++) if (active[i]) idx.push(i)
  for (let i = idx.length - 1; i > 0; i--) {
    const j = integers(s, i + 1)
    const t = idx[i]
    idx[i] = idx[j]
    idx[j] = t
  }
  const out = new Uint8Array(active.length)
  const g = Math.round(growFraction * idx.length)
  idx.forEach((row, k) => (out[row] = k < g ? 1 : 2))
  return out
}
