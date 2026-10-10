/**
 * Group fairness metrics (group-fairness-metrics): per-group rates of a binary classifier and the differences and
 * ratios between groups that demographic parity, equal opportunity (Hardt et al. 2016), equalised odds and predictive
 * parity compare.
 *
 * Every metric takes the true labels $Y$, the predicted labels $\hat{Y}$ and, in its options, the group $A$ of each
 * case; the rates are computed for each group $a$ by `groupRates`. With more than two groups, a difference is the
 * largest minus the smallest group value. A rate a group cannot define (a true-positive rate in a group with no
 * positives) is NaN, and so is any difference or ratio that uses it.
 */

import { classesOf, divide, labelList, positiveOf, sameLength } from 'aifn-compute/learning/metrics'
import { defineMetric, type Label, type Labels } from 'aifn-compute/learning/metrics'

/** The confusion rates of one group $a$, each NaN when its denominator is 0. */
export type GroupRates = {
  /** The group's label $a$. */
  group: Label
  /** The number of cases in the group. */
  n: number
  /** The base rate $P(Y = 1 \mid A = a)$. */
  baseRate: number
  /** The selection rate $P(\hat{Y} = 1 \mid A = a)$. */
  selectionRate: number
  /** The true-positive rate $P(\hat{Y} = 1 \mid Y = 1, A = a)$. */
  truePositiveRate: number
  /** The false-positive rate $P(\hat{Y} = 1 \mid Y = 0, A = a)$. */
  falsePositiveRate: number
  /** The precision $P(Y = 1 \mid \hat{Y} = 1, A = a)$. */
  precision: number
}

/**
 * Options of the fairness metrics: `groups`, the group (protected attribute value) of each case, one per label; and
 * `positive`, the positive class (default `1` or `true` when present, else the last class in label order, as
 * `positiveOf` chooses it from the true and predicted labels).
 */
export type FairnessOptions = { groups: Labels; positive?: Label }

/**
 * The confusion rates of each group of a protected attribute: base rate, selection rate, true- and false-positive
 * rates and precision, counted from the cases of that group alone. Labels and groups of different lengths throw.
 *
 * @param yTrue The true labels, one per case.
 * @param yPred The predicted labels, matching `yTrue`. Every label other than the positive class counts as negative.
 * @param options `groups`, the group of each case, and `positive`, the positive class (see `FairnessOptions`).
 * @returns One `GroupRates` per distinct group, in label order (as `classesOf` sorts them).
 *
 * @example Rates of two groups
 * const yTrue = [1, 1, 0, 0, 1, 1, 0, 0]
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
 * for (const r of groupRates(yTrue, yPred, { groups })) print(r)
 */
export function groupRates(yTrue: Labels, yPred: Labels, options: FairnessOptions): GroupRates[] {
  const t = labelList(yTrue)
  const p = labelList(yPred)
  const g = labelList(options.groups)
  sameLength(t, p, 'groupRates')
  sameLength(t, g, 'groupRates groups')
  const pos = positiveOf(classesOf(t, p), options.positive)
  return classesOf(g).map((group) => {
    let tp = 0
    let fp = 0
    let fn = 0
    let tn = 0
    for (let i = 0; i < t.length; i++) {
      if (g[i] !== group) continue
      const a = t[i] === pos
      const b = p[i] === pos
      if (a && b) tp++
      else if (b) fp++
      else if (a) fn++
      else tn++
    }
    const n = tp + fp + fn + tn
    return {
      group,
      n,
      baseRate: divide(tp + fn, n),
      selectionRate: divide(tp + fp, n),
      truePositiveRate: divide(tp, tp + fn),
      falsePositiveRate: divide(fp, fp + tn),
      precision: divide(tp, tp + fp),
    }
  })
}

/**
 * The spread of a list of group values: the largest minus the smallest (NaN when any value is NaN).
 *
 * @param v One value per group.
 * @returns $\max_a v_a - \min_a v_a$.
 */
const spread = (v: number[]) => Math.max(...v) - Math.min(...v)

/**
 * The metric metadata shared by the fairness metrics: module, `labels` inputs, range $[0, 1]$, the
 * `group-fairness-metrics` note and the `decide` capability.
 *
 * @param key The metric's registry key.
 * @param name The metric's display name.
 * @param direction Whether lower (a difference) or higher (a ratio) is fairer.
 * @returns The spec for `defineMetric`.
 */
const fairnessInfo = (key: string, name: string, direction: 'higher' | 'lower' = 'lower') =>
  ({
    key,
    name,
    module: 'applied/evaluation/fairness',
    inputs: 'labels',
    direction,
    range: [0, 1],
    notes: ['group-fairness-metrics'],
    capability: 'decide',
  }) as const

/**
 * Demographic parity difference: the spread $\max_a - \min_a$ of the selection rates $P(\hat{Y} = 1 \mid A = a)$
 * across groups. The true labels play no part in the rates it compares, but fix the positive class.
 *
 * @param yTrue The true labels, one per case.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `groups`, the group of each case, and `positive`, the positive class (see `FairnessOptions`).
 * @returns The difference in $[0, 1]$; 0 is parity.
 *
 * @example Group a is selected three times as often as group b
 * const yTrue = [1, 1, 0, 0, 1, 1, 0, 0]
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
 * print('demographic parity difference =', demographicParityDifference(yTrue, yPred, { groups }))
 */
export const demographicParityDifference = defineMetric(
  fairnessInfo('demographicParityDifference', 'Demographic parity difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number =>
    spread(groupRates(yTrue, yPred, options).map((r) => r.selectionRate)),
)

/**
 * Demographic parity ratio (disparate impact ratio): the smallest selection rate $P(\hat{Y} = 1 \mid A = a)$ over
 * the largest. NaN when no group has a positive prediction.
 *
 * @param yTrue The true labels, one per case.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `groups`, the group of each case, and `positive`, the positive class (see `FairnessOptions`).
 * @returns The ratio in $[0, 1]$; 1 is parity, and the "four-fifths rule" flags a ratio below $0.8$.
 *
 * @example Selection rates of $0.75$ and $0.25$
 * const yTrue = [1, 1, 0, 0, 1, 1, 0, 0]
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
 * print('demographic parity ratio =', demographicParityRatio(yTrue, yPred, { groups }))
 */
export const demographicParityRatio = defineMetric(
  fairnessInfo('demographicParityRatio', 'Demographic parity ratio', 'higher'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number => {
    const v = groupRates(yTrue, yPred, options).map((r) => r.selectionRate)
    return divide(Math.min(...v), Math.max(...v))
  },
)

/**
 * Equal opportunity difference (Hardt et al. 2016): the spread of true-positive rates
 * $P(\hat{Y} = 1 \mid Y = 1, A = a)$ across groups. NaN when a group has no positive case.
 *
 * @param yTrue The true labels, one per case.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `groups`, the group of each case, and `positive`, the positive class (see `FairnessOptions`).
 * @returns The difference in $[0, 1]$; 0 is equal opportunity.
 *
 * @example True-positive rates of 1 and $0.5$
 * const yTrue = [1, 1, 0, 0, 1, 1, 0, 0]
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
 * print('equal opportunity difference =', equalOpportunityDifference(yTrue, yPred, { groups }))
 */
export const equalOpportunityDifference = defineMetric(
  fairnessInfo('equalOpportunityDifference', 'Equal opportunity difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number =>
    spread(groupRates(yTrue, yPred, options).map((r) => r.truePositiveRate)),
)

/**
 * Equalised odds difference (Hardt et al. 2016): the larger of the spreads of true-positive rates and of
 * false-positive rates across groups. NaN when a group has no positive or no negative case.
 *
 * @param yTrue The true labels, one per case.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `groups`, the group of each case, and `positive`, the positive class (see `FairnessOptions`).
 * @returns The difference in $[0, 1]$; 0 is equalised odds.
 *
 * @example Equalised odds and equal opportunity on the same table
 * const yTrue = [1, 1, 0, 0, 1, 1, 0, 0]
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
 * print('equalised odds difference =', equalisedOddsDifference(yTrue, yPred, { groups }))
 * print('equal opportunity difference =', equalOpportunityDifference(yTrue, yPred, { groups }))
 */
export const equalisedOddsDifference = defineMetric(
  fairnessInfo('equalisedOddsDifference', 'Equalised odds difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number => {
    const r = groupRates(yTrue, yPred, options)
    return Math.max(spread(r.map((x) => x.truePositiveRate)), spread(r.map((x) => x.falsePositiveRate)))
  },
)

/**
 * Predictive parity difference: the spread of precisions $P(Y = 1 \mid \hat{Y} = 1, A = a)$ across groups. NaN when
 * a group has no positive prediction.
 *
 * @param yTrue The true labels, one per case.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `groups`, the group of each case, and `positive`, the positive class (see `FairnessOptions`).
 * @returns The difference in $[0, 1]$; 0 is predictive parity.
 *
 * @example Precisions of $2/3$ and 1
 * const yTrue = [1, 1, 0, 0, 1, 1, 0, 0]
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
 * print('predictive parity difference =', predictiveParityDifference(yTrue, yPred, { groups }))
 */
export const predictiveParityDifference = defineMetric(
  fairnessInfo('predictiveParityDifference', 'Predictive parity difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number =>
    spread(groupRates(yTrue, yPred, options).map((r) => r.precision)),
)
