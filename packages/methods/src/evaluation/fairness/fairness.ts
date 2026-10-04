/**
 * Group fairness metrics (group-fairness-metrics): per-group rates of a binary classifier and the differences and
 * ratios between groups that demographic parity, equal opportunity, equalised odds and predictive parity compare.
 * With more than two groups, a difference is the largest minus the smallest group value.
 */

import { classesOf, divide, labelList, positiveOf, sameLength } from 'aifn-compute/learning/metrics'
import { defineMetric, type Label, type Labels } from 'aifn-compute/learning/metrics'

/** The confusion rates of one group. */
export type GroupRates = {
  group: Label
  n: number
  /** P(Y = 1 | A = a). */
  baseRate: number
  /** P(Ŷ = 1 | A = a). */
  selectionRate: number
  /** P(Ŷ = 1 | Y = 1, A = a). */
  truePositiveRate: number
  /** P(Ŷ = 1 | Y = 0, A = a). */
  falsePositiveRate: number
  /** P(Y = 1 | Ŷ = 1, A = a). */
  precision: number
}

/** Options of the fairness metrics: the group of each case, and the positive class. */
export type FairnessOptions = { groups: Labels; positive?: Label }

/** The confusion rates of each group of a protected attribute. */
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

const spread = (v: number[]) => Math.max(...v) - Math.min(...v)

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

/** Demographic parity difference: the spread of positive-prediction rates P(Ŷ = 1 | A = a) across groups. */
export const demographicParityDifference = defineMetric(
  fairnessInfo('demographicParityDifference', 'Demographic parity difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number =>
    spread(groupRates(yTrue, yPred, options).map((r) => r.selectionRate)),
)

/** Disparate impact ratio: the smallest positive-prediction rate over the largest (1 is parity). */
export const demographicParityRatio = defineMetric(
  fairnessInfo('demographicParityRatio', 'Demographic parity ratio', 'higher'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number => {
    const v = groupRates(yTrue, yPred, options).map((r) => r.selectionRate)
    return divide(Math.min(...v), Math.max(...v))
  },
)

/** Equal opportunity difference: the spread of true-positive rates across groups (Hardt et al. 2016). */
export const equalOpportunityDifference = defineMetric(
  fairnessInfo('equalOpportunityDifference', 'Equal opportunity difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number =>
    spread(groupRates(yTrue, yPred, options).map((r) => r.truePositiveRate)),
)

/** Equalised odds difference: the larger of the true-positive-rate and false-positive-rate spreads across groups. */
export const equalisedOddsDifference = defineMetric(
  fairnessInfo('equalisedOddsDifference', 'Equalised odds difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number => {
    const r = groupRates(yTrue, yPred, options)
    return Math.max(spread(r.map((x) => x.truePositiveRate)), spread(r.map((x) => x.falsePositiveRate)))
  },
)

/** Predictive parity difference: the spread of precision P(Y = 1 | Ŷ = 1, A = a) across groups. */
export const predictiveParityDifference = defineMetric(
  fairnessInfo('predictiveParityDifference', 'Predictive parity difference'),
  (yTrue: Labels, yPred: Labels, options: FairnessOptions): number =>
    spread(groupRates(yTrue, yPred, options).map((r) => r.precision)),
)
