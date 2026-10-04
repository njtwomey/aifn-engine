/**
 * Evaluation of learner models for equity (Twomey et al., 2022, §3.3): how well a model recovers the true abilities,
 * difficulties and discriminations (Pearson and Spearman correlations, Table 3 of the paper), how well it predicts held-out
 * responses (accuracy, F₁, negative log-likelihood and Brier score, Table 2), and whether its ability estimates are
 * biased against a subpopulation. The equity measures are per group: the mean signed error of θ̂ (bias) and its RMSE,
 * and the equity gap, the mean bias of students with a neurodivergent condition minus that of students without. A model
 * that reads context-caused zeros as low ability has a negative gap.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { accuracy, auroc, brierScore, f1, logLoss } from 'aifn-compute/learning/metrics'
import { correlation, spearman } from 'aifn-compute/probability/stats'

/** Bias and error of the ability estimates in one group. */
export interface GroupAbilityError {
  readonly group: number
  readonly size: number
  /** Mean of θ̂ − θ, and the root mean squared error. */
  readonly bias: number
  readonly rmse: number
}

/** Ability error by group, and the equity gap. */
export interface AbilityEquity {
  readonly groups: GroupAbilityError[]
  /** Mean bias over students in any group but `reference`, minus the reference group's mean bias. */
  readonly gap: number
  readonly rmse: number
}

/**
 * The bias and RMSE of ability estimates within each group (labels 0 … G − 1), and the equity gap against the
 * `reference` group (default 0, students without a condition).
 */
export function abilityEquity(
  estimate: ArrayLike<number>,
  truth: ArrayLike<number>,
  group: ArrayLike<number>,
  reference = 0,
): AbilityEquity {
  const n = estimate.length
  if (truth.length !== n || group.length !== n)
    throw new DomainError('abilityEquity', 'abilityEquity: estimate, truth and group need one entry per student')
  let G = 0
  for (let p = 0; p < n; p++) G = Math.max(G, group[p] + 1)
  const sum = new Float64Array(G)
  const sq = new Float64Array(G)
  const count = new Float64Array(G)
  let refSum = 0
  let refCount = 0
  let otherSum = 0
  let otherCount = 0
  let total = 0
  for (let p = 0; p < n; p++) {
    const e = estimate[p] - truth[p]
    sum[group[p]] += e
    sq[group[p]] += e * e
    count[group[p]]++
    total += e * e
    if (group[p] === reference) {
      refSum += e
      refCount++
    } else {
      otherSum += e
      otherCount++
    }
  }
  const groups = Array.from({ length: G }, (_, g) => ({
    group: g,
    size: count[g],
    bias: count[g] ? sum[g] / count[g] : NaN,
    rmse: count[g] ? Math.sqrt(sq[g] / count[g]) : NaN,
  }))
  const gap = refCount && otherCount ? otherSum / otherCount - refSum / refCount : NaN
  return { groups, gap, rmse: Math.sqrt(total / Math.max(1, n)) }
}

/** Pearson and Spearman correlations between true and recovered parameters (Table 3 of the paper). */
export interface ParameterRecovery {
  readonly pearson: number
  readonly spearman: number
}

/** Pearson and Spearman correlation of an estimate with the truth. */
export function parameterRecovery(estimate: ArrayLike<number>, truth: ArrayLike<number>): ParameterRecovery {
  const e = Array.from(estimate)
  const t = Array.from(truth)
  return { pearson: correlation(e, t), spearman: spearman(e, t) }
}

/** Predictive scores of probabilities of a correct answer against observed responses (Table 2 of the paper). */
export interface ResponseScores {
  readonly n: number
  /** Accuracy and F₁ (correct = positive) at the 0.5 threshold. */
  readonly accuracy: number
  readonly f1: number
  /** Mean negative log-likelihood (nats) and Brier score. */
  readonly nll: number
  readonly brier: number
}

/**
 * Predictive scores on the responses selected by `mask` (row-major over the response matrix; NaN responses are
 * skipped). `probabilities` are Pr(Y = 1) in the same layout.
 */
export function responseScores(
  responses: ArrayLike<number>,
  probabilities: ArrayLike<number>,
  mask?: ArrayLike<number>,
): ResponseScores {
  const y: number[] = []
  const q: number[] = []
  for (let k = 0; k < responses.length; k++)
    if (Number.isFinite(responses[k]) && (!mask || mask[k])) {
      y.push(responses[k])
      q.push(Math.min(1 - 1e-12, Math.max(1e-12, probabilities[k])))
    }
  if (!y.length) return { n: 0, accuracy: NaN, f1: NaN, nll: NaN, brier: NaN }
  const hard = q.map((v) => (v >= 0.5 ? 1 : 0))
  return {
    n: y.length,
    accuracy: accuracy(y, hard),
    f1: f1(y, hard),
    nll: logLoss(y, q),
    brier: brierScore(y, q),
  }
}

/**
 * How well structural-zero posteriors separate the zeros that were structural from those that were incorrect answers:
 * the AUROC of the posterior for the truth (1 structural), over the zeros selected.
 */
export function structuralZeroAuroc(posterior: ArrayLike<number>, structural: ArrayLike<number>): number {
  const y = Array.from(structural)
  if (!y.some((v) => v === 1) || !y.some((v) => v === 0)) return NaN
  return auroc(y, Array.from(posterior))
}
