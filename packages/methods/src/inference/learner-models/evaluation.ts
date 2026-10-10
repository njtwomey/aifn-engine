/**
 * Evaluation of learner models for equity: parameter recovery, held-out predictive scores, ability bias by group, and
 * how well structural-zero posteriors find the context-caused zeros.
 *
 * The measures are those of Twomey et al. (2022), §3.3: how well a model recovers the true abilities, difficulties and
 * discriminations (Pearson and Spearman correlations, Table 3 of the paper), how well it predicts held-out responses
 * (accuracy, $F_1$, negative log-likelihood and Brier score, Table 2), and whether its ability estimates are biased
 * against a subpopulation. The equity measures are per group: the mean signed error of $\hat\theta$ (bias) and its
 * RMSE, and the equity gap, the mean bias of students with a neurodivergent condition minus that of students without.
 * A model that reads context-caused zeros as low ability has a negative gap.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { accuracy, auroc, brierScore, f1, logLoss } from 'aifn-compute/learning/metrics'
import { correlation, spearman } from 'aifn-compute/probability/stats'

/** Bias and error of the ability estimates in one group. */
export interface GroupAbilityError {
  /** The group's label. */
  readonly group: number
  /** The number of students in the group. */
  readonly size: number
  /** The mean of $\hat\theta - \theta$ over the group (NaN for an empty group). */
  readonly bias: number
  /** The root mean squared error of $\hat\theta$ over the group (NaN for an empty group). */
  readonly rmse: number
}

/** Ability error by group, and the equity gap. */
export interface AbilityEquity {
  /** One entry per group label, from 0 to the largest. */
  readonly groups: GroupAbilityError[]
  /**
   * The mean bias over students in any group but `reference`, minus the reference group's mean bias (NaN when either
   * side is empty).
   */
  readonly gap: number
  /** The root mean squared error over all students. */
  readonly rmse: number
}

/**
 * The bias and RMSE of ability estimates within each group (labels $0, \dots, G - 1$), and the equity gap against the
 * `reference` group (default 0, students without a condition). Throws `DomainError` unless the three arrays have the
 * same length.
 *
 * @param estimate The estimated abilities $\hat\theta_p$, one per student.
 * @param truth The true abilities $\theta_p$, one per student.
 * @param group Each student's group label, a non-negative integer.
 * @param reference The label of the reference group the gap is measured against.
 * @returns The bias and RMSE per group, the gap and the overall RMSE.
 *
 * @example Students with a condition (group 1) are underestimated
 * const eq = abilityEquity([0.1, -0.2, -0.8, -0.6], [0, 0, 0, 0], [0, 0, 1, 1])
 * print('groups:', eq.groups)
 * print('gap:', eq.gap, 'rmse:', eq.rmse)
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
  /** The Pearson (linear) correlation. */
  readonly pearson: number
  /** The Spearman (rank) correlation. */
  readonly spearman: number
}

/**
 * Pearson and Spearman correlation of an estimate with the truth (`correlation` and `spearman` of
 * `aifn-compute/probability/stats`).
 *
 * @param estimate The estimated parameters.
 * @param truth The true parameters, in the same order.
 * @returns Both correlations.
 *
 * @example The right order, not quite the right scale
 * print(parameterRecovery([1, 2, 3, 5], [1, 2, 3, 4]))
 */
export function parameterRecovery(estimate: ArrayLike<number>, truth: ArrayLike<number>): ParameterRecovery {
  const e = Array.from(estimate)
  const t = Array.from(truth)
  return { pearson: correlation(e, t), spearman: spearman(e, t) }
}

/** Predictive scores of probabilities of a correct answer against observed responses (Table 2 of the paper). */
export interface ResponseScores {
  /** The number of responses scored. */
  readonly n: number
  /** Accuracy at the 0.5 threshold. */
  readonly accuracy: number
  /** $F_1$ at the 0.5 threshold, with a correct answer as the positive class. */
  readonly f1: number
  /** The mean negative log-likelihood, in nats. */
  readonly nll: number
  /** The Brier score. */
  readonly brier: number
}

/**
 * Predictive scores on the responses selected by `mask`, NaN responses skipped. Probabilities are clipped to
 * $[10^{-12}, 1 - 10^{-12}]$ for the log-likelihood, and a probability of at least 0.5 predicts a correct answer. With
 * nothing to score, `n` is 0 and the scores are NaN.
 *
 * @param responses The responses, 0, 1 or NaN (row-major over the response matrix).
 * @param probabilities The predicted $\pr(Y = 1)$, in the same layout.
 * @param mask Which responses to score (non-zero: scored), in the same layout; all of them when left out.
 * @returns The count, accuracy, $F_1$, mean negative log-likelihood and Brier score.
 *
 * @example Three answered items, one not attempted
 * print(responseScores([1, 0, 1, NaN], [0.8, 0.3, 0.4, 0.9]))
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
 * the AUROC of the posterior for the truth. NaN unless both kinds of zero are present.
 *
 * @param posterior The posterior that each zero is structural (`structuralZeroPosterior`).
 * @param structural The truth for each zero: 1 structural, 0 an incorrect answer.
 * @returns The AUROC: 1 when every structural zero has a higher posterior than every incorrect answer, 0.5 for chance.
 *
 * @example A perfect separation, and a single swap
 * print(structuralZeroAuroc([0.9, 0.2, 0.7, 0.1], [1, 0, 1, 0]))
 * print(structuralZeroAuroc([0.9, 0.2, 0.1, 0.7], [1, 0, 1, 0]))
 */
export function structuralZeroAuroc(posterior: ArrayLike<number>, structural: ArrayLike<number>): number {
  const y = Array.from(structural)
  if (!y.some((v) => v === 1) || !y.some((v) => v === 0)) return NaN
  return auroc(y, Array.from(posterior))
}
