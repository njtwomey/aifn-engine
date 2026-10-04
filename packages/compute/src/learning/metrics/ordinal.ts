/**
 * Metrics of ordinal classification, part of `aifn-compute/learning/metrics`: they take ordered labels, class probabilities or
 * scores and name no model. On decided labels: ordinal mean absolute error, its macro average over classes, accuracy
 * within a tolerance, and Cohen's kappa with quadratic weights. On probabilities: the ranked probability score. On a
 * latent score: the ordinal concordance index (C-index).
 */

import { confusionMatrix } from './confusion'
import {
  classesOf,
  defineMetric,
  dense,
  encodeLabels,
  labelList,
  nonEmpty,
  sameLength,
  values,
  type Data,
  type Label,
  type Labels,
  type Rows,
} from './core'
import { kappaFromTable } from './classification'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// ── Ordinal ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Class indices 0 … K − 1 of the truth and prediction in the ordinal order `labels` (default the sorted labels). */
function ordinalIndices(yTrue: Labels, yPred: Labels, labels?: readonly Label[]) {
  const t = labelList(yTrue)
  const p = labelList(yPred)
  sameLength(t, p, 'ordinal metric')
  nonEmpty(t.length, 'ordinal metric')
  const classes = labels ? [...labels] : classesOf(t, p)
  const ti = encodeLabels(t, classes)
  const pi = encodeLabels(p, classes)
  if (ti.includes(-1) || pi.includes(-1))
    throw new DomainError('metrics', 'metrics: ordinal metric: a label is missing from `labels`')
  return { ti, pi, classes }
}

/**
 * Ordinal mean absolute error on the class index, (1/n) Σ |ĵᵢ − jᵢ|, with classes numbered in the order `labels`
 * (default the sorted labels, so the steps between neighbouring classes cost 1) (ordinal-classification-metrics).
 */
export const ordinalMeanAbsoluteError = defineMetric(
  {
    key: 'ordinalMeanAbsoluteError',
    stability: 'stable',
    name: 'Ordinal mean absolute error',
    inputs: 'labels',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[] } = {}): number => {
    const { ti, pi } = ordinalIndices(yTrue, yPred, options.labels)
    let s = 0
    for (let i = 0; i < ti.length; i++) s += Math.abs(ti[i] - pi[i])
    return s / ti.length
  },
)

/**
 * Macro-averaged ordinal MAE (Baccianella et al. 2009): the MAE within each true class, averaged over the classes
 * present in `yTrue`.
 */
export const macroMeanAbsoluteError = defineMetric(
  {
    key: 'macroMeanAbsoluteError',
    stability: 'stable',
    name: 'Macro-averaged MAE',
    inputs: 'labels',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[] } = {}): number => {
    const { ti, pi, classes } = ordinalIndices(yTrue, yPred, options.labels)
    const sums = new Float64Array(classes.length)
    const counts = new Float64Array(classes.length)
    for (let i = 0; i < ti.length; i++) {
      sums[ti[i]] += Math.abs(ti[i] - pi[i])
      counts[ti[i]]++
    }
    let s = 0
    let K = 0
    for (let k = 0; k < classes.length; k++)
      if (counts[k] > 0) {
        s += sums[k] / counts[k]
        K++
      }
    return s / K
  },
)

/** Accuracy within `tolerance` classes (default 1): the fraction with |ĵ − j| ≤ tolerance on the class index. */
export const withinToleranceAccuracy = defineMetric(
  {
    key: 'withinToleranceAccuracy',
    stability: 'stable',
    name: 'Accuracy within one class',
    inputs: 'labels',
    direction: 'higher',
    range: [0, 1],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[]; tolerance?: number } = {}): number => {
    const { ti, pi } = ordinalIndices(yTrue, yPred, options.labels)
    const tol = options.tolerance ?? 1
    let hit = 0
    for (let i = 0; i < ti.length; i++) if (Math.abs(ti[i] - pi[i]) <= tol) hit++
    return hit / ti.length
  },
)

/** Quadratic weighted κ: Cohen's κ with weights (j − k)² on the ordinal class index (ordinal-classification-metrics). */
export const quadraticWeightedKappa = defineMetric(
  {
    key: 'quadraticWeightedKappa',
    stability: 'stable',
    name: 'Quadratic weighted κ',
    inputs: 'labels',
    direction: 'higher',
    range: [-1, 1],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[] } = {}): number =>
    kappaFromTable(confusionMatrix(yTrue, yPred, { labels: options.labels }).matrix, 'quadratic'),
)

// ── Probabilities and scores ─────────────────────────────────────────────────────────────────────────────────────────

/** Class indices 0 … K − 1 of the truth in the ordinal order `labels` (default the sorted labels). */
function truthIndex(yTrue: Labels, labels: readonly Label[] | undefined, what: string) {
  const t = labelList(yTrue)
  if (t.length === 0) throw new DomainError('metrics', `metrics: ${what}: no cases`)
  const classes = labels ? [...labels] : classesOf(t)
  const index = encodeLabels(t, classes)
  if (index.includes(-1)) throw new DomainError('metrics', `metrics: ${what}: a label is missing from \`labels\``)
  return { index, classes }
}

/**
 * The ranked probability score (Epstein, 1969; Murphy, 1971): for a predicted distribution q over K ordered classes
 * and the observed class y, (1/(K − 1)) Σ_{k<K} (Q_k − 1[y ≤ k])², with Q_k = q₁ + … + q_k the predicted cdf,
 * averaged over cases. It is the Brier score of the K − 1 cumulative events "y ≤ k", so it is strictly proper and,
 * unlike the multiclass Brier score, it charges more for probability far from the observed class. For K = 2 it equals
 * the binary Brier score (scikit-learn's `brier_score_loss`); scikit-learn has no ranked probability score. Columns
 * of `probabilities` [n, K] follow `labels` (default the sorted labels of `yTrue`, which must then all occur).
 */
export const rankedProbabilityScore = defineMetric(
  {
    key: 'rankedProbabilityScore',
    stability: 'stable',
    name: 'Ranked probability score',
    inputs: 'probabilities',
    direction: 'lower',
    range: [0, 1],
    notes: ['ordinal-classification-metrics', 'proper-scoring-rule'],
    capability: 'predictive',
  },
  (yTrue: Labels, probabilities: Rows | Data, options: { labels?: readonly Label[] } = {}): number => {
    const { index, classes } = truthIndex(yTrue, options.labels, 'rankedProbabilityScore')
    const P = dense(probabilities, 'rankedProbabilityScore probabilities')
    const n = index.length
    const K = P.cols
    if (P.rows !== n)
      throw new ShapeError('metrics', `metrics: rankedProbabilityScore: ${P.rows} probability rows for ${n} cases`)
    if (K !== classes.length)
      throw new ShapeError(
        'metrics',
        `metrics: rankedProbabilityScore: ${K} probability columns for ${classes.length} classes`,
      )
    if (K < 2) throw new DomainError('metrics', 'metrics: rankedProbabilityScore: needs at least two classes')
    let s = 0
    for (let i = 0; i < n; i++) {
      let Q = 0
      for (let k = 0; k < K - 1; k++) {
        Q += P.data[i * K + k]
        s += (Q - (index[i] <= k ? 1 : 0)) ** 2
      }
    }
    return s / (n * (K - 1))
  },
)

/**
 * The ordinal concordance index (Harrell's C for an ordinal outcome; Waegeman, De Baets and Boullart, 2008, "ROC
 * analysis in ordinal regression learning"): over all pairs of cases with different true classes, the fraction whose
 * scores are ordered as their classes, a tie in score counting ½. Equivalently the pair-count-weighted mean of the
 * pairwise AUCs between classes, and (Somers' D of the score given the class + 1)/2 (scipy's `stats.somersd`). For
 * two classes it is the AUROC (scikit-learn's `roc_auc_score`). Higher scores must mean higher classes. Computed in
 * O(n log n + nK) by sorting the scores.
 */
export const ordinalConcordanceIndex = defineMetric(
  {
    key: 'ordinalConcordanceIndex',
    stability: 'stable',
    name: 'Ordinal C-index',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['ordinal-classification-metrics'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data, options: { labels?: readonly Label[] } = {}): number => {
    const { index, classes } = truthIndex(yTrue, options.labels, 'ordinalConcordanceIndex')
    const s = values(scores)
    sameLength(index, s, 'ordinalConcordanceIndex')
    const n = index.length
    const K = classes.length
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[a] - s[b])
    // below[c]: cases of class c with a strictly smaller score than the current block.
    const below = new Float64Array(K)
    const block = new Float64Array(K)
    let concordant = 0
    for (let start = 0; start < n;) {
      let end = start
      while (end < n && s[order[end]] === s[order[start]]) end++
      block.fill(0)
      for (let r = start; r < end; r++) block[index[order[r]]]++
      for (let c = 0; c < K; c++) {
        if (block[c] === 0) continue
        let lower = 0
        let lowerTied = 0
        for (let c2 = 0; c2 < c; c2++) {
          lower += below[c2]
          lowerTied += block[c2]
        }
        // Pairs (j, i) with class(j) < class(i) = c: concordant when s_j < s_i, half when tied.
        concordant += block[c] * (lower + 0.5 * lowerTied)
      }
      for (let c = 0; c < K; c++) below[c] += block[c]
      start = end
    }
    const counts = new Float64Array(K)
    for (let i = 0; i < n; i++) counts[index[i]]++
    let same = 0
    for (let c = 0; c < K; c++) same += counts[c] * counts[c]
    const pairs = (n * n - same) / 2
    if (pairs === 0)
      throw new DomainError('metrics', 'metrics: ordinalConcordanceIndex: needs cases of at least two classes')
    return concordant / pairs
  },
)
