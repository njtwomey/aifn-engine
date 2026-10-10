/**
 * `aifn-methods/evaluation`: application metrics, defined with the metric registry pattern of
 * `aifn-compute/learning/metrics`.
 *
 * - `aifn-methods/evaluation/text`: BLEU, chrF, ROUGE, word and character error rates, translation edit rate, SQuAD
 *   exact match and $F_1$, BERTScore and backretrieval.
 * - `aifn-methods/evaluation/detection`: box and mask IoU, Dice, mean IoU, detection average precision and mAP,
 *   panoptic quality and the boundary F-score.
 * - `aifn-methods/evaluation/quality`: image and signal quality: PSNR, SSIM, spectral angle, ERGAS and RASE, SNR,
 *   SI-SDR and permutation-invariant scoring.
 * - `aifn-methods/evaluation/generative`: generative models from feature vectors: FID, KID, the Inception score,
 *   precision and recall, density and coverage, and CLIPScore.
 * - `aifn-methods/evaluation/fairness`: group fairness of a classifier: demographic parity, equal opportunity,
 *   equalised odds and predictive parity.
 * - `aifn-methods/evaluation/beyond-accuracy`: recommendation lists: intra-list diversity, catalogue coverage, Gini and
 *   Herfindahl concentration of exposure, and novelty.
 *
 * Every metric carries `info` (range, direction, inputs) and is collected in `evaluationMetricRegistry`, the
 * counterpart of `metricRegistry` in `aifn-compute/learning/metrics`, where the classification, regression, ranking and
 * ordinal metrics are.
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { Metric } from 'aifn-compute/learning/metrics'
import * as text from './text'
import * as detection from './detection'
import * as quality from './quality'
import * as generative from './generative'
import * as fairness from './fairness'
import * as beyondAccuracy from './beyond-accuracy'

/**
 * Every application metric of the family's six modules, keyed by its `info.key` (the counterpart of
 * `aifn-compute/learning/metrics`' `metricRegistry`).
 */
export const evaluationMetricRegistry: Readonly<Record<string, Metric>> = entries(
  'metric',
  text,
  detection,
  quality,
  generative,
  fairness,
  beyondAccuracy,
) as unknown as Readonly<Record<string, Metric>>
