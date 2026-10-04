/**
 * Drawn results of metrics (design S §2.12): curves with fixed axes. Every curve producer in `aifn-compute/learning/metrics`
 * (ROC, precision–recall, DET, gain, cost, precision–recall–gain, reliability) returns a `Curve`, narrowed by `curve`
 * and extended with the few numbers its chart needs.
 */

import type { Kinded } from './kinds'
import type { Tensor } from './numbers'

/** The curves a metric can draw; each has fixed axes and ranges, declared once next to the type. */
export type CurveName = 'roc' | 'pr' | 'det' | 'gain' | 'cost' | 'prg' | 'reliability' | 'calibration'

/**
 * A drawn metric curve: points (x, y) in threshold order, with the summary area where one is defined. The axes per
 * `curve`: roc (FPR, TPR), pr (recall, precision), det (FPR, FNR), gain (fraction targeted, fraction of positives),
 * cost (probability cost, normalised expected cost), prg (recall gain, precision gain), reliability (mean predicted
 * probability, observed frequency).
 */
export interface Curve<C extends CurveName = CurveName> extends Kinded<'curve'> {
  readonly curve: C
  readonly x: Tensor
  readonly y: Tensor
  /** Thresholds in decreasing order, aligned with the points (absent for binned curves). */
  readonly thresholds?: Tensor
  /** The summary number (AUROC, average precision, …). */
  readonly area?: number
  /** The fraction of positives: the chance level of a precision–recall curve. */
  readonly prevalence?: number
}
