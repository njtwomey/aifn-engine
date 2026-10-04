/**
 * Random sample consensus (RANSAC; Fischler & Bolles, 1981, "Random sample consensus: a paradigm for model fitting
 * with applications to image analysis and automated cartography", CACM 24(6)), generic over the model: draw a minimal
 * sample, fit a model to it, count the data within a residual threshold, keep the model with the most inliers, and
 * stop once enough samples have been drawn that an all-inlier one was drawn with high confidence. The model and its
 * residuals are callbacks, so the same loop fits lines, homographies and fundamental matrices.
 */

import { child, choice } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { Scalar, Size, Status, Stream } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A robust fitting problem over n data: how to fit a model to some of them and how far each datum is from a model. */
export interface RansacProblem<M> {
  /** Number of data n. */
  readonly count: Size
  /** The minimal sample size s (2 for a line, 4 for a homography, 8 for the eight-point algorithm). */
  readonly sampleSize: Size
  /** A model through the data at these indices, or null for a degenerate sample. */
  fit(indices: readonly number[]): M | null
  /** The residual of every datum under a model (n values, compared with the threshold). */
  residuals(model: M): ArrayLike<number>
  /** Optional: refit on all inliers once the search ends (default `fit`). */
  refit?(indices: readonly number[]): M | null
}

/** Options for `ransac`. */
export interface RansacOptions {
  /** A datum is an inlier when its residual is at most this. */
  threshold: Scalar
  /** The probability of having drawn at least one all-inlier sample at termination. Default 0.99. */
  confidence?: Scalar
}

/** The state of `ransac`. */
export interface RansacState<M> extends Status {
  /** Samples drawn so far. */
  t: Size
  /** The sample of indices drawn at this step. */
  sample: number[]
  /** The candidate model fitted to `sample`, or null if degenerate. */
  candidate: M | null
  /** The number of inliers for `candidate`. */
  candidateInliers: Size
  /** The best model found across all steps so far, or null if none succeeded. */
  best: M | null
  /** Binary mask where entry $i$ is 1 if datum $i$ is an inlier to `best`, 0 otherwise. */
  inliers: number[]
  /** Total number of inliers for `best`. */
  inlierCount: Size
  /**
   * The adaptive number of samples needed: $\lceil\log(1 - p)/\log(1 - w^s)\rceil$ with $w$ the best inlier fraction
   * so far.
   */
  required: Scalar
  /** Set once $t \ge \text{required}$. */
  terminated: boolean
}

/**
 * The adaptive sample count $N = \left\lceil \frac{\log(1 - p)}{\log(1 - w^s)} \right\rceil$ (Fischler & Bolles, 1981;
 * Hartley & Zisserman, 2004, §4.7.1) needed to draw at least one all-inlier sample of size $s$ with probability $p$,
 * given inlier fraction $w$.
 *
 * @param w The fraction of inliers in the dataset, $w \in [0, 1]$.
 * @param s The sample size: number of data points needed to fit a minimal model (e.g. 2 for a line).
 * @param p The desired probability of having drawn at least one outlier-free sample (e.g. 0.99).
 * @returns The integer number of trials $\lceil N \rceil$, or $\infty$ when $w \le 0$.
 *
 * @example Sample count for a line model
 * print('trials =', ransacTrials(0.6, 2, 0.99))
 */
export function ransacTrials(w: Scalar, s: Size, p: Scalar): Scalar {
  if (w >= 1) return 1
  if (w <= 0) return Infinity
  // log1p keeps log(1 − wˢ) accurate when wˢ is tiny (many outliers, large samples), where 1 − wˢ rounds to 1.
  const den = Math.log1p(-(w ** s))
  return den === 0 ? Infinity : Math.ceil(Math.log1p(-p) / den)
}

/**
 * RANSAC as a traceable algorithm (Fischler & Bolles, 1981). Each step draws $s$ distinct indices from the step's
 * stream, fits a candidate, and counts the data with residual $\le \text{threshold}$; the best candidate is kept. The
 * run terminates when the number of samples reaches the adaptive count $\lceil\log(1 - p)/\log(1 - w^s)\rceil$, $w$
 * being the best inlier fraction so far (Hartley & Zisserman, 2004, §4.7.1).
 *
 * @param problem The model fitting problem: count of data, minimal sample size, and callbacks to fit and score.
 * @param options Configuration: error `threshold` for inliers and optional `confidence` (default 0.99).
 * @returns A step-through `Algorithm` whose state tracks the candidate, current best model, and inlier mask.
 *
 * @example Step through RANSAC fitting a 1D centre
 * const pts = [1.0, 1.0, 1.1, 0.9, 10.0]
 * const problem = {
 *   count: 5,
 *   sampleSize: 1,
 *   fit: ([i]) => pts[i],
 *   residuals: (m) => pts.map((p) => Math.abs(p - m)),
 * }
 * const state = run(ransac(problem, { threshold: 0.2 }), undefined, 10)
 * print('inliers =', state.inlierCount)
 */
export function ransac<M>(problem: RansacProblem<M>, options: RansacOptions): Algorithm<undefined, RansacState<M>> {
  const { count: n, sampleSize: s } = problem
  if (!(s >= 1 && s <= n)) throw new DomainError('ransac', `ransac: need ${s} ≤ ${n} data`)
  const p = options.confidence ?? 0.99
  const score = (model: M | null): { mask: number[]; count: number } => {
    if (!model) return { mask: new Array<number>(n).fill(0), count: 0 }
    const r = problem.residuals(model)
    const mask = Array.from({ length: n }, (_, i): number => (r[i] <= options.threshold ? 1 : 0))
    return { mask, count: mask.reduce((a, b) => a + b, 0) }
  }
  const draw = (stream: Stream) => toFlat(choice(child(stream, 'sample'), n, s, { replace: false }))
  const advance = (prev: RansacState<M> | null, t: number, stream: Stream): RansacState<M> => {
    const sample = draw(stream)
    const candidate = problem.fit(sample)
    const sc = score(candidate)
    const better = !prev || sc.count > prev.inlierCount
    const best = better ? candidate : prev.best
    const inliers = better ? sc.mask : prev.inliers
    const inlierCount = better ? sc.count : prev.inlierCount
    const required = ransacTrials(inlierCount / n, s, p)
    return {
      t,
      sample,
      candidate,
      candidateInliers: sc.count,
      best,
      inliers,
      inlierCount,
      required,
      terminated: t + 1 >= required,
    }
  }
  return {
    name: 'ransac',
    init: (_start, stream) => advance(null, 0, stream),
    step: (state, ctx) => advance(state, state.t + 1, ctx.stream),
  }
}

/** A finished RANSAC fit: the model refitted on the inliers, the inlier mask and the number of samples drawn. */
export interface RansacFit<M> {
  /** The model refitted on the consensus set (or best candidate if refit failed or none found). */
  model: M | null
  /** Binary mask where entry $i$ is 1 if datum $i$ is an inlier, 0 if an outlier. */
  inliers: number[]
  /** Number of inliers found. */
  inlierCount: Size
  /** Total number of samples drawn. */
  samples: Size
}

/**
 * Run `ransac` to termination (at most `maxSamples`, default 1000), then refit the model on the consensus set with
 * `problem.refit` (or `fit`) and recount its inliers.
 *
 * @param problem The model fitting problem: count of data, minimal sample size, and callbacks to fit and score.
 * @param options Configuration: error `threshold`, optional `confidence`, `maxSamples` budget, and PRNG `stream`.
 * @returns The final `RansacFit` containing the refitted model, inlier mask, count, and sample count.
 *
 * @example Fit a 1D centre robust to outliers
 * const pts = [1.0, 1.0, 1.1, 0.9, 10.0]
 * const problem = {
 *   count: 5,
 *   sampleSize: 1,
 *   fit: ([i]) => pts[i],
 *   residuals: (m) => pts.map((p) => Math.abs(p - m)),
 * }
 * const fit = ransacFit(problem, { threshold: 0.2 })
 * print('fitted centre =', fit.model)
 * print('inliers =', fit.inlierCount)
 */
export function ransacFit<M>(
  problem: RansacProblem<M>,
  options: RansacOptions & { maxSamples?: Size; stream?: Stream },
): RansacFit<M> {
  const s = run(ransac(problem, options), undefined, (options.maxSamples ?? 1000) - 1, { stream: options.stream })
  const idx = s.inliers.flatMap((v, i) => (v ? [i] : []))
  const refit = idx.length >= problem.sampleSize ? (problem.refit ?? problem.fit)(idx) : null
  if (!refit) return { model: s.best, inliers: s.inliers, inlierCount: s.inlierCount, samples: s.t + 1 }
  const r = problem.residuals(refit)
  const inliers = Array.from({ length: problem.count }, (_, i): number => (r[i] <= options.threshold ? 1 : 0))
  return { model: refit, inliers, inlierCount: inliers.reduce((a, b) => a + b, 0), samples: s.t + 1 }
}
