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
  /** Samples drawn. */
  t: Size
  /** The sample drawn at this step, its model (null if degenerate) and that model's inlier count. */
  sample: number[]
  candidate: M | null
  candidateInliers: Size
  /** The best model so far, its inliers (0/1 per datum) and their count. */
  best: M | null
  inliers: number[]
  inlierCount: Size
  /** The adaptive number of samples needed: ⌈log(1 − p)/log(1 − wˢ)⌉ with w the best inlier fraction so far. */
  required: Scalar
  /** Set once t ≥ required. */
  terminated: boolean
}

/** The adaptive sample count for inlier fraction w, sample size s and confidence p. */
export function ransacTrials(w: Scalar, s: Size, p: Scalar): Scalar {
  if (w >= 1) return 1
  if (w <= 0) return Infinity
  // log1p keeps log(1 − wˢ) accurate when wˢ is tiny (many outliers, large samples), where 1 − wˢ rounds to 1.
  const den = Math.log1p(-(w ** s))
  return den === 0 ? Infinity : Math.ceil(Math.log1p(-p) / den)
}

/**
 * RANSAC as a traceable algorithm (Fischler & Bolles, 1981). Each step draws s distinct indices from the step's
 * stream, fits a candidate, and counts the data with residual ≤ threshold; the best candidate is kept. The run
 * terminates when the number of samples reaches the adaptive count ⌈log(1 − p)/log(1 − wˢ)⌉, w being the best inlier
 * fraction so far (Hartley & Zisserman, 2004, §4.7.1).
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
  model: M | null
  inliers: number[]
  inlierCount: Size
  samples: Size
}

/**
 * Run `ransac` to termination (at most `maxSamples`, default 1000), then refit the model on the consensus set with
 * `problem.refit` (or `fit`) and recount its inliers.
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
