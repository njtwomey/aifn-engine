/**
 * Ordinal likelihoods: the probability of each of K ordered classes given a latent linear predictor η and K − 1
 * increasing thresholds θ₁ < … < θ_{K−1}, for the three classical models (Agresti, 2010, "Analysis of Ordinal
 * Categorical Data", 2nd ed., ch. 3–4):
 *
 * - `cumulative` (McCullagh, 1980, "Regression models for ordinal data", JRSS B 42): P(y ≤ k) = F(θ_k − η), so
 *   P(y = k) = F(θ_k − η) − F(θ_{k−1} − η). With the logit link this is the proportional-odds model.
 * - `continuation-ratio` (sequential; Fienberg, 1980; Tutz, 1990): P(y = k | y ≥ k) = F(θ_k − η), so
 *   P(y = k) = F(θ_k − η) Π_{j<k} (1 − F(θ_j − η)).
 * - `adjacent-category` (Agresti, 2010, §4.1): log(P(y = k + 1)/P(y = k)) = η − θ_k, logit only.
 *
 * Classes are 0 … K − 1. F is the latent cdf of the link: logit (logistic), probit (normal) or cloglog (Gumbel minimum,
 * F(z) = 1 − exp(−e^z)). Everything is computed in log space from primitives (log F and log(1 − F) in closed form per
 * link), so log-likelihoods are differentiable in η and θ. η is a number or a tensor of any batch shape; class
 * log-probabilities have shape [...batch, K]. Thresholds must be increasing (build them with `orderedBijector` from
 * `aifn-compute/probability/bijectors` to optimise them unconstrained).
 */

import { logDiffExp, logSigmoid, logSoftmax, normalLogCdf, sigmoid, normalCdf } from 'aifn-compute/numerics/special'
import {
  add,
  concat,
  exp,
  fromData,
  isTensor,
  full,
  gather,
  matmul,
  neg,
  reshape,
  shapeOfValue,
  slice,
  sub,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Index, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The three ordinal models. */
export type OrdinalModel = 'cumulative' | 'continuation-ratio' | 'adjacent-category'

/** The latent cdf of an ordinal model. */
export type OrdinalLinkName = 'logit' | 'probit' | 'cloglog'

/** A latent cdf F with log F and log(1 − F) written stably with primitives. */
type LatentCdf = {
  cdf(z: Value): Value
  logCdf(z: Value): Value
  logSurvival(z: Value): Value
}

const LATENT: Record<OrdinalLinkName, LatentCdf> = {
  logit: { cdf: sigmoid, logCdf: logSigmoid, logSurvival: (z) => logSigmoid(neg(z)) },
  probit: { cdf: normalCdf, logCdf: normalLogCdf, logSurvival: (z) => normalLogCdf(neg(z)) },
  // F = 1 − exp(−e^z): log(1 − F) = −e^z, and log F = log(1 − exp(−e^z)) = logDiffExp(0, −e^z).
  cloglog: {
    cdf: (z) => sub(1, exp(neg(exp(z)))),
    logCdf: (z) => logDiffExp(0, neg(exp(z))),
    logSurvival: (z) => neg(exp(z)),
  },
}

/** An ordinal likelihood: class probabilities and log-likelihoods as functions of η and the thresholds θ. */
export interface OrdinalLikelihood {
  readonly model: OrdinalModel
  readonly link: OrdinalLinkName
  /** log P(y = k | η, θ) for k = 0 … K − 1 (K = len θ + 1): shape [...shape(η), K]. */
  logProbabilities(eta: Value, thresholds: Value): Value
  /** P(y = k | η, θ), shape [...shape(η), K]; each lane sums to 1. */
  probabilities(eta: Value, thresholds: Value): Value
  /** The pointwise log-likelihood log P(y | η, θ) of observed classes y (shape of η; a number for one observation). */
  logLik(y: Index | Tensor | ArrayLike<Index>, eta: Value, thresholds: Value): Value
}

/** The number of thresholds, checking that θ is a vector. */
function thresholdCount(thresholds: Value): Size {
  const shape = shapeOfValue(thresholds)
  if (shape.length !== 1 || shape[0] < 1) {
    throw new ShapeError('ordinal', `ordinal: thresholds must be a non-empty vector, got shape [${shape.join(', ')}]`)
  }
  return shape[0]
}

/** η with a trailing axis of length 1 ([...batch] → [...batch, 1]); a scalar η stays a scalar. */
function asColumn(eta: Value): Value {
  const batch = shapeOfValue(eta)
  return batch.length === 0 ? eta : reshape(eta, [...batch, 1])
}

/** A constant tensor of the batch shape of η with a trailing axis of length 1. */
function padding(eta: Value, value: number): Tensor {
  return full([...shapeOfValue(eta), 1], value)
}

/** The strictly upper-triangular (m × n) matrix Uⱼₖ = 1 for j < k: v · U is the exclusive cumulative sum of v. */
function exclusiveCumsum(m: Size, n: Size): Tensor {
  const u = new Float64Array(m * n)
  for (let j = 0; j < m; j++) for (let k = j + 1; k < n; k++) u[j * n + k] = 1
  return fromData(u, [m, n])
}

/**
 * The ordinal likelihood of a model over a latent cdf (default `cumulative` with `logit`, the proportional-odds model).
 * `adjacent-category` is defined for the logit only.
 *
 * @example
 * const po = ordinalLikelihood('cumulative', 'logit')
 * po.probabilities(0.5, tensor([-1, 0, 1])) // P(y = 0 … 3) at η = 0.5
 */
export function ordinalLikelihood(
  model: OrdinalModel = 'cumulative',
  linkName: OrdinalLinkName = 'logit',
): OrdinalLikelihood {
  const F = LATENT[linkName]
  if (!F) throw new DomainError('ordinalLikelihood', `ordinalLikelihood: unknown link "${linkName as string}"`)
  if (model === 'adjacent-category' && linkName !== 'logit') {
    throw new DomainError(
      'ordinalLikelihood',
      'ordinalLikelihood: the adjacent-category model is defined for the logit link only',
    )
  }
  const logProbabilities = (eta: Value, thresholds: Value): Value => {
    const m = thresholdCount(thresholds)
    const k = m + 1
    const column = asColumn(eta)
    const z = sub(thresholds, column) // θ_j − η, shape [...batch, m]
    switch (model) {
      case 'cumulative': {
        // log F padded with log F(−∞) = −∞ and log F(∞) = 0 as constants (never differentiated through infinities).
        const logF = concat([padding(eta, -Infinity), F.logCdf(z), padding(eta, 0)], -1)
        const upper = sliceLast(logF, 1, k + 1)
        const lower = sliceLast(logF, 0, k)
        return logDiffExp(upper, lower)
      }
      case 'continuation-ratio': {
        // log P(y = k) = log F_k + Σ_{j<k} log(1 − F_j), with log F_{K−1} = 0 for the last class.
        const stop = concat([F.logCdf(z), padding(eta, 0)], -1)
        return add(stop, matmul(F.logSurvival(z), exclusiveCumsum(m, k)))
      }
      case 'adjacent-category': {
        // Unnormalised log P(y = k) = Σ_{j<k} (η − θ_j) = −Σ_{j<k} z_j; normalised by a log-softmax over classes.
        return logSoftmax(neg(matmul(z, exclusiveCumsum(m, k))), { axis: -1 })
      }
    }
    throw new DomainError('ordinalLikelihood', `ordinalLikelihood: unknown model "${model as string}"`)
  }
  return {
    model,
    link: linkName,
    logProbabilities,
    probabilities: (eta, thresholds) => exp(logProbabilities(eta, thresholds)),
    logLik: (y, eta, thresholds) => {
      const logP = logProbabilities(eta, thresholds)
      const shape = shapeOfValue(logP)
      const k = shape[shape.length - 1]
      const classes = typeof y === 'number' ? [y] : isTensor(y) ? Array.from(toFlat(y)) : Array.from(y)
      const batch = shape.slice(0, -1)
      const count = batch.reduce((a, b) => a * b, 1)
      if (classes.length !== count) {
        throw new ShapeError('ordinal logLik', `ordinal logLik: ${classes.length} classes for a batch of ${count}`)
      }
      const indices = Int32Array.from(classes, (c, i) => {
        if (!Number.isInteger(c) || c < 0 || c >= k)
          throw new DomainError('ordinal logLik', `ordinal logLik: class ${c} outside 0 … ${k - 1}`)
        return i * k + c
      })
      const picked = gather(logP, indices, batch)
      return batch.length === 0 ? unwrapScalar(picked) : picked
    },
  }
}

/** A rank-0 result as a number when untraced (a traced value is returned as is). */
function unwrapScalar(v: Value): Value {
  const raw = unwrap(v)
  return raw === v && typeof raw !== 'number' ? toFlat(raw)[0] : v
}

/** The slice [start, stop) of the last axis of v. */
function sliceLast(v: Value, start: number, stop: number): Value {
  const rank = shapeOfValue(v).length
  return slice(v, ...Array.from({ length: rank - 1 }, () => null), [start, stop])
}
