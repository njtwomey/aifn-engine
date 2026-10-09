/**
 * Ordinal likelihoods: the probability of each of $K$ ordered classes given a latent linear predictor $\eta$ and
 * $K - 1$ increasing thresholds, for the three classical models (Agresti, 2010, "Analysis of Ordinal Categorical
 * Data", 2nd ed., ch. 3–4).
 *
 * Classes are $0, \dots, K - 1$ and thresholds $\theta_0 < \dots < \theta_{K-2}$, with $\theta_{-1} = -\infty$ and
 * $\theta_{K-1} = \infty$:
 *
 * - `cumulative` (McCullagh, 1980, "Regression models for ordinal data", JRSS B 42):
 *   $\pr(y \le k) = F(\theta_k - \eta)$, so $\pr(y = k) = F(\theta_k - \eta) - F(\theta_{k-1} - \eta)$. With the
 *   logit link this is the proportional-odds model.
 * - `continuation-ratio` (sequential; Fienberg, 1980; Tutz, 1990): $\pr(y = k \mid y \ge k) = F(\theta_k - \eta)$,
 *   so $\pr(y = k) = F(\theta_k - \eta) \prod_{j<k} (1 - F(\theta_j - \eta))$.
 * - `adjacent-category` (Agresti, 2010, §4.1): $\log(\pr(y = k + 1)/\pr(y = k)) = \eta - \theta_k$, logit only.
 *
 * $F$ is the latent cdf of the link: logit (logistic), probit (normal) or cloglog (Gumbel minimum,
 * $F(z) = 1 - \exp(-e^z)$). Everything is computed in log space from primitives ($\log F$ and $\log(1 - F)$ in
 * closed form per link), so log-likelihoods are differentiable in $\eta$ and $\thetavec$. $\eta$ is a number or a
 * tensor of any batch shape; class log-probabilities have shape $[\dots, K]$, the batch shape of $\eta$ followed by
 * the classes. Thresholds must be increasing (build them with `orderedBijector` from
 * `aifn-compute/probability/bijectors` to optimise them unconstrained); this is not checked.
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

/**
 * A latent cdf $F$ with $\log F$ and $\log(1 - F)$ written stably with primitives: `cdf` is $F(z)$, `logCdf` is
 * $\log F(z)$ and `logSurvival` is $\log(1 - F(z))$.
 */
type LatentCdf = {
  cdf(z: Value): Value
  logCdf(z: Value): Value
  logSurvival(z: Value): Value
}

/** The latent cdf of each link. */
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

/**
 * An ordinal likelihood: class probabilities and log-likelihoods as functions of $\eta$ and the thresholds
 * $\thetavec$ (a vector of $K - 1$ values; `ShapeError` otherwise).
 */
export interface OrdinalLikelihood {
  /** The model. */
  readonly model: OrdinalModel
  /** The latent cdf's link. */
  readonly link: OrdinalLinkName
  /**
   * $\log \pr(y = k \mid \eta, \thetavec)$ for $k = 0, \dots, K - 1$, with $K$ one more than the number of
   * thresholds: the batch shape of $\eta$ followed by $K$.
   */
  logProbabilities(eta: Value, thresholds: Value): Value
  /** $\pr(y = k \mid \eta, \thetavec)$, shaped as `logProbabilities`; each lane sums to 1. */
  probabilities(eta: Value, thresholds: Value): Value
  /**
   * The pointwise log-likelihood $\log \pr(y \mid \eta, \thetavec)$ of observed classes $y$, one per element of
   * $\eta$ (the shape of $\eta$; a number for one untraced observation). Throws `ShapeError` when the count of
   * classes differs from the size of $\eta$, and `DomainError` for a class outside $0, \dots, K - 1$.
   */
  logLik(y: Index | Tensor | ArrayLike<Index>, eta: Value, thresholds: Value): Value
}

/**
 * The number of thresholds, checking that $\thetavec$ is a non-empty vector (`ShapeError` otherwise).
 *
 * @param thresholds The thresholds $\thetavec$.
 * @returns Their number, $K - 1$.
 */
function thresholdCount(thresholds: Value): Size {
  const shape = shapeOfValue(thresholds)
  if (shape.length !== 1 || shape[0] < 1) {
    throw new ShapeError('ordinal', `ordinal: thresholds must be a non-empty vector, got shape [${shape.join(', ')}]`)
  }
  return shape[0]
}

/**
 * $\eta$ with a trailing axis of length 1, so that it broadcasts against the thresholds; a scalar $\eta$ stays a
 * scalar.
 *
 * @param eta The linear predictor $\eta$.
 * @returns $\eta$ reshaped to its batch shape followed by 1.
 */
function asColumn(eta: Value): Value {
  const batch = shapeOfValue(eta)
  return batch.length === 0 ? eta : reshape(eta, [...batch, 1])
}

/**
 * A constant tensor of the batch shape of $\eta$ with a trailing axis of length 1, to pad the class axis.
 *
 * @param eta The linear predictor, whose shape is taken.
 * @param value The constant (here $-\infty$ or 0, a log-probability).
 * @returns The constant tensor.
 */
function padding(eta: Value, value: number): Tensor {
  return full([...shapeOfValue(eta), 1], value)
}

/**
 * The strictly upper-triangular $m \times n$ matrix with $U_{jk} = 1$ for $j < k$: $\vvec^\top\Umat$ is the
 * exclusive cumulative sum of $\vvec$.
 *
 * @param m The number of rows (the length of $\vvec$).
 * @param n The number of columns (the length of the sums).
 * @returns $\Umat$ as a float64 tensor.
 */
function exclusiveCumsum(m: Size, n: Size): Tensor {
  const u = new Float64Array(m * n)
  for (let j = 0; j < m; j++) for (let k = j + 1; k < n; k++) u[j * n + k] = 1
  return fromData(u, [m, n])
}

/**
 * The ordinal likelihood of a model over a latent cdf (default `cumulative` with `logit`, the proportional-odds model).
 * `adjacent-category` is defined for the logit only; it and an unknown link throw `DomainError`.
 *
 * @param model The model: `cumulative`, `continuation-ratio` or `adjacent-category`.
 * @param linkName The latent cdf: `logit`, `probit` or `cloglog`.
 * @returns The likelihood, as functions of $\eta$ and the thresholds.
 *
 * @example The proportional-odds model at $\eta = 0.5$ with four classes
 * const po = ordinalLikelihood('cumulative', 'logit')
 * const p = po.probabilities(0.5, tensor([-1, 0, 1]))
 * print('P(y = 0 ... 3):', p, 'sum:', sum(p))
 * print('log P(y = 2):', po.logLik(2, 0.5, tensor([-1, 0, 1])))
 *
 * @example The three models on the same predictor and thresholds
 * const theta = tensor([-1, 0, 1])
 * print('cumulative:', ordinalLikelihood('cumulative').probabilities(0.5, theta))
 * print('continuation-ratio:', ordinalLikelihood('continuation-ratio').probabilities(0.5, theta))
 * print('adjacent-category:', ordinalLikelihood('adjacent-category').probabilities(0.5, theta))
 *
 * @example A batch of predictors, and the gradient in $\eta$
 * const po = ordinalLikelihood('cumulative', 'probit')
 * print('log-likelihoods:', po.logLik([0, 2], tensor([-2, 3]), tensor([-1, 1])))
 * print('d/d eta:', grad((eta) => po.logLik(0, eta, tensor([-1, 1])))(0))
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

/**
 * A rank-0 result as a number when untraced (a traced value is returned as is).
 *
 * @param v The rank-0 value.
 * @returns A number, or `v` itself when traced.
 */
function unwrapScalar(v: Value): Value {
  const raw = unwrap(v)
  return raw === v && typeof raw !== 'number' ? toFlat(raw)[0] : v
}

/**
 * The slice $[\text{start}, \text{stop})$ of the last axis of `v`, every other axis kept whole.
 *
 * @param v The value to slice.
 * @param start The first index kept.
 * @param stop One past the last index kept.
 * @returns The slice.
 */
function sliceLast(v: Value, start: number, stop: number): Value {
  const rank = shapeOfValue(v).length
  return slice(v, ...Array.from({ length: rank - 1 }, () => null), [start, stop])
}
