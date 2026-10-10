/**
 * Generative classifiers: naive Bayes (Gaussian, multinomial, Bernoulli) and linear and quadratic discriminant
 * analysis. Each fits $p(\xvec \mid y = k)$ and $p(y = k)$ and classifies by Bayes' rule; `forward` and `score`
 * return the joint log-likelihoods $\log p(\xvec, y = k)$ (up to a constant shared by the classes) as an
 * $m \times K$ matrix, `predictive` their softmax as a Bernoulli law of class 1 ($K = 2$) or a Categorical law, and
 * `decide` the most probable class. Labels are the integers $0, \dots, K - 1$, and the class priors are the training
 * frequencies unless `priors` are given.
 *
 * References: Hastie, Tibshirani and Friedman (2009), "The Elements of Statistical Learning", §4.3 (LDA, QDA) and
 * §6.6.3 (naive Bayes); McCallum and Nigam (1998), "A comparison of event models for naive Bayes text
 * classification" (multinomial and Bernoulli models). Conventions match scikit-learn's `GaussianNB`, `MultinomialNB`,
 * `BernoulliNB`, `LinearDiscriminantAnalysis` (the default `svd` solver) and `QuadraticDiscriminantAnalysis`.
 */

import type {
  AnyUnivariate,
  Decides,
  Estimator,
  Fitted,
  Predicts,
  Scores,
  Supervised,
} from 'aifn-compute/learning/estimators'
import { cholesky, eigh, solveTriangular } from 'aifn-compute/numerics/linalg'
import { dense, fromData, square, sum, type Tensor } from 'aifn-compute/foundation/tensor'
import { classLabels, inputs, mat, matrix, probabilityModel, softmaxRows, vec } from '../util'
import { defineModel } from 'aifn-compute/learning/estimators'
import { real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A fitted generative classifier: joint log-likelihoods as its head, a class law as its predictive. */
export interface GenerativeClassifier
  extends Fitted<Tensor, Tensor>, Scores<Tensor>, Decides<Tensor, Tensor>, Predicts<Tensor, AnyUnivariate> {
  /** The number of classes $K$. */
  readonly classes: number
  /** The log-priors $\log p(y = k)$, $K$ values. */
  readonly logPrior: Tensor
}

/**
 * Class counts and log-priors (empirical frequencies unless `priors` are given). Throws `ShapeError` when `priors`
 * does not have $K$ entries.
 *
 * @param y The integer label of each training row.
 * @param K The number of classes.
 * @param priors Prior class weights, one per class, normalised to sum to 1 here; undefined for the training
 *   frequencies.
 * @param where The caller's name, for error messages.
 * @returns `counts`, the number of rows of each class, and `logPrior`, the $K$ log-priors.
 */
function priorsOf(y: Int32Array, K: number, priors: readonly number[] | undefined, where: string) {
  const counts = new Float64Array(K)
  for (const c of y) counts[c]++
  const logPrior = new Float64Array(K)
  if (priors) {
    if (priors.length !== K) throw new ShapeError(where, `${where}: ${K} classes but ${priors.length} priors`)
    const total = priors.reduce((a, b) => a + b, 0)
    for (let c = 0; c < K; c++) logPrior[c] = Math.log(priors[c] / total)
  } else {
    for (let c = 0; c < K; c++) logPrior[c] = Math.log(counts[c] / y.length)
  }
  return { counts, logPrior }
}

/**
 * Per-class means, $K \times d$ row-major (zero for a class with no rows).
 *
 * @param v The training matrix, $n \times d$ row-major.
 * @param y The integer label of each row.
 * @param n The number of rows.
 * @param d The number of features.
 * @param K The number of classes.
 * @param counts The number of rows of each class, as `priorsOf` returns them.
 * @returns The class means, row $k$ the mean of class $k$.
 */
function classMeans(v: Float64Array, y: Int32Array, n: number, d: number, K: number, counts: Float64Array) {
  const means = new Float64Array(K * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) means[y[i] * d + j] += v[i * d + j]
  for (let c = 0; c < K; c++) for (let j = 0; j < d; j++) means[c * d + j] /= counts[c] || 1
  return means
}

// ── Naive Bayes ──────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted Gaussian naive Bayes model. */
export interface GaussianNaiveBayesModel extends GenerativeClassifier {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gaussian-naive-bayes'
  /** Per-class feature means, $K \times d$. */
  readonly means: Tensor
  /** Per-class feature variances, $K \times d$ (maximum likelihood, plus the smoothing `epsilon`). */
  readonly variances: Tensor
  /** The variance added to every entry: `varianceSmoothing` times the largest feature variance over all the data. */
  readonly epsilon: number
}

/**
 * Gaussian naive Bayes, as scikit-learn's `GaussianNB`: features independent given the class, each Gaussian with a
 * per-class mean and maximum-likelihood variance, so
 * $\log p(\xvec, y = k) = \log \pi_k + \sum_j \log \Gauss(x_j; \mu_{kj}, \sigma_{kj}^2)$.
 *
 * @param params `varianceSmoothing` (default 1e-9): the fraction of the largest feature variance added to every
 *   variance, for stability, as scikit-learn's `var_smoothing`. `priors`: the class priors, one weight per class
 *   (normalised; default the training frequencies).
 * @returns The estimator: `fit({ x, y })` returns a `GaussianNaiveBayesModel`.
 *
 * @example Two Gaussian classes
 * // Two Gaussian classes of twenty points, centred at (0, 0) and (3, 3).
 * const x = concat([normals(stream(0), [20, 2]), add(normals(stream(1), [20, 2]), tensor([3, 3]))])
 * const y = tensor(Array.from({ length: 40 }, (_, i) => (i < 20 ? 0 : 1)))
 * const model = gaussianNaiveBayes().fit({ x, y })
 * print('means =', model.means)
 * print('variances =', model.variances)
 * print('P(class 1) =', model.predictive(tensor([[0, 0], [1.5, 1.5], [3, 3]])).mean())
 */
export function gaussianNaiveBayes(
  params: { varianceSmoothing?: number; priors?: readonly number[] } = {},
): Estimator<Supervised<Tensor, Tensor>, GaussianNaiveBayesModel> {
  const { varianceSmoothing = 1e-9, priors } = params
  return {
    name: 'gaussian-naive-bayes',
    params: { varianceSmoothing, priors },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'gaussianNaiveBayes')
      const { y: labels, k: K } = classLabels(y, n, 'gaussianNaiveBayes')
      const { counts, logPrior } = priorsOf(labels, K, priors, 'gaussianNaiveBayes')
      const means = classMeans(v, labels, n, d, K, counts)
      // ε = smoothing × the largest variance of any feature over all the data.
      let largest = 0
      for (let j = 0; j < d; j++) {
        let m = 0
        for (let i = 0; i < n; i++) m += v[i * d + j]
        m /= n
        let s = 0
        for (let i = 0; i < n; i++) s += (v[i * d + j] - m) ** 2
        largest = Math.max(largest, s / n)
      }
      const epsilon = varianceSmoothing * largest
      const variances = new Float64Array(K * d)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) variances[labels[i] * d + j] += (v[i * d + j] - means[labels[i] * d + j]) ** 2
      }
      for (let c = 0; c < K; c++)
        for (let j = 0; j < d; j++) variances[c * d + j] = variances[c * d + j] / (counts[c] || 1) + epsilon
      const head = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'gaussianNaiveBayes')
        const out = new Float64Array(m * K)
        for (let i = 0; i < m; i++) {
          for (let c = 0; c < K; c++) {
            let s = logPrior[c]
            for (let j = 0; j < d; j++) {
              const s2 = variances[c * d + j]
              s -= 0.5 * Math.log(2 * Math.PI * s2) + (qv[i * d + j] - means[c * d + j]) ** 2 / (2 * s2)
            }
            out[i * K + c] = s
          }
        }
        return out
      }
      return {
        kind: 'model',
        name: 'gaussian-naive-bayes',
        classes: K,
        logPrior: vec(logPrior),
        means: mat(means, K, d),
        variances: mat(variances, K, d),
        epsilon,
        ...probabilityModel(head, (h, m) => softmaxRows(h, m, K), K),
      }
    },
  }
}

/** A fitted multinomial or Bernoulli naive Bayes model. */
export interface DiscreteNaiveBayesModel extends GenerativeClassifier {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'multinomial-naive-bayes' | 'bernoulli-naive-bayes'
  /**
   * $\log \theta_{kj}$, $K \times d$: the log-probability of feature $j$ in class $k$, of one count unit
   * (multinomial) or of a one (Bernoulli).
   */
  readonly featureLogProb: Tensor
  /** The smoothed per-class feature totals $N_{kj} + \alpha$, $K \times d$. */
  readonly featureCounts: Tensor
}

/**
 * Multinomial naive Bayes for count features (such as word counts), as scikit-learn's `MultinomialNB`:
 * $\log p(\xvec \mid k) = \sum_j x_j \log \theta_{kj} + \text{const}$, with
 * $\theta_{kj} = (N_{kj} + \alpha) / \sum_j (N_{kj} + \alpha)$, $N_{kj}$ the total count of feature $j$ in class $k$.
 * `fit` throws `DomainError` for a negative or NaN training feature; query features are not checked.
 *
 * @param params `alpha`: the additive (Laplace) smoothing $\alpha$ (default 1). `priors`: the class priors, one
 *   weight per class (normalised; default the training frequencies).
 * @returns The estimator: `fit({ x, y })` returns a `DiscreteNaiveBayesModel`.
 *
 * @example Word counts of sport and politics
 * // Counts of the words (ball, goal, vote, poll) in four short texts: sport (0) or politics (1).
 * const x = tensor([[3, 2, 0, 0], [2, 3, 1, 0], [0, 0, 3, 2], [1, 0, 2, 3]])
 * const model = multinomialNaiveBayes().fit({ x, y: tensor([0, 0, 1, 1]) })
 * print('word probabilities =', exp(model.featureLogProb))
 * print('classes =', model.decide(tensor([[2, 1, 0, 0], [0, 1, 2, 2]])))
 */
export function multinomialNaiveBayes(
  params: { alpha?: number; priors?: readonly number[] } = {},
): Estimator<Supervised<Tensor, Tensor>, DiscreteNaiveBayesModel> {
  const { alpha = 1, priors } = params
  return {
    name: 'multinomial-naive-bayes',
    params: { alpha, priors },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'multinomialNaiveBayes')
      const { y: labels, k: K } = classLabels(y, n, 'multinomialNaiveBayes')
      const { logPrior } = priorsOf(labels, K, priors, 'multinomialNaiveBayes')
      for (const u of v)
        if (!(u >= 0))
          throw new DomainError('multinomialNaiveBayes', 'multinomialNaiveBayes: features must be non-negative counts')
      const counts = new Float64Array(K * d).fill(alpha)
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) counts[labels[i] * d + j] += v[i * d + j]
      const logProb = new Float64Array(K * d)
      for (let c = 0; c < K; c++) {
        let total = 0
        for (let j = 0; j < d; j++) total += counts[c * d + j]
        for (let j = 0; j < d; j++) logProb[c * d + j] = Math.log(counts[c * d + j]) - Math.log(total)
      }
      const head = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'multinomialNaiveBayes')
        const out = new Float64Array(m * K)
        for (let i = 0; i < m; i++) {
          for (let c = 0; c < K; c++) {
            let s = logPrior[c]
            for (let j = 0; j < d; j++) s += qv[i * d + j] * logProb[c * d + j]
            out[i * K + c] = s
          }
        }
        return out
      }
      return {
        kind: 'model',
        name: 'multinomial-naive-bayes',
        classes: K,
        logPrior: vec(logPrior),
        featureLogProb: mat(logProb, K, d),
        featureCounts: mat(counts, K, d),
        ...probabilityModel(head, (h, m) => softmaxRows(h, m, K), K),
      }
    },
  }
}

/**
 * Bernoulli naive Bayes for binary features, as scikit-learn's `BernoulliNB`: feature $j$ is 1 with probability
 * $\theta_{kj} = (N_{kj} + \alpha) / (N_k + 2\alpha)$ in class $k$ ($N_{kj}$ the rows of class $k$ with a one,
 * $N_k$ the rows of class $k$), and absent features count too (unlike the multinomial model).
 *
 * @param params `alpha`: the additive (Laplace) smoothing $\alpha$ (default 1). `binarize`: features, in training and
 *   at prediction, become 1 when greater than it and 0 otherwise (default 0); `null` uses them as given, for features
 *   that already are 0 or 1. `priors`: the class priors, one weight per class (normalised; default the training
 *   frequencies).
 * @returns The estimator: `fit({ x, y })` returns a `DiscreteNaiveBayesModel`.
 *
 * @example Which words appear, in sport and politics
 * // Whether the words (ball, goal, vote, poll) appear in four short texts: sport (0) or politics (1).
 * const x = tensor([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 1]])
 * const model = bernoulliNaiveBayes().fit({ x, y: tensor([0, 0, 1, 1]) })
 * print('P(word | class) =', exp(model.featureLogProb))
 * print('P(politics) =', model.predictive(tensor([[1, 0, 0, 0], [0, 0, 1, 0]])).mean())
 */
export function bernoulliNaiveBayes(
  params: { alpha?: number; binarize?: number | null; priors?: readonly number[] } = {},
): Estimator<Supervised<Tensor, Tensor>, DiscreteNaiveBayesModel> {
  const { alpha = 1, binarize = 0, priors } = params
  const bit = (u: number) => (binarize === null ? u : u > binarize ? 1 : 0)
  return {
    name: 'bernoulli-naive-bayes',
    params: { alpha, binarize, priors },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'bernoulliNaiveBayes')
      const { y: labels, k: K } = classLabels(y, n, 'bernoulliNaiveBayes')
      const { counts: classCounts, logPrior } = priorsOf(labels, K, priors, 'bernoulliNaiveBayes')
      const counts = new Float64Array(K * d).fill(alpha)
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) counts[labels[i] * d + j] += bit(v[i * d + j])
      const logP = new Float64Array(K * d)
      const logQ = new Float64Array(K * d)
      for (let c = 0; c < K; c++) {
        for (let j = 0; j < d; j++) {
          const p = counts[c * d + j] / (classCounts[c] + 2 * alpha)
          logP[c * d + j] = Math.log(p)
          logQ[c * d + j] = Math.log1p(-p)
        }
      }
      const head = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'bernoulliNaiveBayes')
        const out = new Float64Array(m * K)
        for (let i = 0; i < m; i++) {
          for (let c = 0; c < K; c++) {
            let s = logPrior[c]
            for (let j = 0; j < d; j++) {
              const b = bit(qv[i * d + j])
              s += b * logP[c * d + j] + (1 - b) * logQ[c * d + j]
            }
            out[i * K + c] = s
          }
        }
        return out
      }
      return {
        kind: 'model',
        name: 'bernoulli-naive-bayes',
        classes: K,
        logPrior: vec(logPrior),
        featureLogProb: mat(logP, K, d),
        featureCounts: mat(counts, K, d),
        ...probabilityModel(head, (h, m) => softmaxRows(h, m, K), K),
      }
    },
  }
}

// ── Discriminant analysis ────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Cholesky factor of a covariance, with $\log\det\Sigmamat$ and the jitter that was needed. Throws
 * `DomainError` when the covariance does not factor even with jitter (it is not positive semi-definite), rather than
 * return a partial factor that gives infinite or NaN log-likelihoods.
 *
 * @param cov The covariance $\Sigmamat$, $d \times d$ row-major (only its lower triangle is read).
 * @param d The number of features.
 * @param where The public function, named in the error.
 * @returns The factor `L` (a tensor), `logDet` $= \log\det(\Sigmamat + j\Imat)$ and the `jitter` $j$ added to the
 *   diagonal.
 */
function factorCovariance(cov: Float64Array, d: number, where: string) {
  const { L, jitter, failed } = cholesky(fromData(cov, [d, d]))
  if (failed) throw new DomainError(where, `${where}: a covariance is not positive definite, even with jitter`)
  const l = Float64Array.from(L.data as Float64Array)
  let logDet = 0
  for (let j = 0; j < d; j++) logDet += 2 * Math.log(l[j * d + j])
  return { L, logDet, jitter }
}

/**
 * $\lVert \Lmat^{-1}(\xvec_i - \muvec) \rVert^2$ for every row $\xvec_i$ of a query, by `solveTriangular`: the squared
 * Mahalanobis distances to $\muvec$ when $\Lmat\Lmat^\top = \Sigmamat$.
 *
 * @param L The lower-triangular Cholesky factor of $\Sigmamat$, $d \times d$.
 * @param q The query rows, $m \times d$ row-major.
 * @param m The number of query rows.
 * @param d The number of features.
 * @param mean The centre $\muvec$, $d$ values.
 * @returns The $m$ squared distances.
 */
function mahalanobisRows(L: Tensor, q: Float64Array, m: number, d: number, mean: Float64Array): Float64Array {
  const res = new Float64Array(d * m) // column i is xᵢ − μ
  for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) res[j * m + i] = q[i * d + j] - mean[j]
  return dense.data(sum(square(solveTriangular(L, fromData(res, [d, m]))), 0))
}

/** A fitted linear or quadratic discriminant analysis. */
export interface DiscriminantModel extends GenerativeClassifier {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'linear-discriminant' | 'quadratic-discriminant'
  /** Class means, $K \times d$. */
  readonly means: Tensor
  /**
   * LDA: the pooled within-class covariance, $d \times d$ (after shrinkage). QDA: one covariance per class,
   * $K \times d \times d$ (after regularisation).
   */
  readonly covariance: Tensor
  /**
   * The jitter added to each covariance's diagonal to factor it (0 when it was positive definite): one value for LDA,
   * $K$ for QDA.
   */
  readonly jitter: Tensor
}

/** A fitted LDA, which also projects onto its discriminant directions. */
export interface LinearDiscriminantModel extends DiscriminantModel {
  /** Always `'model'`. */
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'linear-discriminant'
  /**
   * Discriminant directions as the columns of a $d \times r$ matrix, $r = \min(K - 1, d)$, solving
   * $\Smat_b \wvec = \lambda \Smat_w \wvec$ with $\wvec^\top \Smat_w \wvec = 1$ ($\Smat_w$ the shared covariance,
   * $\Smat_b$ the prior-weighted scatter of the class means), so the projected classes have unit within-class
   * variance. Ordered by $\lambda$, descending.
   */
  readonly scalings: Tensor
  /** The share of the between-class variance along each direction, $r$ values. */
  readonly explainedVarianceRatio: Tensor
  /**
   * Projections of the rows of $\xvec$ minus the prior-weighted mean of the class means onto `scalings`,
   * $m \times r$.
   */
  transform(x: Tensor): Tensor
}

/**
 * Linear discriminant analysis: Gaussian classes sharing one covariance $\Sigmamat$, estimated as the within-class
 * scatter divided by $n$ (the maximum-likelihood estimate, which scikit-learn's predictions use). The discriminant is
 * $\delta_k(\xvec) = \xvec^\top\Sigmamat^{-1}\muvec_k - \tfrac12 \muvec_k^\top\Sigmamat^{-1}\muvec_k + \log \pi_k$,
 * linear in $\xvec$; `forward` gives $\log \pi_k - \tfrac12 (\xvec - \muvec_k)^\top\Sigmamat^{-1}(\xvec - \muvec_k)$,
 * which differs from it by a term shared by the classes. The model also projects onto its discriminant directions
 * (Fisher, 1936; Rao, 1948), as scikit-learn's `LinearDiscriminantAnalysis.transform`. Throws `DomainError` at once
 * when `shrinkage` is outside $[0, 1]$.
 *
 * @param params `priors`: the class priors, one weight per class (normalised; default the training frequencies).
 *   `shrinkage` $s \in [0, 1]$ (default 0) replaces $\Sigmamat$ by $(1 - s)\Sigmamat + s (\trace \Sigmamat / d) \Imat$.
 * @returns The estimator: `fit({ x, y })` returns a `LinearDiscriminantModel`.
 *
 * @example Two Gaussian classes sharing a covariance
 * // Two Gaussian classes of twenty points, centred at (0, 0) and (3, 3).
 * const x = concat([normals(stream(0), [20, 2]), add(normals(stream(1), [20, 2]), tensor([3, 3]))])
 * const y = tensor(Array.from({ length: 40 }, (_, i) => (i < 20 ? 0 : 1)))
 * const model = linearDiscriminant().fit({ x, y })
 * print('class means =', model.means)
 * print('shared covariance =', model.covariance)
 * print('P(class 1) =', model.predictive(tensor([[0, 0], [1.5, 1.5], [3, 3]])).mean())
 *
 * @example Projected onto the discriminant direction, the class means are about four apart
 * // Two Gaussian classes of twenty points, centred at (0, 0) and (3, 3).
 * const x = concat([normals(stream(0), [20, 2]), add(normals(stream(1), [20, 2]), tensor([3, 3]))])
 * const y = tensor(Array.from({ length: 40 }, (_, i) => (i < 20 ? 0 : 1)))
 * const model = linearDiscriminant().fit({ x, y })
 * print('direction =', model.scalings)
 * print('projected class centres =', model.transform(tensor([[0, 0], [3, 3]])))
 */
export function linearDiscriminant(
  params: { priors?: readonly number[]; shrinkage?: number } = {},
): Estimator<Supervised<Tensor, Tensor>, LinearDiscriminantModel> {
  const { priors, shrinkage = 0 } = params
  if (!(shrinkage >= 0 && shrinkage <= 1))
    throw new DomainError('linearDiscriminant', 'linearDiscriminant: shrinkage must lie in [0, 1]')
  return {
    name: 'linear-discriminant',
    params: { priors, shrinkage },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'linearDiscriminant')
      const { y: labels, k: K } = classLabels(y, n, 'linearDiscriminant')
      const { counts, logPrior } = priorsOf(labels, K, priors, 'linearDiscriminant')
      const means = classMeans(v, labels, n, d, K, counts)
      const cov = new Float64Array(d * d)
      for (let i = 0; i < n; i++) {
        const c = labels[i]
        for (let a = 0; a < d; a++) {
          const ra = v[i * d + a] - means[c * d + a]
          for (let b = 0; b <= a; b++) cov[a * d + b] += ra * (v[i * d + b] - means[c * d + b])
        }
      }
      for (let a = 0; a < d; a++) for (let b = 0; b <= a; b++) cov[b * d + a] = cov[a * d + b] /= n
      if (shrinkage > 0) {
        let tr = 0
        for (let a = 0; a < d; a++) tr += cov[a * d + a]
        for (let a = 0; a < d; a++) {
          for (let b = 0; b < d; b++)
            cov[a * d + b] = (1 - shrinkage) * cov[a * d + b] + (a === b ? (shrinkage * tr) / d : 0)
        }
      }
      const f = factorCovariance(cov, d, 'linearDiscriminant')
      // Discriminant directions: with Σ = LLᵀ, the eigenvectors u of L⁻¹ S_b L⁻ᵀ give w = L⁻ᵀ u (Fisher, 1936; Rao,
      // 1948). S_b is the prior-weighted scatter of the class means about their weighted mean.
      const prior = Float64Array.from(logPrior, Math.exp)
      const centre = new Float64Array(d)
      for (let c = 0; c < K; c++) for (let j = 0; j < d; j++) centre[j] += prior[c] * means[c * d + j]
      // Columns √π_k (μ_k − μ̄), then L⁻¹ of them: Zᵀ [d, K].
      const D = new Float64Array(d * K)
      for (let c = 0; c < K; c++)
        for (let i = 0; i < d; i++) D[i * K + c] = Math.sqrt(prior[c]) * (means[c * d + i] - centre[i])
      const Zt = dense.data(solveTriangular(f.L, fromData(D, [d, K])))
      const Z = new Float64Array(K * d) // rows: √π_k L⁻¹(μ_k − μ̄)
      for (let c = 0; c < K; c++) for (let i = 0; i < d; i++) Z[c * d + i] = Zt[i * K + c]
      const B = new Float64Array(d * d)
      for (let c = 0; c < K; c++)
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) B[a * d + b] += Z[c * d + a] * Z[c * d + b]
      const e = eigh(fromData(B, [d, d]))
      const r = Math.min(K - 1, d)
      const U = e.vectors.data as Float64Array
      const lambda = e.values.data as Float64Array
      let lambdaTotal = 0
      for (let j = 0; j < d; j++) lambdaTotal += Math.max(lambda[j], 0)
      // w = L⁻ᵀ u for the first r eigenvectors.
      const Ur = new Float64Array(d * r)
      for (let i = 0; i < d; i++) for (let col = 0; col < r; col++) Ur[i * r + col] = U[i * d + col]
      const W = dense.data(solveTriangular(f.L, fromData(Ur, [d, r]), { transpose: true }))
      const head = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'linearDiscriminant')
        const out = new Float64Array(m * K)
        for (let c = 0; c < K; c++) {
          const r2 = mahalanobisRows(f.L, qv, m, d, means.subarray(c * d, (c + 1) * d))
          for (let i = 0; i < m; i++) out[i * K + c] = logPrior[c] - 0.5 * r2[i]
        }
        return out
      }
      return {
        kind: 'model',
        name: 'linear-discriminant',
        classes: K,
        logPrior: vec(logPrior),
        means: mat(means, K, d),
        covariance: mat(cov, d, d),
        jitter: vec([f.jitter]),
        scalings: mat(W, d, r),
        explainedVarianceRatio: vec(Array.from({ length: r }, (_, j) => Math.max(lambda[j], 0) / (lambdaTotal || 1))),
        transform: (q: Tensor) => {
          const { n: m, v: qv } = inputs(q, d, 'linearDiscriminant.transform')
          const out = new Float64Array(m * r)
          for (let i = 0; i < m; i++) {
            for (let col = 0; col < r; col++) {
              let s = 0
              for (let j = 0; j < d; j++) s += (qv[i * d + j] - centre[j]) * W[j * r + col]
              out[i * r + col] = s
            }
          }
          return mat(out, m, r)
        },
        ...probabilityModel(head, (h, m) => softmaxRows(h, m, K), K),
      }
    },
  }
}

/**
 * Quadratic discriminant analysis: Gaussian classes each with its own covariance $\Sigmamat_k$ (the
 * maximum-likelihood estimate, divided by $n_k$, as scikit-learn $\ge 1.6$), so the boundaries are quadrics:
 * $\delta_k(\xvec) = \log \pi_k - \tfrac12 \log\det\Sigmamat_k - \tfrac12 r_k^2$, with the squared Mahalanobis
 * distance $r_k^2 = (\xvec - \muvec_k)^\top\Sigmamat_k^{-1}(\xvec - \muvec_k)$, which `forward` returns. Throws
 * `DomainError` at once when `regularisation` is outside $[0, 1]$, and `fit` throws it when a class has fewer than two
 * rows.
 *
 * @param params `priors`: the class priors, one weight per class (normalised; default the training frequencies).
 *   `regularisation` $\rho$, in $[0, 1]$ (default 0), replaces each $\Sigmamat_k$ by
 *   $(1 - \rho)\Sigmamat_k + \rho\Imat$, as scikit-learn's `reg_param`.
 * @returns The estimator: `fit({ x, y })` returns a `DiscriminantModel`.
 *
 * @example A tight class inside a spread one: the boundary curves around the tight class
 * // Class 0 is tight around the origin, class 1 three times as spread.
 * const x = concat([mul(normals(stream(0), [30, 2]), 0.5), mul(normals(stream(1), [30, 2]), 1.5)])
 * const y = tensor(Array.from({ length: 60 }, (_, i) => (i < 30 ? 0 : 1)))
 * const model = quadraticDiscriminant().fit({ x, y })
 * print('covariances =', model.covariance)
 * print('P(class 1) at 0, 1 and 2 from the origin =', model.predictive(tensor([[0, 0], [1, 0], [2, 0]])).mean())
 */
export function quadraticDiscriminant(
  params: { priors?: readonly number[]; regularisation?: number } = {},
): Estimator<Supervised<Tensor, Tensor>, DiscriminantModel> {
  const { priors, regularisation = 0 } = params
  if (!(regularisation >= 0 && regularisation <= 1))
    throw new DomainError('quadraticDiscriminant', 'quadraticDiscriminant: regularisation must lie in [0, 1]')
  return {
    name: 'quadratic-discriminant',
    params: { priors, regularisation },
    fit({ x, y }) {
      const { n, d, v } = matrix(x, 'quadraticDiscriminant')
      const { y: labels, k: K } = classLabels(y, n, 'quadraticDiscriminant')
      const { counts, logPrior } = priorsOf(labels, K, priors, 'quadraticDiscriminant')
      for (let c = 0; c < K; c++) {
        if (counts[c] < 2)
          throw new DomainError('quadraticDiscriminant', `quadraticDiscriminant: class ${c} has fewer than two rows`)
      }
      const means = classMeans(v, labels, n, d, K, counts)
      const covs = new Float64Array(K * d * d)
      for (let i = 0; i < n; i++) {
        const c = labels[i]
        for (let a = 0; a < d; a++) {
          const ra = v[i * d + a] - means[c * d + a]
          for (let b = 0; b < d; b++) covs[c * d * d + a * d + b] += ra * (v[i * d + b] - means[c * d + b])
        }
      }
      const factors: ReturnType<typeof factorCovariance>[] = []
      for (let c = 0; c < K; c++) {
        const cov = covs.subarray(c * d * d, (c + 1) * d * d)
        for (let a = 0; a < d; a++) {
          for (let b = 0; b < d; b++) {
            cov[a * d + b] = (1 - regularisation) * (cov[a * d + b] / counts[c]) + (a === b ? regularisation : 0)
          }
        }
        factors.push(factorCovariance(Float64Array.from(cov), d, 'quadraticDiscriminant'))
      }
      const head = (q: Tensor) => {
        const { n: m, v: qv } = inputs(q, d, 'quadraticDiscriminant')
        const out = new Float64Array(m * K)
        for (let c = 0; c < K; c++) {
          const r2 = mahalanobisRows(factors[c].L, qv, m, d, means.subarray(c * d, (c + 1) * d))
          for (let i = 0; i < m; i++) out[i * K + c] = logPrior[c] - 0.5 * factors[c].logDet - 0.5 * r2[i]
        }
        return out
      }
      return {
        kind: 'model',
        name: 'quadratic-discriminant',
        classes: K,
        logPrior: vec(logPrior),
        means: mat(means, K, d),
        covariance: fromData(covs, [K, d, d]),
        jitter: vec(factors.map((f) => f.jitter)),
        ...probabilityModel(head, (h, m) => softmaxRows(h, m, K), K),
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'gaussianNaiveBayes',
    module: 'learning/generative-classifiers',
    name: 'Gaussian naive Bayes',
    summary: 'Class-conditional independent Gaussians per feature.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({ varianceSmoothing: real(1e-12, 1e-3, { default: 1e-9, scale: 'log' }) }),
    notes: ['naive-bayes'],
    cite: ['hastie2009'],
  },
  gaussianNaiveBayes,
)

defineModel(
  {
    key: 'multinomialNaiveBayes',
    module: 'learning/generative-classifiers',
    name: 'Multinomial naive Bayes',
    summary: 'Class-conditional multinomial counts with additive smoothing.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({ alpha: real(0, 10, { default: 1, label: 'α' }) }),
    notes: ['naive-bayes'],
    cite: ['manning2008'],
  },
  multinomialNaiveBayes,
)

defineModel(
  {
    key: 'bernoulliNaiveBayes',
    module: 'learning/generative-classifiers',
    name: 'Bernoulli naive Bayes',
    summary: 'Class-conditional independent binary features with additive smoothing.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({ alpha: real(0, 10, { default: 1, label: 'α' }), binarize: real(-10, 10, { default: 0 }) }),
    notes: ['naive-bayes'],
    cite: ['manning2008'],
  },
  bernoulliNaiveBayes,
)

defineModel(
  {
    key: 'linearDiscriminant',
    module: 'learning/generative-classifiers',
    name: 'Linear discriminant analysis',
    summary: 'Gaussian classes with a shared covariance; also projects onto the discriminant directions.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score', 'transform'],
    hyper: space({ shrinkage: real(0, 1, { default: 0 }) }),
    notes: ['linear-discriminant-analysis'],
    cite: ['hastie2009'],
  },
  linearDiscriminant,
)

defineModel(
  {
    key: 'quadraticDiscriminant',
    module: 'learning/generative-classifiers',
    name: 'Quadratic discriminant analysis',
    summary: 'Gaussian classes, each with its own covariance.',
    task: 'classification',
    capabilities: ['forward', 'decide', 'predictive', 'score'],
    hyper: space({ regularisation: real(0, 1, { default: 0 }) }),
    notes: ['quadratic-discriminant-analysis'],
    cite: ['hastie2009'],
  },
  quadraticDiscriminant,
)
