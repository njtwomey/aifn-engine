/**
 * Metrics of generative models computed from feature vectors (the caller supplies the features, e.g. Inception or CLIP
 * embeddings): the Fréchet distance and FID, the kernel inception distance (with subset averaging), the Inception
 * score, k-NN precision and recall, density and coverage, and CLIPScore.
 *
 * Features are matrices with one sample per row ($n \times d$), real and generated sets having the same $d$. FID
 * compares the sets' Gaussian moments, KID their kernel mean embeddings; precision, recall, density and coverage
 * compare the $k$-nearest-neighbour balls around each point, so they cost $O(nm)$ distances. No network is run here:
 * the scores depend on the feature extractor as much as on the samples, so compare only scores from the same one.
 */

import { eigh, pairwiseDistances } from 'aifn-compute/numerics/linalg'
import type { Stream } from 'aifn-compute/foundation/random'
import { defineMetric, type Rows } from 'aifn-compute/learning/metrics'
import { denseMatrix as dense, divide, matrix, type Dense } from 'aifn-compute/learning/metrics'
import { child, integers } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * Column means and the sample covariance (divisor $n - 1$, as `numpy.cov`) of the rows of $\Xmat$.
 *
 * @param X The samples, an $n \times d$ matrix with one sample per row ($n \ge 2$).
 * @returns `mean`, the $d$ column means, and `cov`, the $d \times d$ covariance, row-major.
 */
function moments(X: Dense): { mean: Float64Array; cov: Float64Array } {
  const { rows: n, cols: d, data } = X
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) mean[c] += data[i * d + c] / n
  const cov = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) {
      const da = data[i * d + a] - mean[a]
      for (let b = 0; b <= a; b++) cov[a * d + b] += (da * (data[i * d + b] - mean[b])) / (n - 1)
    }
  for (let a = 0; a < d; a++) for (let b = a + 1; b < d; b++) cov[a * d + b] = cov[b * d + a]
  return { mean, cov }
}

/**
 * Eigenvalues of a symmetric matrix, with round-off negatives set to 0 and counted. One below
 * $-10^{-10} d \max(1, \lvert\lambda_{\max}\rvert)$ throws `DomainError`: the matrix is not positive semi-definite.
 *
 * @param a The symmetric matrix, $d \times d$ row-major; not modified.
 * @param d The number of rows (and columns).
 * @param what The caller's name for the matrix, for the error message.
 * @returns `lambda`, the eigenvalues as `eigh` orders them (negatives clipped to 0), `V`, the eigenvectors as columns,
 *   row-major, and `clipped`, how many eigenvalues were set to 0.
 */
function psdEigen(a: Float64Array, d: number, what: string) {
  const { values: v, vectors } = eigh(matrix(a, d, d))
  const lambda = Float64Array.from(v.data)
  const tol = 1e-10 * Math.max(1, Math.abs(lambda[0])) * d
  let clipped = 0
  for (let i = 0; i < d; i++)
    if (lambda[i] < 0) {
      if (lambda[i] < -tol)
        throw new DomainError('metrics', `metrics: ${what}: matrix is not positive semi-definite (λ = ${lambda[i]})`)
      lambda[i] = 0
      clipped++
    }
  return { lambda, V: vectors.data as Float64Array, clipped }
}

/** The Fréchet distance between two Gaussians and its parts. */
export type FrechetDistance = {
  /** The squared distance, `meanTerm` plus `covarianceTerm`. */
  distance: number
  /** $\lVert \muvec_1 - \muvec_2 \rVert^2$. */
  meanTerm: number
  /** $\trace(\Sigmamat_1 + \Sigmamat_2 - 2(\Sigmamat_1\Sigmamat_2)^{1/2})$. */
  covarianceTerm: number
  /** Eigenvalues at round-off level below 0 that were set to 0 in the matrix square roots. */
  clippedEigenvalues: number
}

/**
 * The Fréchet (Wasserstein-2) distance between $\Gauss(\muvec_1, \Sigmamat_1)$ and $\Gauss(\muvec_2, \Sigmamat_2)$,
 * squared: $\lVert \muvec_1 - \muvec_2 \rVert^2 + \trace(\Sigmamat_1 + \Sigmamat_2 - 2(\Sigmamat_1\Sigmamat_2)^{1/2})$
 * (Dowson and Landau 1982; frechet-inception-distance).
 * $\trace(\Sigmamat_1\Sigmamat_2)^{1/2} = \sum_i \sqrt{\lambda_i}$ with $\lambda_i$ the eigenvalues of the symmetric
 * $\Sigmamat_1^{1/2}\Sigmamat_2\Sigmamat_1^{1/2}$, which is similar to $\Sigmamat_1\Sigmamat_2$. Mismatched
 * dimensions throw `ShapeError`, and a covariance that is clearly not positive semi-definite throws `DomainError`.
 *
 * @param mean1 The first mean $\muvec_1$, $d$ values.
 * @param cov1 The first covariance $\Sigmamat_1$, $d \times d$ and symmetric.
 * @param mean2 The second mean $\muvec_2$, $d$ values.
 * @param cov2 The second covariance $\Sigmamat_2$, $d \times d$ and symmetric.
 * @returns The squared distance, its mean and covariance terms, and how many round-off eigenvalues were clipped.
 *
 * @example A shift by 1 and a doubling of the scale: $1 + (2 + 8 - 2 \cdot 4)$
 * print(frechetDistance([0, 0], [[1, 0], [0, 1]], [1, 0], [[4, 0], [0, 4]]))
 */
export function frechetDistance(
  mean1: ArrayLike<number>,
  cov1: Rows,
  mean2: ArrayLike<number>,
  cov2: Rows,
): FrechetDistance {
  const S1 = dense(cov1, 'frechetDistance')
  const S2 = dense(cov2, 'frechetDistance')
  const d = S1.rows
  if (S1.cols !== d || S2.rows !== d || S2.cols !== d || mean1.length !== d || mean2.length !== d)
    throw new ShapeError('metrics', 'metrics: frechetDistance: dimensions differ')
  let meanTerm = 0
  for (let c = 0; c < d; c++) meanTerm += (mean1[c] - mean2[c]) ** 2
  // Σ₁^{1/2} = V diag(√λ) Vᵀ.
  const e1 = psdEigen(S1.data, d, 'frechetDistance Σ₁')
  const root = new Float64Array(d * d)
  for (let a = 0; a < d; a++)
    for (let b = 0; b < d; b++) {
      let s = 0
      for (let k = 0; k < d; k++) s += e1.V[a * d + k] * Math.sqrt(e1.lambda[k]) * e1.V[b * d + k]
      root[a * d + b] = s
    }
  // M = Σ₁^{1/2} Σ₂ Σ₁^{1/2}.
  const tmp = new Float64Array(d * d)
  for (let a = 0; a < d; a++)
    for (let b = 0; b < d; b++) for (let k = 0; k < d; k++) tmp[a * d + b] += root[a * d + k] * S2.data[k * d + b]
  const M = new Float64Array(d * d)
  for (let a = 0; a < d; a++)
    for (let b = 0; b < d; b++) for (let k = 0; k < d; k++) M[a * d + b] += tmp[a * d + k] * root[k * d + b]
  for (let a = 0; a < d; a++)
    for (let b = a + 1; b < d; b++) M[a * d + b] = M[b * d + a] = (M[a * d + b] + M[b * d + a]) / 2
  const e = psdEigen(M, d, 'frechetDistance Σ₁^{1/2}Σ₂Σ₁^{1/2}')
  let traceRoot = 0
  for (const l of e.lambda) traceRoot += Math.sqrt(l)
  let trace = 0
  for (let c = 0; c < d; c++) trace += S1.data[c * d + c] + S2.data[c * d + c]
  const covarianceTerm = trace - 2 * traceRoot
  return { distance: meanTerm + covarianceTerm, meanTerm, covarianceTerm, clippedEigenvalues: e1.clipped + e.clipped }
}

/**
 * The Fréchet inception distance (Heusel et al. 2017) between the rows of real and generated feature matrices: the
 * Fréchet distance (`frechetDistance`) of their means and sample covariances (divisor $n - 1$, as the reference
 * implementation). Biased upwards with finite samples; compare at equal sample sizes.
 *
 * @param realFeatures The real samples' features, $n \times d$, one sample per row ($n \ge 2$).
 * @param generatedFeatures The generated samples' features, $m \times d$.
 * @returns The squared Fréchet distance, at least 0 up to round-off.
 *
 * @example The same distribution, and one shifted by 1 in each of two coordinates
 * const real = normal(stream(0), 0, 1, { shape: [500, 2] })
 * const same = normal(stream(1), 0, 1, { shape: [500, 2] })
 * const shifted = normal(stream(2), 1, 1, { shape: [500, 2] })
 * print('FID, same distribution =', fid(real, same))
 * print('FID, shifted =', fid(real, shifted))
 */
export const fid = defineMetric(
  {
    module: 'applied/evaluation/generative',
    key: 'fid',
    name: 'Fréchet inception distance',
    inputs: 'features',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['frechet-inception-distance'],
  },
  (realFeatures: Rows, generatedFeatures: Rows): number => {
    const a = moments(dense(realFeatures, 'fid'))
    const b = moments(dense(generatedFeatures, 'fid'))
    const d = a.mean.length
    return frechetDistance(a.mean, matrix(a.cov, d, d), b.mean, matrix(b.cov, d, d)).distance
  },
)

/**
 * Options of the polynomial kernel $k(\xvec, \yvec) = (\gamma \xvec^\top\yvec + c)^p$: `degree` $p$ (default 3),
 * `gamma` $\gamma$ (default $1/d$, $d$ the feature dimension) and `coef0` $c$ (default 1).
 */
export type PolynomialKernel = { degree?: number; gamma?: number; coef0?: number }

/**
 * The unbiased estimate of the squared MMD with a polynomial kernel between the rows of $\Xmat$ and $\Ymat$: the
 * within-set kernel means over distinct pairs, less twice the cross-set mean. Feature dimensions that differ throw
 * `ShapeError`.
 *
 * @param X The first sample, $n \times d$ ($n \ge 2$).
 * @param Y The second sample, $m \times d$ ($m \ge 2$).
 * @param k The kernel's degree, $\gamma$ and constant (see `PolynomialKernel`).
 * @returns The estimate of $\text{MMD}^2$, which can be slightly negative.
 */
function mmd2(X: Dense, Y: Dense, k: PolynomialKernel): number {
  const d = X.cols
  if (Y.cols !== d) throw new ShapeError('metrics', 'metrics: kid: feature dimensions differ')
  const gamma = k.gamma ?? 1 / d
  const coef = k.coef0 ?? 1
  const degree = k.degree ?? 3
  const kernel = (A: Dense, i: number, B: Dense, j: number) => {
    let s = 0
    for (let c = 0; c < d; c++) s += A.data[i * d + c] * B.data[j * d + c]
    return (gamma * s + coef) ** degree
  }
  const within = (A: Dense) => {
    let s = 0
    for (let i = 0; i < A.rows; i++) for (let j = 0; j < A.rows; j++) if (i !== j) s += kernel(A, i, A, j)
    return s / (A.rows * (A.rows - 1))
  }
  let cross = 0
  for (let i = 0; i < X.rows; i++) for (let j = 0; j < Y.rows; j++) cross += kernel(X, i, Y, j)
  return within(X) + within(Y) - (2 * cross) / (X.rows * Y.rows)
}

/**
 * The kernel inception distance (Bińkowski et al. 2018; kernel-inception-distance): the unbiased estimate of the
 * squared MMD between real and generated features with the cubic kernel $(\xvec^\top\yvec/d + 1)^3$. Unbiased at any
 * sample size, so it can be slightly negative; do not clip it. Costs $O((n + m)^2 d)$.
 *
 * @param realFeatures The real samples' features, $n \times d$, one sample per row ($n \ge 2$).
 * @param generatedFeatures The generated samples' features, $m \times d$ ($m \ge 2$).
 * @param options The kernel (see `PolynomialKernel`; default the cubic kernel above).
 * @returns The KID estimate: near 0 for the same distribution.
 *
 * @example The same distribution, and a shifted one
 * const real = normal(stream(0), 0, 1, { shape: [200, 2] })
 * print('KID, same distribution =', kid(real, normal(stream(1), 0, 1, { shape: [200, 2] })))
 * print('KID, shifted =', kid(real, normal(stream(2), 1, 1, { shape: [200, 2] })))
 */
export const kid = defineMetric(
  {
    module: 'applied/evaluation/generative',
    key: 'kid',
    name: 'Kernel inception distance',
    inputs: 'features',
    direction: 'lower',
    range: [-Infinity, Infinity],
    notes: ['kernel-inception-distance'],
  },
  (realFeatures: Rows, generatedFeatures: Rows, options: PolynomialKernel = {}): number =>
    mmd2(dense(realFeatures, 'kid'), dense(generatedFeatures, 'kid'), options),
)

/**
 * KID averaged over `subsets` random subsets of `subsetSize` rows drawn without replacement with the stream, as
 * practice does for large sets, with the standard deviation over subsets. A `subsetSize` larger than either set throws
 * `DomainError`.
 *
 * @param s The stream the subsets are drawn from. Subset $k$ uses the children `child(s, 'real', k)` and
 *   `child(s, 'generated', k)`, so `s` itself is not advanced.
 * @param realFeatures The real samples' features, $n \times d$, one sample per row.
 * @param generatedFeatures The generated samples' features, $m \times d$.
 * @param options The kernel (see `PolynomialKernel`), and the subsets.
 * @param options.subsets The number of subsets (default 10).
 * @param options.subsetSize The rows per subset, drawn from each set (default $\min(1000, n, m)$).
 * @returns `mean` and `std` (population standard deviation) of the subset KIDs, and the KIDs themselves as `values`.
 *
 * @example Five subsets of 50 from sets of 200
 * const real = normal(stream(0), 0, 1, { shape: [200, 2] })
 * const generated = normal(stream(1), 0.5, 1, { shape: [200, 2] })
 * const { mean, std } = kidSubsets(stream(2), real, generated, { subsets: 5, subsetSize: 50 })
 * print('KID =', mean, '+/-', std)
 * print('on all rows =', kid(real, generated))
 */
export function kidSubsets(
  s: Stream,
  realFeatures: Rows,
  generatedFeatures: Rows,
  options: PolynomialKernel & { subsets?: number; subsetSize?: number } = {},
): { mean: number; std: number; values: number[] } {
  const X = dense(realFeatures, 'kidSubsets')
  const Y = dense(generatedFeatures, 'kidSubsets')
  const size = options.subsetSize ?? Math.min(1000, X.rows, Y.rows)
  const pick = (A: Dense, stream: Stream): Dense => {
    const idx = Array.from({ length: A.rows }, (_, i) => i)
    for (let i = 0; i < size; i++) {
      const j = i + integers(stream, A.rows - i)
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    const data = new Float64Array(size * A.cols)
    for (let i = 0; i < size; i++) data.set(A.data.subarray(idx[i] * A.cols, (idx[i] + 1) * A.cols), i * A.cols)
    return { rows: size, cols: A.cols, data }
  }
  const vals = Array.from({ length: options.subsets ?? 10 }, (_, k) =>
    mmd2(pick(X, child(s, 'real', k)), pick(Y, child(s, 'generated', k)), options),
  )
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length
  const std = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length)
  return { mean, std, values: vals }
}

/**
 * The Inception score $\exp(\expect_x \KL(p(y \mid x) \Vert p(y))) = e^{I(x; y)}$ from an $n \times K$ matrix of
 * class probabilities of generated images (Salimans et al. 2016; inception-score), in $[1, K]$: high when each image is
 * confidently one class and the classes are used evenly. With `splits` $> 1$, the mean and standard deviation over
 * equal consecutive splits, as the reference implementation reports.
 *
 * @param probabilities The class probabilities $p(y \mid x)$ of each generated image, an $n \times K$ matrix whose rows
 *   sum to 1. The marginal $p(y)$ is their mean over the rows of a split.
 * @param options `splits`, the number of consecutive row blocks scored separately (default 1).
 * @returns `mean` and `std` (population standard deviation) of the splits' scores; `std` is 0 for one split.
 *
 * @example Confident and diverse, confident but collapsed, and uncertain
 * print('diverse:', inceptionScoreSplits([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
 * print('collapsed:', inceptionScoreSplits([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))
 * print('uncertain:', inceptionScoreSplits([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]))
 *
 * @example Two splits of four rows
 * const p = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
 * print(inceptionScoreSplits(p, { splits: 2 }))
 */
export function inceptionScoreSplits(
  probabilities: Rows,
  options: { splits?: number } = {},
): { mean: number; std: number } {
  const P = dense(probabilities, 'inceptionScore')
  const splits = options.splits ?? 1
  const scores: number[] = []
  for (let s = 0; s < splits; s++) {
    const lo = Math.floor((s * P.rows) / splits)
    const hi = Math.floor(((s + 1) * P.rows) / splits)
    const marginal = new Float64Array(P.cols)
    for (let i = lo; i < hi; i++) for (let k = 0; k < P.cols; k++) marginal[k] += P.data[i * P.cols + k] / (hi - lo)
    let kl = 0
    for (let i = lo; i < hi; i++)
      for (let k = 0; k < P.cols; k++) {
        const p = P.data[i * P.cols + k]
        if (p > 0) kl += (p * Math.log(p / marginal[k])) / (hi - lo)
      }
    scores.push(Math.exp(kl))
  }
  const mean = scores.reduce((a, b) => a + b, 0) / splits
  return { mean, std: Math.sqrt(scores.reduce((a, v) => a + (v - mean) ** 2, 0) / splits) }
}

/**
 * The Inception score of all rows as one split (see `inceptionScoreSplits`).
 *
 * @param probabilities The class probabilities $p(y \mid x)$ of each generated image, an $n \times K$ matrix whose rows
 *   sum to 1.
 * @returns The Inception score, in $[1, K]$.
 *
 * @example Four images spread over two classes
 * print('IS =', inceptionScore([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]))
 */
export const inceptionScore = defineMetric(
  {
    module: 'applied/evaluation/generative',
    key: 'inceptionScore',
    name: 'Inception score',
    inputs: 'probabilities',
    direction: 'higher',
    range: [1, Infinity],
    notes: ['inception-score'],
  },
  (probabilities: Rows): number => inceptionScoreSplits(probabilities).mean,
)

/**
 * Euclidean distance from each row of $\Amat$ to each row of $\Bmat$ (linalg's `pairwiseDistances`).
 *
 * @param A The first set, one point per row.
 * @param B The second set, with as many columns as `A`.
 * @returns The distances, row-major `[A.rows, B.rows]`: entry `i * B.rows + j` is from row `i` of `A` to row `j` of
 *   `B`.
 */
function distances(A: Dense, B: Dense): Float64Array {
  const D = pairwiseDistances(fromData(A.data, [A.rows, A.cols]), fromData(B.data, [B.rows, B.cols]))
  return Float64Array.from(toFlat(D))
}

/**
 * Each row's distance to its $k$-th nearest neighbour within its own set (itself excluded). Throws `DomainError` when
 * the set has no more than $k$ points.
 *
 * @param A The set, one point per row.
 * @param k Which neighbour: 1 is the nearest other point.
 * @returns The radius of each row's $k$-nearest-neighbour ball, one per row.
 */
function kthNeighbourRadii(A: Dense, k: number): Float64Array {
  if (k >= A.rows) throw new DomainError('metrics', `metrics: k = ${k} needs more than ${k} points`)
  const D = distances(A, A)
  return Float64Array.from({ length: A.rows }, (_, i) => {
    const row = Array.from(D.subarray(i * A.rows, (i + 1) * A.rows))
    row.sort((a, b) => a - b)
    return row[k] // row[0] is the point itself
  })
}

/**
 * Improved precision and recall for generative models (Kynkäänniemi et al. 2019;
 * precision-and-recall-for-generative-models): each set's manifold is the union of balls reaching each point's $k$-th
 * nearest neighbour; precision is the fraction of generated points inside the real manifold, recall the fraction of
 * real points inside the generated manifold. Each set needs more than $k$ points, or `DomainError` is thrown.
 *
 * @param realFeatures The real samples' features, $n \times d$, one sample per row.
 * @param generatedFeatures The generated samples' features, $m \times d$.
 * @param options `k`, the neighbour that sets each ball's radius (default 3).
 * @returns `precision` (fidelity) and `recall` (diversity), each in $[0, 1]$.
 *
 * @example A generator that collapses to the centre: precise, with poor recall
 * const real = normal(stream(0), 0, 1, { shape: [150, 2] })
 * const collapsed = normal(stream(1), 0, 0.2, { shape: [150, 2] })
 * print(generativePrecisionRecall(real, collapsed))
 * print('same distribution:', generativePrecisionRecall(real, normal(stream(2), 0, 1, { shape: [150, 2] })))
 */
export function generativePrecisionRecall(
  realFeatures: Rows,
  generatedFeatures: Rows,
  options: { k?: number } = {},
): { precision: number; recall: number } {
  const R = dense(realFeatures, 'generativePrecisionRecall')
  const G = dense(generatedFeatures, 'generativePrecisionRecall')
  const k = options.k ?? 3
  const rr = kthNeighbourRadii(R, k)
  const rg = kthNeighbourRadii(G, k)
  const D = distances(G, R)
  let inReal = 0
  for (let i = 0; i < G.rows; i++) {
    let hit = false
    for (let j = 0; j < R.rows && !hit; j++) hit = D[i * R.rows + j] <= rr[j]
    if (hit) inReal++
  }
  let inGenerated = 0
  for (let j = 0; j < R.rows; j++) {
    let hit = false
    for (let i = 0; i < G.rows && !hit; i++) hit = D[i * R.rows + j] <= rg[i]
    if (hit) inGenerated++
  }
  return { precision: inReal / G.rows, recall: inGenerated / R.rows }
}

/**
 * Density and coverage (Naeem et al. 2020): density is $\frac{1}{kM} \sum_j \sum_i [\yvec_j \in B(\xvec_i)]$, the
 * number of real $k$-NN balls $B(\xvec_i)$ containing each of the $M$ generated points $\yvec_j$, scaled (it can
 * exceed 1); coverage is the fraction of real points whose $k$-NN ball contains a generated point. More robust to
 * outliers than precision and recall. The real set needs more than $k$ points, or `DomainError` is thrown.
 *
 * @param realFeatures The real samples' features, $n \times d$, one sample per row.
 * @param generatedFeatures The generated samples' features, $M \times d$.
 * @param options `k`, the neighbour that sets each real ball's radius (default 5).
 * @returns `density`, about 1 for matching distributions, and `coverage`, in $[0, 1]$.
 *
 * @example A matching generator, and one collapsed to the centre
 * const real = normal(stream(0), 0, 1, { shape: [150, 2] })
 * print('matching:', densityCoverage(real, normal(stream(1), 0, 1, { shape: [150, 2] })))
 * print('collapsed:', densityCoverage(real, normal(stream(2), 0, 0.2, { shape: [150, 2] })))
 */
export function densityCoverage(
  realFeatures: Rows,
  generatedFeatures: Rows,
  options: { k?: number } = {},
): { density: number; coverage: number } {
  const R = dense(realFeatures, 'densityCoverage')
  const G = dense(generatedFeatures, 'densityCoverage')
  const k = options.k ?? 5
  const rr = kthNeighbourRadii(R, k)
  const D = distances(G, R)
  let contained = 0
  const covered = new Uint8Array(R.rows)
  for (let i = 0; i < G.rows; i++)
    for (let j = 0; j < R.rows; j++)
      if (D[i * R.rows + j] <= rr[j]) {
        contained++
        covered[j] = 1
      }
  return { density: contained / (k * G.rows), coverage: covered.reduce((a, b) => a + b, 0) / R.rows }
}

/**
 * CLIPScore (Hessel et al. 2021; clip-score): $w \max(\cos(\evec_i, \evec_c), 0)$, averaged over matched rows of image
 * and caption embeddings $\evec_i$ and $\evec_c$. Embeddings of different shapes throw `ShapeError`.
 *
 * @param imageEmbeddings The image embeddings, $n \times d$, one image per row.
 * @param textEmbeddings The caption embeddings, $n \times d$: row `i` is the caption of image `i`.
 * @param options `weight`, the scale $w$ (default 2.5, as the paper).
 * @returns The mean score, in $[0, w]$.
 *
 * @example Two images, one with a matching caption
 * const images = [[1, 0, 0], [0, 1, 0]]
 * const captions = [[0.9, 0.1, 0], [0, 0, 1]]
 * print('CLIPScore =', clipScore(images, captions))
 */
export const clipScore = defineMetric(
  {
    module: 'applied/evaluation/generative',
    key: 'clipScore',
    name: 'CLIPScore',
    inputs: 'vectors',
    direction: 'higher',
    range: [0, 2.5],
    notes: ['clip-score'],
  },
  (imageEmbeddings: Rows, textEmbeddings: Rows, options: { weight?: number } = {}): number => {
    const I = dense(imageEmbeddings, 'clipScore')
    const T = dense(textEmbeddings, 'clipScore')
    if (I.rows !== T.rows || I.cols !== T.cols)
      throw new ShapeError('metrics', 'metrics: clipScore: embeddings differ in shape')
    const w = options.weight ?? 2.5
    let s = 0
    for (let i = 0; i < I.rows; i++) {
      let dot = 0
      let a = 0
      let b = 0
      for (let c = 0; c < I.cols; c++) {
        dot += I.data[i * I.cols + c] * T.data[i * I.cols + c]
        a += I.data[i * I.cols + c] ** 2
        b += T.data[i * I.cols + c] ** 2
      }
      s += w * Math.max(divide(dot, Math.sqrt(a * b)), 0)
    }
    return s / I.rows
  },
)
