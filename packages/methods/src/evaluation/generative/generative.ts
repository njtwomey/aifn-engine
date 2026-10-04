/**
 * Metrics of generative models computed from feature vectors (the caller supplies the features, e.g. Inception or CLIP
 * embeddings): the Fréchet distance and FID, the kernel inception distance (with subset averaging), the Inception
 * score, k-NN precision and recall, density and coverage, and CLIPScore.
 */

import { eigh, pairwiseDistances } from 'aifn-compute/numerics/linalg'
import type { Stream } from 'aifn-compute/foundation/random'
import { defineMetric, type Rows } from 'aifn-compute/learning/metrics'
import { denseMatrix as dense, divide, matrix, type Dense } from 'aifn-compute/learning/metrics'
import { child, integers } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Column means and the sample covariance (divisor n − 1, as `numpy.cov`) of the rows of X. */
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

/** Eigenvalues of a symmetric matrix, with round-off negatives set to 0 and counted; a clearly negative one throws. */
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
  distance: number
  /** ‖μ₁ − μ₂‖². */
  meanTerm: number
  /** tr(Σ₁ + Σ₂ − 2(Σ₁Σ₂)^{1/2}). */
  covarianceTerm: number
  /** Eigenvalues at round-off level below 0 that were set to 0 in the matrix square roots. */
  clippedEigenvalues: number
}

/**
 * The Fréchet (Wasserstein-2) distance between N(μ₁, Σ₁) and N(μ₂, Σ₂), squared: ‖μ₁ − μ₂‖² + tr(Σ₁ + Σ₂ −
 * 2(Σ₁Σ₂)^{1/2}) (Dowson and Landau 1982; frechet-inception-distance). tr(Σ₁Σ₂)^{1/2} = Σᵢ √λᵢ with λ the eigenvalues
 * of the symmetric Σ₁^{1/2}Σ₂Σ₁^{1/2}, which is similar to Σ₁Σ₂.
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
 * Fréchet distance of their means and sample covariances (divisor n − 1, as the reference implementation). Biased
 * upwards with finite samples; compare at equal sample sizes.
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

/** Options of the polynomial kernel k(x, y) = (γ xᵀy + c)^degree; defaults γ = 1/d, c = 1, degree 3. */
export type PolynomialKernel = { degree?: number; gamma?: number; coef0?: number }

/** The unbiased MMD² estimate with a polynomial kernel between the rows of X and Y. */
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
 * squared MMD between real and generated features with the cubic kernel (xᵀy/d + 1)³. Unbiased at any sample size,
 * so it can be slightly negative; do not clip it.
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
 * KID averaged over `subsets` random subsets (default 10) of `subsetSize` rows (default min(1000, n, m)) drawn without
 * replacement with the stream, as practice does for large sets, with the standard deviation over subsets.
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
 * The Inception score exp(Eₓ KL(p(y|x) ‖ p(y))) = e^{I(x; y)} from an n × K matrix of class probabilities of generated
 * images (Salimans et al. 2016; inception-score), in [1, K]. With `splits` > 1, the mean and standard deviation over
 * equal consecutive splits, as the reference implementation reports.
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

/** The Inception score of all rows as one split (see `inceptionScoreSplits`). */
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

/** Euclidean distance from each row of A to each row of B (linalg's `pairwiseDistances`), row-major [A.rows, B.rows]. */
function distances(A: Dense, B: Dense): Float64Array {
  const D = pairwiseDistances(fromData(A.data, [A.rows, A.cols]), fromData(B.data, [B.rows, B.cols]))
  return Float64Array.from(toFlat(D))
}

/** Each row's distance to its k-th nearest neighbour within its own set (itself excluded). */
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
 * precision-and-recall-for-generative-models): each set's manifold is the union of balls reaching each point's k-th
 * nearest neighbour (default k = 3); precision is the fraction of generated points inside the real manifold, recall
 * the fraction of real points inside the generated manifold.
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
 * Density and coverage (Naeem et al. 2020): density is (1/(kM)) Σ over generated points of the number of real k-NN
 * balls containing it (can exceed 1); coverage is the fraction of real points whose k-NN ball contains a generated
 * point. Default k = 5. More robust to outliers than precision and recall.
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
 * CLIPScore (Hessel et al. 2021; clip-score): w · max(cos(eᵢ, e_c), 0) with w = 2.5, averaged over matched rows of
 * image and caption embeddings.
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
