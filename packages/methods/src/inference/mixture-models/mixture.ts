/**
 * Variational Bayesian Gaussian mixture by coordinate ascent (CAVI), and its predictive density (Bishop, 2006, §10.2;
 * Attias, 2000).
 *
 * The model is $\pivec \sim \Dir(\alpha_0)$, $\Lambdamat_k \sim \Wishart(\Wmat_0, \nu_0)$,
 * $\muvec_k \mid \Lambdamat_k \sim \Gauss(\mvec_0, (\beta_0\Lambdamat_k)^{-1})$, $z_n \sim \Cat(\pivec)$ and
 * $\xvec_n \mid z_n = k \sim \Gauss(\muvec_k, \Lambdamat_k^{-1})$. The variational family is
 * $q(\Zmat)\,q(\pivec)\prod_k q(\muvec_k, \Lambdamat_k)$ with $q(\pivec) = \Dir(\alphavec)$,
 * $q(\muvec_k, \Lambdamat_k) = \Gauss(\mvec_k, (\beta_k\Lambdamat_k)^{-1})\,\Wishart(\Lambdamat_k \mid \Wmat_k, \nu_k)$
 * and $q(\Zmat)$ given by the responsibilities $r_{nk}$. With a small $\alpha_0$, components the data do not need are
 * emptied (their $N_k \to 0$) and $\expect[\pi_k] \to 0$. Data and parameters are held as row-major float64 arrays.
 */

import { inverse, logDet } from 'aifn-compute/numerics/linalg'
import type { Status } from 'aifn-compute/foundation/contracts'
import { integers } from 'aifn-compute/foundation/random'
import { digamma, logGamma } from 'aifn-compute/numerics/special'
import {
  fromData,
  isTensor,
  tensor,
  toFlat,
  type Matrix,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { dense } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64

const LOG_2PI = Math.log(2 * Math.PI)
/**
 * The digamma function $\psi(x)$, as a number.
 *
 * @param x The argument, a positive number.
 * @returns $\psi(x)$.
 */
const dg = (x: number) => digamma(x) as number
/**
 * The log-gamma function $\ln\Gamma(x)$, as a number.
 *
 * @param x The argument, a positive number.
 * @returns $\ln\Gamma(x)$.
 */
const lg = (x: number) => logGamma(x) as number

/**
 * Prior hyperparameters of `caviGaussianMixture`, in Bishop's notation. Each one left out takes a data-scaled default.
 */
export type MixturePrior = {
  /** $\alpha_0$, the concentration of the symmetric Dirichlet prior on the weights (default $1/K$). */
  alpha0?: number
  /** $\beta_0$, the prior precision of each mean as a multiple of $\Lambdamat_k$ (default 1). */
  beta0?: number
  /** $\mvec_0$, the prior mean of every component mean, $D$ values (default the data mean). */
  mean0?: ArrayLike<number>
  /**
   * $\Wmat_0$, the Wishart scale matrix, $D \times D$ as a tensor or as rows (default $(\nu_0\hat{\Sigmamat})^{-1}$,
   * $\hat{\Sigmamat}$ the data covariance, so that $\expect[\Lambdamat] = \hat{\Sigmamat}^{-1}$).
   */
  W0?: Tensor | readonly (readonly number[])[]
  /** $\nu_0$, the Wishart degrees of freedom (default $D$). */
  nu0?: number
}

/** The state of `caviGaussianMixture`. */
export type MixtureState = Status & {
  /** Coordinate sweeps done. */
  t: number
  /** Responsibilities $r_{nk}$, $N \times K$; each row sums to 1. */
  responsibilities: Matrix
  /** The effective counts $N_k = \sum_n r_{nk}$, $K$ values. */
  counts: Vector
  /** The parameters $\alpha_k = \alpha_0 + N_k$ of $q(\pivec) = \Dir(\alphavec)$, $K$ values. */
  alpha: Vector
  /** The expected weights $\expect[\pi_k] = \alpha_k / \sum_j \alpha_j$, $K$ values. */
  weights: Vector
  /** The means $\mvec_k$ of $q(\muvec_k, \Lambdamat_k)$, $K \times D$. */
  means: Matrix
  /** $\beta_k = \beta_0 + N_k$, the precision of $q(\muvec_k \mid \Lambdamat_k)$ as a multiple of $\Lambdamat_k$. */
  beta: Vector
  /** The Wishart scale matrices $\Wmat_k$ of $q(\Lambdamat_k)$, $K \times D \times D$. */
  W: Tensor
  /** The Wishart degrees of freedom $\nu_k = \nu_0 + N_k$, $K$ values. */
  nu: Vector
  /**
   * $\expect[\Lambdamat_k]^{-1} = (\nu_k\Wmat_k)^{-1}$, $K \times D \times D$: a point summary of each component's
   * covariance.
   */
  covariances: Tensor
  /** The evidence lower bound at this state. */
  elbo: number
  /** The change in the ELBO since the previous state (NaN at step 0). */
  elboChange: number
  /** True when the ELBO changed by less than `tolerance` times $\max(1, \lvert \text{ELBO} \rvert)$. */
  converged: boolean
  /** True when the ELBO is not finite. */
  diverged: boolean
}

/**
 * The variational parameters after an M-like step, row-major: `alpha`, `beta`, `nu` and `counts` ($K$ values each:
 * $\alpha_k$, $\beta_k$, $\nu_k$, $N_k$), `m` ($K \times D$, the means $\mvec_k$) and `W` ($K \times D \times D$, the
 * Wishart scales $\Wmat_k$).
 */
type Params = { alpha: F64; beta: F64; m: F64; W: F64; nu: F64; counts: F64 }

/**
 * The data as a row-major float64 copy with its shape: an $N \times D$ matrix, or $N$ numbers taken as $D = 1$.
 *
 * @param x The data: a tensor ($N \times D$, or a vector of $N$ values), an array of rows, or an array of numbers.
 *   Not modified.
 * @returns `data`, the $N D$ values row by row, with `N` and `D`. Throws `ShapeError` for a tensor of rank above 2.
 */
function toMatrix(x: Tensor | readonly (readonly number[])[] | ArrayLike<number>): { data: F64; N: number; D: number } {
  if (!isTensor(x) && typeof (x as ArrayLike<unknown>)[0] === 'number') {
    const data = Float64Array.from(x as ArrayLike<number>)
    return { data, N: data.length, D: 1 }
  }
  const t = isTensor(x) ? x : tensor(x as number[][])
  if (t.shape.length === 1) return { data: Float64Array.from(toFlat(t)), N: t.shape[0], D: 1 }
  if (t.shape.length !== 2)
    throw new ShapeError('caviGaussianMixture', 'caviGaussianMixture: data must be N×D (or a vector for D = 1)')
  return { data: Float64Array.from(toFlat(t)), N: t.shape[0], D: t.shape[1] }
}

/**
 * A $D \times D$ tensor viewing a row-major array.
 *
 * @param a The $D^2$ entries, row by row.
 * @param D The number of rows (and columns).
 * @returns The $D \times D$ matrix.
 */
const mat = (a: F64, D: number) => fromData(a, [D, D])
/**
 * The inverse of a $D \times D$ matrix, by `inverse`.
 *
 * @param a The matrix as $D^2$ row-major values; not modified.
 * @param D The number of rows (and columns).
 * @returns The inverse as a new row-major array of $D^2$ values.
 */
const invD = (a: F64, D: number): F64 => Float64Array.from(toFlat(inverse(mat(a, D))))
/**
 * The log-determinant $\ln\lvert \Amat \rvert$ of a $D \times D$ matrix, by `logDet`.
 *
 * @param a The matrix $\Amat$ as $D^2$ row-major values; not modified.
 * @param D The number of rows (and columns).
 * @returns $\ln\lvert \Amat \rvert$.
 */
const logDetD = (a: F64, D: number): number => logDet(mat(a, D)) as number

/**
 * The quadratic form $(\xvec - \mvec)^\top \Amat (\xvec - \mvec)$, on vectors and a matrix read at offsets into larger
 * row-major arrays.
 *
 * @param x The array holding $\xvec$ at entries `xo` to `xo + D - 1`.
 * @param m The array holding $\mvec$ at entries `mo` to `mo + D - 1`.
 * @param A The array holding $\Amat$, row-major $D \times D$, at entries `Ao` to `Ao + D * D - 1`.
 * @param D The dimension of the vectors.
 * @param xo The offset of $\xvec$ in `x`.
 * @param mo The offset of $\mvec$ in `m`.
 * @param Ao The offset of $\Amat$ in `A`.
 * @returns The value of the form.
 */
function quad(
  x: F64 | ArrayLike<number>,
  m: ArrayLike<number>,
  A: ArrayLike<number>,
  D: number,
  xo = 0,
  mo = 0,
  Ao = 0,
) {
  let s = 0
  for (let i = 0; i < D; i++)
    for (let j = 0; j < D; j++) s += (x[xo + i] - m[mo + i]) * A[Ao + i * D + j] * (x[xo + j] - m[mo + j])
  return s
}

/**
 * $\ln B(\Wmat, \nu)$, the log normaliser of the Wishart density (Bishop, 2006, eq. B.79).
 *
 * @param logDetW $\ln\lvert \Wmat \rvert$, the log-determinant of the scale matrix.
 * @param nu The degrees of freedom $\nu$.
 * @param D The dimension $D$ of $\Wmat$.
 * @returns $\ln B(\Wmat, \nu)$.
 */
function logWishartB(logDetW: number, nu: number, D: number) {
  let s = 0
  for (let i = 1; i <= D; i++) s += lg((nu + 1 - i) / 2)
  return -(nu / 2) * logDetW - ((nu * D) / 2) * Math.LN2 - ((D * (D - 1)) / 4) * Math.log(Math.PI) - s
}

/**
 * CAVI for the Bayesian Gaussian mixture (Bishop, 2006, §10.2.1), as a step-through algorithm. Each step is an M-like
 * update of $q(\pivec)$ and $q(\muvec_k, \Lambdamat_k)$ from the responsibilities (eqs. 10.51 to 10.63), then an
 * E-like update
 * $r_{nk} \propto \exp(\expect[\ln \pi_k] + \tfrac{1}{2}\expect[\ln\lvert \Lambdamat_k \rvert] - D/(2\beta_k) -
 * \tfrac{\nu_k}{2}(\xvec_n - \mvec_k)^\top \Wmat_k (\xvec_n - \mvec_k))$ (eqs. 10.46 to 10.67). The ELBO
 * (eqs. 10.70 to 10.77) never decreases; the run stops when it changes by less than `tolerance` relative to its size
 * (`converged`), or when it is not finite (`diverged`).
 *
 * `init` draws $\min(K, N)$ distinct data points from its stream as centres, assigns each point to its nearest
 * centre, and makes one M-like and one E-like update from those hard assignments: that is step 0. The algorithm takes
 * no start value (`run(alg, undefined, n)`). CAVI finds a local optimum: from an unlucky start, one component can
 * end up covering two clusters, so the run's stream matters.
 *
 * @param x The data: an $N \times D$ tensor or array of rows, or $N$ numbers (or a vector) for $D = 1$. Copied, not
 *   modified. Throws `ShapeError` for a tensor of rank above 2.
 * @param K The number of components. Surplus components are emptied when $\alpha_0$ is small.
 * @param prior The prior hyperparameters; each one left out takes its data-scaled default: $\alpha_0 = 1/K$,
 *   $\beta_0 = 1$, $\mvec_0$ the data mean, $\nu_0 = D$ and $\Wmat_0 = (\nu_0\hat{\Sigmamat})^{-1}$, so that
 *   $\expect[\Lambdamat] = \hat{\Sigmamat}^{-1}$ ($\hat{\Sigmamat}$ the data covariance).
 * @param options The stopping rule.
 * @param options.tolerance The relative change in the ELBO below which a state is `converged` (default $10^{-8}$).
 * @returns The algorithm, whose states (`MixtureState`) hold the variational parameters, the responsibilities, the
 *   expected weights, the covariance summaries and the ELBO.
 *
 * @example Two blobs in the plane: the means and weights recovered
 * const s = stream(0)
 * const x = concat([normals(s, [20, 2], 0, 0.5), normals(s, [20, 2], 4, 0.5)])
 * const final = run(caviGaussianMixture(x, 2), undefined, 100, { stream: stream(0) })
 * print('means', final.means)
 * print('weights', final.weights)
 * print('sweeps', final.t, 'converged', final.converged)
 *
 * @example A surplus component is emptied
 * const s = stream(2)
 * const x = concat([normals(s, [30], -3, 0.5), normals(s, [30], 3, 0.5)])
 * const final = run(caviGaussianMixture(x, 4, { alpha0: 0.01 }), undefined, 300, { stream: stream(3) })
 * print('counts N_k', final.counts)
 * print('weights', final.weights)
 * print('means', final.means)
 */
export function caviGaussianMixture(
  x: Tensor | readonly (readonly number[])[] | ArrayLike<number>,
  K: number,
  prior: MixturePrior = {},
  options: { tolerance?: number } = {},
): Algorithm<void, MixtureState> {
  const { data: X, N, D } = toMatrix(x)
  const tolerance = options.tolerance ?? 1e-8
  const dataMean = new Float64Array(D)
  for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) dataMean[i] += X[n * D + i] / N
  const cov = new Float64Array(D * D)
  for (let n = 0; n < N; n++)
    for (let i = 0; i < D; i++)
      for (let j = 0; j < D; j++) cov[i * D + j] += ((X[n * D + i] - dataMean[i]) * (X[n * D + j] - dataMean[j])) / N
  const alpha0 = prior.alpha0 ?? 1 / K
  const beta0 = prior.beta0 ?? 1
  const nu0 = prior.nu0 ?? D
  const m0 = prior.mean0 ? Float64Array.from(prior.mean0) : dataMean
  const W0 = prior.W0
    ? Float64Array.from(toFlat(isTensor(prior.W0) ? prior.W0 : tensor(prior.W0 as number[][])))
    : invD(
        cov.map((v) => v * nu0),
        D,
      )
  const W0inv = invD(W0, D)
  const logDetW0 = logDetD(W0, D)

  /**
   * $q(\pivec)$ and $q(\muvec_k, \Lambdamat_k)$ from responsibilities (Bishop eqs. 10.51 to 10.53, 10.58, 10.60 to
   * 10.63).
   */
  function mStep(r: F64): Params {
    const counts = new Float64Array(K)
    const xbar = new Float64Array(K * D)
    const S = new Float64Array(K * D * D)
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++) {
        counts[k] += r[n * K + k]
        for (let i = 0; i < D; i++) xbar[k * D + i] += r[n * K + k] * X[n * D + i]
      }
    for (let k = 0; k < K; k++) for (let i = 0; i < D; i++) xbar[k * D + i] /= counts[k] + 1e-300
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++)
        for (let i = 0; i < D; i++)
          for (let j = 0; j < D; j++)
            S[(k * D + i) * D + j] += r[n * K + k] * (X[n * D + i] - xbar[k * D + i]) * (X[n * D + j] - xbar[k * D + j])
    const alpha = counts.map((c) => alpha0 + c)
    const beta = counts.map((c) => beta0 + c)
    const nu = counts.map((c) => nu0 + c)
    const m = new Float64Array(K * D)
    const W = new Float64Array(K * D * D)
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < D; i++) m[k * D + i] = (beta0 * m0[i] + counts[k] * xbar[k * D + i]) / beta[k]
      const Winv = new Float64Array(D * D)
      const c = (beta0 * counts[k]) / (beta0 + counts[k])
      for (let i = 0; i < D; i++)
        for (let j = 0; j < D; j++)
          Winv[i * D + j] =
            W0inv[i * D + j] + S[(k * D + i) * D + j] + c * (xbar[k * D + i] - m0[i]) * (xbar[k * D + j] - m0[j])
      W.set(invD(Winv, D), k * D * D)
    }
    return { alpha, beta, m, W, nu, counts }
  }

  const expectations = (p: Params) => {
    const alphaHat = p.alpha.reduce((a, b) => a + b, 0)
    const logPi = p.alpha.map((a) => dg(a) - dg(alphaHat))
    const logDetW = Float64Array.from({ length: K }, (_, k) => logDetD(p.W.slice(k * D * D, (k + 1) * D * D), D))
    const logLambda = Float64Array.from({ length: K }, (_, k) => {
      let s = D * Math.LN2 + logDetW[k]
      for (let i = 1; i <= D; i++) s += dg((p.nu[k] + 1 - i) / 2)
      return s
    })
    return { alphaHat, logPi, logDetW, logLambda }
  }

  /** Responsibilities (Bishop eqs. 10.46, 10.49, 10.64–10.67). */
  function eStep(p: Params): F64 {
    const { logPi, logLambda } = expectations(p)
    const r = new Float64Array(N * K)
    for (let n = 0; n < N; n++) {
      let max = -Infinity
      for (let k = 0; k < K; k++) {
        const v =
          logPi[k] +
          0.5 * logLambda[k] -
          D / (2 * p.beta[k]) -
          (p.nu[k] / 2) * quad(X, p.m, p.W, D, n * D, k * D, k * D * D)
        r[n * K + k] = v
        if (v > max) max = v
      }
      let s = 0
      for (let k = 0; k < K; k++) s += r[n * K + k] = Math.exp(r[n * K + k] - max)
      for (let k = 0; k < K; k++) r[n * K + k] /= s
    }
    return r
  }

  /**
   * The ELBO (Bishop eqs. 10.70 to 10.77), with the statistics $N_k$, $\bar{\xvec}_k$, $\Smat_k$ recomputed from
   * `r`.
   */
  function elboOf(p: Params, r: F64): number {
    const { alphaHat, logPi, logDetW, logLambda } = expectations(p)
    const counts = new Float64Array(K)
    const xbar = new Float64Array(K * D)
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++) {
        counts[k] += r[n * K + k]
        for (let i = 0; i < D; i++) xbar[k * D + i] += r[n * K + k] * X[n * D + i]
      }
    for (let k = 0; k < K; k++) for (let i = 0; i < D; i++) xbar[k * D + i] /= counts[k] + 1e-300
    let L = 0
    for (let k = 0; k < K; k++) {
      const Wk = p.W.subarray(k * D * D, (k + 1) * D * D)
      // Tr(Sₖ Wₖ) Nₖ = Σₙ rₙₖ (xₙ − x̄ₖ)ᵀ Wₖ (xₙ − x̄ₖ).
      let trSW = 0
      for (let n = 0; n < N; n++) if (r[n * K + k] > 0) trSW += r[n * K + k] * quad(X, xbar, Wk, D, n * D, k * D)
      const dev = quad(xbar, p.m, Wk, D, k * D, k * D)
      // 10.71
      L += 0.5 * (counts[k] * (logLambda[k] - D / p.beta[k] - p.nu[k] * dev - D * LOG_2PI) - p.nu[k] * trSW)
      // 10.72
      for (let n = 0; n < N; n++) L += r[n * K + k] * logPi[k]
      // 10.74
      let trW0W = 0
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) trW0W += W0inv[i * D + j] * Wk[j * D + i]
      L +=
        0.5 *
          (D * Math.log(beta0 / (2 * Math.PI)) +
            logLambda[k] -
            (D * beta0) / p.beta[k] -
            beta0 * p.nu[k] * quad(p.m, m0, Wk, D, k * D, 0)) +
        logWishartB(logDetW0, nu0, D) +
        ((nu0 - D - 1) / 2) * logLambda[k] -
        0.5 * p.nu[k] * trW0W
      // −10.75
      for (let n = 0; n < N; n++) {
        const v = r[n * K + k]
        if (v > 0) L -= v * Math.log(v)
      }
      // −10.77, with H[q(Λₖ)] = −ln B(Wₖ, νₖ) − (νₖ − D − 1)/2 E[ln|Λₖ|] + νₖD/2.
      const entropyW = -logWishartB(logDetW[k], p.nu[k], D) - ((p.nu[k] - D - 1) / 2) * logLambda[k] + (p.nu[k] * D) / 2
      L -= 0.5 * logLambda[k] + (D / 2) * Math.log(p.beta[k] / (2 * Math.PI)) - D / 2 - entropyW
    }
    // 10.73 and −10.76: ln C(α) = ln Γ(Σα) − Σ ln Γ(αₖ).
    const logC0 = lg(K * alpha0) - K * lg(alpha0)
    let logC = lg(alphaHat)
    for (let k = 0; k < K; k++) {
      logC -= lg(p.alpha[k])
      L += (alpha0 - 1) * logPi[k] - (p.alpha[k] - 1) * logPi[k]
    }
    return L + logC0 - logC
  }

  const toState = (t: number, p: Params, r: F64, previous: number): MixtureState => {
    const value = elboOf(p, r)
    const alphaHat = p.alpha.reduce((a, b) => a + b, 0)
    const covariances = new Float64Array(K * D * D)
    for (let k = 0; k < K; k++)
      covariances.set(
        invD(
          p.W.slice(k * D * D, (k + 1) * D * D).map((v) => v * p.nu[k]),
          D,
        ),
        k * D * D,
      )
    const change = value - previous
    return {
      t,
      responsibilities: fromData(r, [N, K]),
      counts: fromData(p.counts, [K]),
      alpha: fromData(p.alpha, [K]),
      weights: fromData(
        p.alpha.map((a) => a / alphaHat),
        [K],
      ),
      means: fromData(p.m, [K, D]),
      beta: fromData(p.beta, [K]),
      W: fromData(p.W, [K, D, D]),
      nu: fromData(p.nu, [K]),
      covariances: fromData(covariances, [K, D, D]),
      elbo: value,
      elboChange: change,
      converged: Math.abs(change) < tolerance * Math.max(1, Math.abs(value)),
      diverged: !Number.isFinite(value),
    }
  }

  return {
    name: 'cavi-gaussian-mixture',
    init: (_start, st) => {
      // K distinct data points as centres (partial Fisher–Yates), then hard nearest-centre responsibilities.
      const order = Int32Array.from({ length: N }, (_, i) => i)
      for (let k = 0; k < Math.min(K, N); k++) {
        const j = k + integers(st, N - k)
        ;[order[k], order[j]] = [order[j], order[k]]
      }
      const r0 = new Float64Array(N * K)
      for (let n = 0; n < N; n++) {
        let best = 0
        let bestD = Infinity
        for (let k = 0; k < K; k++) {
          const c = order[k % N]
          let dist = 0
          for (let i = 0; i < D; i++) dist += (X[n * D + i] - X[c * D + i]) ** 2
          if (dist < bestD) [best, bestD] = [k, dist]
        }
        r0[n * K + best] = 1
      }
      const p = mStep(r0)
      return toState(0, p, eStep(p), NaN)
    },
    step: (s) => {
      const p = mStep(Float64Array.from(toFlat(s.responsibilities)))
      return toState(s.t + 1, p, eStep(p), s.elbo)
    },
  }
}

/**
 * The predictive density $p(\hat{\xvec} \mid \Xmat)$ of the variational mixture (Bishop, 2006, eqs. 10.81 and 10.82):
 * a mixture of Student t densities,
 * $\frac{1}{\hat{\alpha}} \sum_k \alpha_k \operatorname{St}(\hat{\xvec} \mid \mvec_k, \Lmat_k, \nu_k + 1 - D)$ with
 * $\hat{\alpha} = \sum_k \alpha_k$ and precision $\Lmat_k = \frac{(\nu_k + 1 - D)\beta_k}{1 + \beta_k}\Wmat_k$.
 *
 * @param state A state of `caviGaussianMixture`; its `alpha`, `beta`, `nu`, `means` and `W` are read.
 * @param points The $M$ points to evaluate at: an $M \times D$ tensor or array of rows, read row by row; for $D = 1$,
 *   any $M$ numbers.
 * @returns `density`, the $M$ predictive densities, and `components` ($K \times M$), each component's weighted term
 *   $\frac{\alpha_k}{\hat{\alpha}}\operatorname{St}(\cdot)$, whose columns sum to `density`.
 *
 * @example The predictive density of a fitted mixture of two blobs on a line
 * const s = stream(4)
 * const x = concat([normals(s, [30], -2, 0.5), normals(s, [30], 2, 0.5)])
 * const final = run(caviGaussianMixture(x, 2), undefined, 100, { stream: stream(5) })
 * const { density, components } = mixturePredictiveDensity(final, [-2, 0, 2])
 * print('density at -2, 0, 2', density)
 * print('per component', components)
 */
export function mixturePredictiveDensity(
  state: MixtureState,
  points: Tensor | readonly (readonly number[])[] | ArrayLike<number>,
): { density: Vector; components: Matrix } {
  const K = state.alpha.shape[0]
  const D = state.means.shape[1]
  const P = isTensor(points)
    ? Float64Array.from(toFlat(points))
    : typeof (points as ArrayLike<unknown>)[0] === 'number'
      ? Float64Array.from(points as ArrayLike<number>)
      : Float64Array.from((points as readonly (readonly number[])[]).flat())
  const M = P.length / D
  const alpha = toFlat(state.alpha)
  const alphaHat = alpha.reduce((a, b) => a + b, 0)
  const beta = toFlat(state.beta)
  const nu = toFlat(state.nu)
  const m = toFlat(state.means)
  const W = Float64Array.from(toFlat(state.W))
  const comps = new Float64Array(K * M)
  const density = new Float64Array(M)
  for (let k = 0; k < K; k++) {
    const dof = nu[k] + 1 - D
    const c = (dof * beta[k]) / (1 + beta[k])
    const Lk = W.subarray(k * D * D, (k + 1) * D * D).map((v) => v * c)
    const logDetL = logDetD(Float64Array.from(Lk), D)
    const logNorm = lg((dof + D) / 2) - lg(dof / 2) + 0.5 * logDetL - (D / 2) * Math.log(dof * Math.PI)
    for (let j = 0; j < M; j++) {
      const q = quad(P, m, Lk, D, j * D, k * D)
      const v = (alpha[k] / alphaHat) * Math.exp(logNorm - ((dof + D) / 2) * Math.log1p(q / dof))
      comps[k * M + j] = v
      density[j] += v
    }
  }
  return { density: fromData(density, [M]), components: fromData(comps, [K, M]) }
}
