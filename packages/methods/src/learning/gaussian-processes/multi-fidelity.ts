/**
 * Multi-fidelity Gaussian-process regression: the autoregressive model of Kennedy and O'Hagan (2000) for $M$ ordered
 * fidelities, fitted in the recursive form of co-kriging (Forrester, Sóbester and Keane, 2007; Le Gratiet and
 * Garnier, 2014).
 *
 * The lowest fidelity is a GP, and each higher one is a scaled copy of the one below plus an independent discrepancy:
 *
 * $$ f_0 \sim \mathcal{GP}(0, k_0), \qquad f_t(\xvec) = \rho_{t-1} f_{t-1}(\xvec) + \delta_t(\xvec), \quad
 * \delta_t \sim \mathcal{GP}(0, k_t), $$
 *
 * each fidelity observed with Gaussian noise of variance $\sigma_t^2$. The recursive fit works level by level: $f_0$
 * is fitted to its own data, and level $t$ to the residuals $\yvec_t - \rho_{t-1}\mvec_t$, where $\mvec_t$ is the
 * posterior mean of level $t - 1$ at level $t$'s inputs. Given $k_t$ and $\sigma_t^2$, the scale that maximises the
 * likelihood is the generalised least-squares estimate
 *
 * $$ \hat\rho_{t-1} = \frac{\mvec_t^\top (\Kmat_t + \sigma_t^2\Imat)^{-1} \yvec_t}
 * {\mvec_t^\top (\Kmat_t + \sigma_t^2\Imat)^{-1} \mvec_t}, $$
 *
 * and when the kernels are fitted too, $\hat\rho$ and the kernel's hyperparameters (by `fitGp`) are updated in turn,
 * from two starts (this estimate under the starting kernel, and the least-squares $\mvec_t^\top\yvec_t /
 * \mvec_t^\top\mvec_t$), keeping the one of larger likelihood.
 * The prediction at level $t$ has mean $\rho_{t-1}\mu_{t-1}(\xvec) + \mu_{\delta_t}(\xvec)$ and variance
 * $\rho_{t-1}^2 s_{t-1}^2(\xvec) + s_{\delta_t}^2(\xvec)$. When the designs are nested (each level's inputs among the
 * level below's) and noise-free, this is the exact posterior of the autoregressive model (Le Gratiet and Garnier,
 * 2014); otherwise the level below enters through its posterior mean, an approximation.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { asRows, rbf, type Kernel } from 'aifn-compute/learning/kernels'
import { fitGp, gpPosterior, type GpPosterior } from './regression'

/** Options of `fitMultiFidelityGp`. */
export type MultiFidelityGpOptions = {
  /**
   * The kernels, one per fidelity: $k_0$ of the lowest fidelity, then $k_t$ of each discrepancy $\delta_t$ (default
   * `rbf` with lengthscale 1 and variance 1 for each). They are the starting values when `fitKernels` is true.
   */
  kernels?: readonly Kernel[]
  /** The scales $\rho_0, \dots, \rho_{M-2}$, one per fidelity above the lowest, or `'fit'` (default) to estimate. */
  rho?: readonly number[] | 'fit'
  /**
   * The noise variances $\sigma_t^2$: one number for every fidelity, one per fidelity, or `'fit'` to fit them with the
   * kernels (default $10^{-6}$, nearly noise-free).
   */
  noise?: number | readonly number[] | 'fit'
  /** Fit each kernel's hyperparameters by maximising its level's log marginal likelihood (default true). */
  fitKernels?: boolean
  /** The most L-BFGS steps of each `fitGp` (default 100). */
  maxSteps?: number
  /** The most rounds of updating $\hat\rho$ and the kernel in turn, per level (default 5). */
  rounds?: number
}

/** One level of a fitted multi-fidelity GP. */
export type MultiFidelityGpLevel = {
  /** The kernel: of $f_0$ at level 0, of the discrepancy $\delta_t$ above. */
  kernel: Kernel
  /** The noise variance $\sigma_t^2$. */
  noiseVariance: number
  /** The GP posterior of the level's latent function ($f_0$, or $\delta_t$ given its residuals). */
  posterior: GpPosterior
  /** The log marginal likelihood of the level's targets (of its residuals above level 0). */
  logMarginal: number
}

/** A fitted multi-fidelity GP, from `fitMultiFidelityGp`. */
export interface MultiFidelityGp {
  /** The number of fidelities $M$. */
  readonly fidelities: number
  /** The scales $\rho_0, \dots, \rho_{M-2}$. */
  readonly rho: number[]
  /** The sum of the levels' log marginal likelihoods (exact for nested, noise-free designs). */
  readonly logMarginalLikelihood: number
  /** The levels, lowest fidelity first. */
  readonly levels: MultiFidelityGpLevel[]
  /** The posterior mean and standard deviation of $f_t$ at inputs `x` (`[m, d]` or `[m]`), each `[m]`. */
  predict(x: Tensor, fidelity: number): { mean: Tensor; sd: Tensor }
  /**
   * The posterior covariance of $f_a(\xvec)$ and $f_b(\xvec)$ at each input, `[m]`: for $a \le b$ it is
   * $\rho_a \cdots \rho_{b-1}\,s_a^2(\xvec)$.
   */
  covariance(x: Tensor, a: number, b: number): Tensor
}

/**
 * The rows of an $n \times d$ matrix at the given indices.
 *
 * @param flat The matrix, row-major.
 * @param d The number of columns.
 * @param rows The indices of the rows to keep.
 * @returns The rows, as an `[rows.length, d]` tensor.
 */
function pick(flat: ArrayLike<number>, d: number, rows: readonly number[]): Tensor {
  const out = new Float64Array(rows.length * d)
  rows.forEach((i, k) => {
    for (let j = 0; j < d; j++) out[k * d + j] = flat[i * d + j]
  })
  return fromData(out, [rows.length, d])
}

/**
 * Fits the autoregressive multi-fidelity GP of Kennedy and O'Hagan (2000) to observations at $M$ ordered fidelities,
 * level by level (see the file comment): each level's kernel and noise by `fitGp` (type-II maximum likelihood) when
 * `fitKernels` holds, and each scale $\rho$ by generalised least squares, alternating with the kernel's fit until
 * $\hat\rho$ settles. The prior mean of every level is 0. Throws `DomainError` for fidelities that are not integers
 * from 0, a fidelity with no observation, scales or noise of the wrong count, a negative noise, or `'fit'` noise
 * without `fitKernels`, and `ShapeError` when the inputs, fidelities and targets differ in number.
 *
 * @param x The inputs, $n \times d$ (or $n$ one-dimensional inputs), of every observation at every fidelity.
 * @param fidelity The fidelity of each observation, $n$ integers from 0 (the lowest) to $M - 1$ (the highest).
 * @param y The observed values, $n$.
 * @param options The kernels, the scales, the noise and the fitting, as `MultiFidelityGpOptions`.
 * @returns The fitted model, with `predict` at any fidelity, the scales and the log marginal likelihood.
 *
 * @example Forrester's two fidelities: many cheap points, four expensive ones
 * // Forrester, Sobester and Keane's function and its cheap version 0.5 f(x) + 10 (x - 0.5) + 5
 * const hi = (v) => (6 * v - 2) ** 2 * Math.sin(12 * v - 4)
 * const lo = (v) => 0.5 * hi(v) + 10 * (v - 0.5) + 5
 * const xl = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
 * const xh = [0, 0.4, 0.6, 1]
 * const x = tensor([...xl, ...xh])
 * const y = tensor([...xl.map(lo), ...xh.map(hi)])
 * const gp = fitMultiFidelityGp(x, [...xl.map(() => 0), ...xh.map(() => 1)], y)
 * print('rho =', gp.rho[0].toFixed(3))
 * const p = gp.predict(tensor([0.25, 0.75]), 1)
 * print('high fidelity at 0.25, 0.75:', p.mean, ' truth:', [0.25, 0.75].map(hi))
 *
 * @example Fixed hyperparameters: the scale by generalised least squares
 * const x = tensor([0, 0.5, 1, 0, 1])
 * const y = tensor([1, 2, 3, 2.1, 6.1])
 * const gp = fitMultiFidelityGp(x, [0, 0, 0, 1, 1], y, { fitKernels: false })
 * print('rho =', gp.rho, ' log marginal likelihood =', gp.logMarginalLikelihood.toFixed(4))
 */
export function fitMultiFidelityGp(
  x: Tensor,
  fidelity: Int32Array | readonly number[] | Tensor,
  y: Tensor,
  options: MultiFidelityGpOptions = {},
): MultiFidelityGp {
  const where = 'fitMultiFidelityGp'
  const { rho = 'fit', noise = 1e-6, fitKernels = true, maxSteps = 100, rounds = 5 } = options
  const X = asRows(x) as Tensor
  const [n, d] = X.shape
  const xs = toFlat(X) as ArrayLike<number>
  const z = Array.from((isTensor(fidelity) ? toFlat(fidelity) : fidelity) as ArrayLike<number>)
  const ys = Array.from(toFlat(y) as ArrayLike<number>)
  if (z.length !== n || ys.length !== n)
    throw new ShapeError(where, `${where}: ${n} inputs, ${z.length} fidelities and ${ys.length} targets`)
  if (!z.every((v) => Number.isInteger(v) && v >= 0))
    throw new DomainError(where, `${where}: the fidelities must be integers from 0`)
  const M = n === 0 ? 0 : Math.max(...z) + 1
  const rows = Array.from({ length: M }, (_, t) => z.flatMap((v, i) => (v === t ? [i] : [])))
  if (M === 0 || rows.some((r) => r.length === 0))
    throw new DomainError(where, `${where}: every fidelity from 0 to the highest needs an observation`)
  const kernels = options.kernels ?? Array.from({ length: M }, () => rbf({ lengthscale: 1, variance: 1 }))
  if (kernels.length !== M) throw new DomainError(where, `${where}: ${kernels.length} kernels for ${M} fidelities`)
  if (rho !== 'fit' && rho.length !== M - 1)
    throw new DomainError(where, `${where}: ${rho.length} scales for ${M} fidelities (needs ${M - 1})`)
  if (noise === 'fit' && !fitKernels) throw new DomainError(where, `${where}: noise 'fit' needs fitKernels`)
  const noises =
    noise === 'fit' ? null : typeof noise === 'number' ? Array.from({ length: M }, () => noise) : Array.from(noise)
  if (noises !== null && (noises.length !== M || !noises.every((v) => v >= 0)))
    throw new DomainError(where, `${where}: the noise must be ${M} non-negative variances, or one`)

  const levels: MultiFidelityGpLevel[] = []
  const rhos: number[] = []

  // The posterior mean and variance of f_t at the rows S, from the levels fitted so far.
  const moments = (S: Tensor, t: number): { mean: Float64Array<ArrayBuffer>; variance: Float64Array<ArrayBuffer> } => {
    const p = levels[0].posterior.predict(S)
    const mean = Float64Array.from(toFlat(p.mean) as ArrayLike<number>)
    const variance = Float64Array.from(toFlat(p.variance) as ArrayLike<number>)
    for (let k = 1; k <= t; k++) {
      const q = levels[k].posterior.predict(S)
      const [mq, vq] = [toFlat(q.mean) as ArrayLike<number>, toFlat(q.variance) as ArrayLike<number>]
      const r = rhos[k - 1]
      for (let i = 0; i < mean.length; i++) {
        mean[i] = r * mean[i] + mq[i]
        variance[i] = r * r * variance[i] + vq[i]
      }
    }
    return { mean, variance }
  }

  // Fits a kernel (and the noise, when fitted) to targets r at inputs Xt.
  const fitLevel = (kernel: Kernel, Xt: Tensor, r: Float64Array, start: number) => {
    const target = fromData(r, [r.length])
    if (!fitKernels) return { kernel, noiseVariance: start }
    const fit = fitGp(kernel, Xt, target, {
      noiseVariance: noises === null ? 1e-2 : Math.max(start, 1e-12),
      fitNoise: noises === null,
      maxSteps,
    })
    if (!Number.isFinite(fit.logMarginal)) return { kernel, noiseVariance: start }
    return { kernel: fit.kernel, noiseVariance: noises === null ? fit.noiseVariance : start }
  }

  for (let t = 0; t < M; t++) {
    const Xt = pick(xs, d, rows[t])
    const yt = Float64Array.from(rows[t], (i) => ys[i])
    let kernel = kernels[t]
    let noiseVariance = noises === null ? 1e-2 : noises[t]
    let scale = 0
    let m = new Float64Array(yt.length)
    if (t === 0) ({ kernel, noiseVariance } = fitLevel(kernel, Xt, yt, noiseVariance))
    else {
      m = moments(Xt, t - 1).mean
      const target = fromData(yt, [yt.length])
      const mT = fromData(m, [m.length])
      const gls = (k: Kernel, v: number) => {
        const ay = toFlat(gpPosterior(k, Xt, target, { noiseVariance: v }).alpha) as ArrayLike<number>
        const am = toFlat(gpPosterior(k, Xt, mT, { noiseVariance: v }).alpha) as ArrayLike<number>
        let [num, den] = [0, 0]
        for (let i = 0; i < m.length; i++) [num, den] = [num + m[i] * ay[i], den + m[i] * am[i]]
        return den > 0 ? num / den : 0
      }
      const residual = (r: number) => yt.map((v, i) => v - r * m[i])
      if (rho !== 'fit') {
        scale = rho[t - 1]
        ;({ kernel, noiseVariance } = fitLevel(kernel, Xt, residual(scale), noiseVariance))
      } else {
        scale = gls(kernel, noiseVariance)
        if (fitKernels) {
          // Kernel fits and GLS scales in turn, from the GLS scale of the starting kernel and from the ordinary
          // least-squares scale; the start reaching the larger likelihood is kept.
          let [mm, my] = [0, 0]
          for (let i = 0; i < m.length; i++) [mm, my] = [mm + m[i] * m[i], my + m[i] * yt[i]]
          let best: { kernel: Kernel; noiseVariance: number; scale: number; lml: number } | null = null
          for (const start of mm > 0 ? [scale, my / mm] : [scale]) {
            let [k, v, sc] = [kernel, noiseVariance, start]
            for (let round = 0; round < rounds; round++) {
              ;({ kernel: k, noiseVariance: v } = fitLevel(k, Xt, residual(sc), v))
              const next = gls(k, v)
              const settled = Math.abs(next - sc) <= 1e-8 * (1 + Math.abs(sc))
              sc = next
              if (settled) break
            }
            const lml = gpPosterior(k, Xt, fromData(residual(sc), [m.length]), { noiseVariance: v }).logMarginal.value
            if (best === null || lml > best.lml) best = { kernel: k, noiseVariance: v, scale: sc, lml }
          }
          ;({ kernel, noiseVariance, scale } = best!)
        }
      }
      rhos.push(scale)
    }
    const r = t === 0 ? yt : yt.map((v, i) => v - scale * m[i])
    const posterior = gpPosterior(kernel, Xt, fromData(r, [r.length]), { noiseVariance })
    levels.push({ kernel, noiseVariance, posterior, logMarginal: posterior.logMarginal.value })
  }

  const check = (t: number) => {
    if (!(Number.isInteger(t) && t >= 0 && t < M))
      throw new DomainError(where, `${where}: fidelity ${t} is not one of 0 to ${M - 1}`)
  }
  return {
    fidelities: M,
    rho: rhos,
    logMarginalLikelihood: levels.reduce((a, l) => a + l.logMarginal, 0),
    levels,
    predict: (at, t) => {
      check(t)
      const { mean, variance } = moments(asRows(at) as Tensor, t)
      return {
        mean: fromData(mean, [mean.length]),
        sd: fromData(
          variance.map((v) => Math.sqrt(Math.max(v, 0))),
          [variance.length],
        ),
      }
    },
    covariance: (at, a, b) => {
      check(a)
      check(b)
      const [lo, hi] = a <= b ? [a, b] : [b, a]
      const { variance } = moments(asRows(at) as Tensor, lo)
      let p = 1
      for (let k = lo; k < hi; k++) p *= rhos[k]
      return fromData(
        variance.map((v) => p * v),
        [variance.length],
      )
    },
  }
}
