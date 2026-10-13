/**
 * Acquisitions of a batch of points evaluated together: the multi-point expected improvement (q-EI; Schonlau, 1997;
 * Ginsbourger, Le Riche and Carraro, 2010) by Monte Carlo, and local penalisation (González, Dai, Hennig and Lawrence,
 * 2016), which shrinks an acquisition around the points already chosen.
 *
 * The q-EI of $q$ points with joint posterior $\yvec \sim \Gauss(\muvec, \Sigmamat)$ is the expected improvement
 * of the best of them over the incumbent $f^+$,
 *
 * $$ \mathrm{EI}_q = \expect\bigl[\max(f^+ - \min_j y_j, 0)\bigr], $$
 *
 * for minimisation. Local penalisation multiplies an acquisition $\alpha$ by one factor per pending point $\xvec_j$,
 * the probability that $\xvec$ lies outside the ball around $\xvec_j$ that a Lipschitz constant $L$ shows cannot hold
 * the minimum:
 *
 * $$ \tilde\alpha(\xvec) = g(\alpha(\xvec)) \prod_j \Phi\!\left(
 * \frac{L\lVert \xvec - \xvec_j \rVert - \mu(\xvec_j) + M}{\sigma(\xvec_j)}\right), $$
 *
 * with $M$ the minimum value, $\mu$ and $\sigma$ the posterior mean and standard deviation, and $g$ the identity for a
 * non-negative acquisition or the soft-plus $\ln(1 + e^{\alpha})$ otherwise.
 */

import { normal, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { expectedImprovement } from './acquisitions'

/** Options of `multiPointExpectedImprovement`. */
export type MultiPointOptions = {
  /** The number of Monte Carlo draws of the $q$ values (default 10000). */
  samples?: number
  /** The stream the draws come from (needed when $q > 1$). */
  stream?: Stream
  /** Minimise the objective (default true); false maximises it. */
  minimise?: boolean
}

/** The Monte Carlo estimate of the multi-point expected improvement. */
export type MultiPointEstimate = {
  /** The estimate of $\mathrm{EI}_q$: the mean improvement over the draws. */
  value: number
  /** Its standard error, the standard deviation of the improvements over $\sqrt{S}$ (0 when exact, $q = 1$). */
  se: number
}

/**
 * The lower Cholesky factor of a small symmetric positive semi-definite matrix, with jitter $10^{-12}$, $10^{-10}$,
 * and so on up to $10^{-2}$ times the mean diagonal added until it factors (so a singular covariance, as of a point
 * chosen twice, factors); the zero matrix gives a zero factor. Throws `DomainError` when no jitter is enough.
 *
 * @param a The matrix, $q \times q$ row-major.
 * @param q The order $q$.
 * @param where The caller's name, for error messages.
 * @returns The factor $\Lmat$, row-major.
 */
function choleskyJittered(a: Float64Array, q: number, where: string): Float64Array {
  let mean = 0
  for (let i = 0; i < q; i++) mean += a[i * q + i] / q
  if (mean === 0 && a.every((v) => v === 0)) return new Float64Array(q * q)
  for (const jitter of [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2].map((e) => e * mean)) {
    const L = new Float64Array(q * q)
    let ok = true
    for (let j = 0; j < q && ok; j++) {
      let s = a[j * q + j] + jitter
      for (let k = 0; k < j; k++) s -= L[j * q + k] ** 2
      if (!(s > 0)) ok = false
      else {
        L[j * q + j] = Math.sqrt(s)
        for (let i = j + 1; i < q; i++) {
          let t = a[i * q + j]
          for (let k = 0; k < j; k++) t -= L[i * q + k] * L[j * q + k]
          L[i * q + j] = t / L[j * q + j]
        }
      }
    }
    if (ok) return L
  }
  throw new DomainError(where, `${where}: the covariance is not positive semi-definite`)
}

/**
 * The multi-point expected improvement $\mathrm{EI}_q = \expect[\max(f^+ - \min_j y_j, 0)]$ of $q$ points whose
 * values are jointly Gaussian, $\yvec \sim \Gauss(\muvec, \Sigmamat)$ (Schonlau, 1997; Ginsbourger, Le Riche and
 * Carraro, 2010), estimated from `samples` draws $\muvec + \Lmat\zvec$ with $\Lmat\Lmat^\top = \Sigmamat$, with
 * its standard error. When maximising it is $\expect[\max(\max_j y_j - f^+, 0)]$. For $q = 1$ it is the closed-form
 * expected improvement, exactly (standard error 0). Unlike $q$ separate expected improvements it rewards points whose
 * values are not strongly correlated, since two copies of one point improve no more than one. Throws `ShapeError` when
 * `cov` is not $q \times q$, and `DomainError` for a covariance that is not positive semi-definite, fewer than two
 * samples, or no stream when $q > 1$.
 *
 * @param mean The posterior means $\muvec$ of the $q$ points, `[q]`.
 * @param cov Their posterior covariance $\Sigmamat$, `[q, q]`.
 * @param best The incumbent $f^+$.
 * @param options The number of draws, the stream and the direction, as `MultiPointOptions`.
 * @returns The estimate and its standard error.
 *
 * @example Two correlated points improve less together than two independent ones
 * const mean = tensor([0, 0])
 * const near = multiPointExpectedImprovement(mean, tensor([[1, 0.95], [0.95, 1]]), 0, { stream: stream(0) })
 * const far = multiPointExpectedImprovement(mean, tensor([[1, 0], [0, 1]]), 0, { stream: stream(0) })
 * print('correlated:', near.value.toFixed(4), '+/-', near.se.toFixed(4))
 * print('independent:', far.value.toFixed(4), '+/-', far.se.toFixed(4))
 * print('one point (closed form):', multiPointExpectedImprovement(tensor([0]), tensor([[1]]), 0).value.toFixed(4))
 */
export function multiPointExpectedImprovement(
  mean: Tensor,
  cov: Tensor,
  best: number,
  options: MultiPointOptions = {},
): MultiPointEstimate {
  const where = 'multiPointExpectedImprovement'
  const { samples = 10000, stream, minimise = true } = options
  const mu = Float64Array.from(toFlat(mean))
  const q = mu.length
  const c = Float64Array.from(toFlat(cov))
  if (q === 0 || c.length !== q * q) throw new ShapeError(where, `${where}: the covariance must be [${q}, ${q}]`)
  if (q === 1) {
    if (!(c[0] >= 0)) throw new DomainError(where, `${where}: the covariance is not positive semi-definite`)
    return { value: expectedImprovement(mu[0], Math.sqrt(c[0]), best, { minimise }), se: 0 }
  }
  if (!(Number.isInteger(samples) && samples >= 2)) throw new DomainError(where, `${where}: needs at least 2 samples`)
  if (!stream) throw new DomainError(where, `${where}: needs a stream when q > 1`)
  const L = choleskyJittered(c, q, where)
  const z = toFlat(normal(stream, 0, 1, { shape: [samples, q] }) as Tensor)
  let sum = 0
  let sum2 = 0
  for (let s = 0; s < samples; s++) {
    let top = minimise ? Infinity : -Infinity
    for (let i = 0; i < q; i++) {
      let y = mu[i]
      for (let k = 0; k <= i; k++) y += L[i * q + k] * z[s * q + k]
      top = minimise ? Math.min(top, y) : Math.max(top, y)
    }
    const gain = Math.max(minimise ? best - top : top - best, 0)
    sum += gain
    sum2 += gain * gain
  }
  const value = sum / samples
  const variance = Math.max(sum2 / samples - value * value, 0) * (samples / (samples - 1))
  return { value, se: Math.sqrt(variance / samples) }
}

/**
 * The Lipschitz constant of a function known on a set of points: the largest slope between two of them,
 * $\max_{i \ne j} \lvert \mu_i - \mu_j \rvert / \lVert \xvec_i - \xvec_j \rVert$. On a fine grid it approaches the
 * largest gradient norm of the function, which González et al. (2016) use for $L$.
 *
 * @param values The function's values $\mu_i$, $m$ of them.
 * @param points The points, $m \times d$ row-major.
 * @param d The dimension $d$.
 * @returns The constant (0 for a constant function or a single point).
 */
export function gridLipschitz(values: ArrayLike<number>, points: ArrayLike<number>, d: number): number {
  const m = values.length
  let L = 0
  for (let i = 0; i < m; i++)
    for (let j = i + 1; j < m; j++) {
      let r2 = 0
      for (let k = 0; k < d; k++) r2 += (points[i * d + k] - points[j * d + k]) ** 2
      if (r2 > 0) L = Math.max(L, Math.abs(values[i] - values[j]) / Math.sqrt(r2))
    }
  return L
}

/** Options of `localPenalisation`. */
export type PenalisationOptions = {
  /** The posterior mean $\mu$ on the candidates, `[m]`. */
  mean: Tensor
  /** The posterior standard deviation $\sigma \ge 0$ on the candidates, `[m]`. */
  sd: Tensor
  /**
   * The Lipschitz constant $L$ of the objective, in the candidates' coordinates (default the largest slope of the mean
   * between two candidates, `gridLipschitz`, at least $10^{-7}$).
   */
  lipschitz?: number
  /**
   * The minimum value $M$ (default the smallest mean on the candidates, or the largest when maximising); the best value
   * observed is the usual choice.
   */
  minimum?: number
  /** Minimise the objective (default true); false maximises it, $M$ then being the maximum. */
  minimise?: boolean
}

/**
 * Local penalisation of an acquisition (González, Dai, Hennig and Lawrence, 2016): the acquisition on the candidates,
 * passed through the soft-plus $\ln(1 + e^{\alpha})$ if any entry is negative, times one factor per pending point
 * $\xvec_j$, $\Phi((L\lVert \xvec - \xvec_j \rVert - \mu(\xvec_j) + M) / \sigma(\xvec_j))$: the probability that
 * $\xvec$ is outside the ball of radius $(\mu(\xvec_j) - M) / L$ around $\xvec_j$, where the minimum cannot be (when
 * maximising, $\Phi((L\lVert \xvec - \xvec_j \rVert + \mu(\xvec_j) - M) / \sigma(\xvec_j))$). Maximising the result
 * chooses the next point of a batch without refitting the surrogate. A pending point's mean and standard deviation are
 * those of its nearest candidate (its own when it is one); where that standard deviation is 0 its factor is 1 outside
 * the ball and 0 inside. Throws `ShapeError` when the candidates, acquisition, mean and standard deviation disagree in
 * size or the pending points are not in the candidates' dimension.
 *
 * @param acq The acquisition $\alpha$ on the candidates, `[m]`.
 * @param grid The candidates, `[m, d]` (or `[m]` when $d = 1$).
 * @param pending The points already chosen, `[p, d]` (or `[p]` when $d = 1$); none (`p = 0`) leaves $g(\alpha)$.
 * @param options The posterior on the candidates, $L$, $M$ and the direction, as `PenalisationOptions`.
 * @returns The penalised acquisition on the candidates, `[m]`.
 *
 * @example The acquisition is cut down around a pending point, and less so far away
 * const grid = tensor([0, 0.25, 0.5, 0.75, 1])
 * const acq = tensor([1, 1, 1, 1, 1])
 * const mean = tensor([0.5, 0.2, 0, 0.2, 0.5])
 * const sd = tensor([0.1, 0.1, 0.1, 0.1, 0.1])
 * print('L = 1:', localPenalisation(acq, grid, tensor([0.5]), { mean, sd, lipschitz: 1, minimum: -0.2 }))
 * print('L = 4:', localPenalisation(acq, grid, tensor([0.5]), { mean, sd, lipschitz: 4, minimum: -0.2 }))
 */
export function localPenalisation(acq: Tensor, grid: Tensor, pending: Tensor, options: PenalisationOptions): Tensor {
  const where = 'localPenalisation'
  const a = Float64Array.from(toFlat(acq))
  const m = a.length
  const d = grid.shape.length === 1 ? 1 : grid.shape[1]
  const g = toFlat(grid)
  const mu = toFlat(options.mean)
  const sd = toFlat(options.sd)
  if (g.length !== m * d || mu.length !== m || sd.length !== m)
    throw new ShapeError(where, `${where}: the grid, acquisition, mean and sd must have ${m} entries`)
  const ps = pending.shape
  if (!((ps.length === 2 && ps[1] === d) || (ps.length === 1 && d === 1)))
    throw new ShapeError(where, `${where}: pending points must be [p, ${d}]`)
  const P = toFlat(pending)
  const p = P.length / d
  const { minimise = true } = options
  let M = options.minimum ?? (minimise ? Infinity : -Infinity)
  if (options.minimum === undefined) for (const v of mu) M = minimise ? Math.min(M, v) : Math.max(M, v)
  const L = Math.max(options.lipschitz ?? gridLipschitz(mu, g, d), 1e-7)
  const out = a.some((v) => v < 0) ? a.map((v) => (v > 30 ? v : Math.log1p(Math.exp(v)))) : a
  const dist = (i: number, j: number) => {
    let r2 = 0
    for (let k = 0; k < d; k++) r2 += (g[i * d + k] - P[j * d + k]) ** 2
    return Math.sqrt(r2)
  }
  for (let j = 0; j < p; j++) {
    let near = 0
    for (let i = 1; i < m; i++) if (dist(i, j) < dist(near, j)) near = i
    const shift = minimise ? M - mu[near] : mu[near] - M
    const s = sd[near]
    for (let i = 0; i < m; i++) {
      const top = L * dist(i, j) + shift
      out[i] *= s > 0 ? normalCdf(top / s) : top > 0 ? 1 : 0
    }
  }
  return fromData(out, [m])
}
