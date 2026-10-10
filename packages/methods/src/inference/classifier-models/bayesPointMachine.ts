/**
 * The Bayes point machine by expectation propagation: a Gaussian posterior over the weights of a linear classifier,
 * and its predictive probabilities.
 *
 * The model (Herbrich, Graepel and Campbell 2001; Minka 2001) has weights $\wvec \sim \Gauss(\zeros, \sigma_0^2 \Imat)$
 * and labels $y_i = \pm 1$ with a noise-free step likelihood $\indicator(y_i \wvec^\top \xvec_i > 0)$ or a probit
 * $\Phi(y_i \wvec^\top \xvec_i / s)$. Each likelihood acts on $\wvec$ only through $s_i = \wvec^\top \xvec_i$, so EP
 * approximates it by a one-dimensional Gaussian site on $s_i$ with precision $\tau_i$ and shift $\nu_i$, and the
 * posterior is $\Sigmamat = (\Imat / \sigma_0^2 + \sum_i \tau_i \xvec_i \xvec_i^\top)^{-1}$,
 * $\mvec = \Sigmamat \sum_i \nu_i \xvec_i$. The posterior mean $\mvec$ is the Bayes point.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normalCdf } from 'aifn-compute/numerics/special'
import { fromData, toRows, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { inverse } from 'aifn-compute/numerics/linalg'
import { probitTilted, stepTilted, type Tilted } from 'aifn-compute/inference/expectation-propagation'

// ── Bayes point machine ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link bayesPointMachine}. */
export interface BayesPointMachineOptions {
  /** Inputs, $n \times d$, as a matrix or rows (append a column of ones for a bias). */
  x: Matrix | readonly (readonly number[])[]
  /** Labels $\pm 1$, length $n$. */
  y: ArrayLike<number>
  /** The prior variance $\sigma_0^2$ of $\wvec \sim \Gauss(\zeros, \sigma_0^2 \Imat)$. Default 1. */
  priorVariance?: number
  /**
   * `step` (noise-free: $\indicator(y \wvec^\top \xvec > 0)$) or `probit` ($\Phi(y \wvec^\top \xvec / s)$). Default
   * `probit`.
   */
  likelihood?: 'step' | 'probit'
  /** $s^2$ for the probit likelihood. Default 0.1. */
  noiseVariance?: number
  /** The share $\delta$ of the old site kept at each update: new $= (1 - \delta)$ refit $+ \delta$ old. Default 0. */
  damping?: number
  /** Converged when no site changes by more than this over a whole sweep. Default $10^{-8}$. */
  tolerance?: number
}

/** The state of EP for the Bayes point machine. */
export interface BayesPointMachineState extends Status {
  /** Site updates done. */
  t: number
  /** The likelihood, for predictions (`bayesPointMachinePredict`). */
  likelihood: 'step' | 'probit'
  /** The probit's noise variance $s^2$, for predictions. */
  noiseVariance: number
  /** The precisions $\tau_i$ of the sites, one-dimensional Gaussians on $s_i = \wvec^\top \xvec_i$ (length $n$). */
  sitePrecision: Tensor
  /** The shifts $\nu_i$ (precision times mean) of the sites (length $n$). */
  siteShift: Tensor
  /** The posterior mean $\mvec$ of $\wvec$: the Bayes point. */
  mean: Vector
  /** The posterior covariance $\Sigmamat$ of $\wvec$ ($d \times d$). */
  covariance: Matrix
  /** The point updated last ($-1$ before the first update). */
  point: number
  /** Its cavity on $s_i$: the marginal of $s_i$ with the site removed (NaN when the cavity was improper). */
  cavity: { mean: number; variance: number }
  /** Its tilted moments: the cavity times the likelihood, with $\log Z$. */
  tilted: Tilted
  /** False when the cavity or the tilted variance was not positive, and the site was left unchanged. */
  ok: boolean
  /** Sweeps completed. */
  sweep: number
  /** The next point's position in the sweep. */
  position: number
  /** The largest change of the site's precision or shift in the last update (0 when it was skipped). */
  change: number
  /** The largest change so far in the current sweep. */
  sweepChange: number
  /** The largest change in the last complete sweep ($\infty$ before the first). */
  lastSweepChange: number
  /** True once a whole sweep changed no site by more than `tolerance` (at once for no data). */
  converged: boolean
}

/**
 * The posterior $\Sigmamat = (\Imat / \sigma_0^2 + \sum_i \tau_i \xvec_i \xvec_i^\top)^{-1}$ and
 * $\mvec = \Sigmamat \sum_i \nu_i \xvec_i$ of the weights given the sites.
 *
 * @param x The inputs as rows, $n \times d$.
 * @param tau The site precisions $\tau_i$ (length $n$).
 * @param nu The site shifts $\nu_i$ (length $n$).
 * @param priorVariance The prior variance $\sigma_0^2$.
 * @returns `mean` ($\mvec$, length $d$) and `covariance` ($\Sigmamat$, as rows).
 */
function bpmPosterior(x: number[][], tau: Float64Array, nu: Float64Array, priorVariance: number) {
  const d = x[0]?.length ?? 0
  const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 / priorVariance : 0)))
  const b = new Array<number>(d).fill(0)
  x.forEach((xi, i) => {
    for (let p = 0; p < d; p++) {
      b[p] += nu[i] * xi[p]
      for (let q = 0; q < d; q++) A[p][q] += tau[i] * xi[p] * xi[q]
    }
  })
  const inv = toRows(inverse(matrixOf(A)))
  const mean = inv.map((row) => row.reduce((s, v, j) => s + v * b[j], 0))
  return { mean, covariance: inv }
}

/**
 * Rows packed into a matrix (its column count from the first row).
 *
 * @param rows The rows, of equal length.
 * @returns The matrix.
 */
const matrixOf = (rows: number[][]): Matrix =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])

/**
 * The Bayes point machine by EP (Minka 2001, thesis §5.2): each point's likelihood acts on $\wvec$ only through
 * $s_i = \wvec^\top \xvec_i$, so its site is a one-dimensional Gaussian on $s_i$. An update projects $q(\wvec)$ onto
 * $s_i$ (mean $\mvec^\top \xvec_i$, variance $\xvec_i^\top \Sigmamat \xvec_i$), removes the site, applies the step or
 * probit factor, and refits the site (damped by `damping`). One point per step, in order, sweep after sweep; no start.
 * A point whose cavity or tilted variance is not positive is skipped (`ok` false). Once converged, a step changes
 * nothing but `t`.
 *
 * @param o The data, the prior, the likelihood and the EP settings.
 * @returns The algorithm, named `ep.bayes-point-machine`.
 *
 * @example Four points on a line, with a bias column
 * const x = [[2, 1], [1, 1], [-1, 1], [-2, 1]]
 * const s = run(bayesPointMachine({ x, y: [1, 1, -1, -1] }), undefined, 200)
 * print('sweeps:', s.sweep, 'converged:', s.converged)
 * print('Bayes point:', s.mean)
 * print('covariance:', s.covariance)
 *
 * @example The noise-free step likelihood is surer
 * const x = [[2, 1], [1, 1], [-1, 1], [-2, 1]]
 * const y = [1, 1, -1, -1]
 * for (const likelihood of ['probit', 'step']) {
 *   const s = run(bayesPointMachine({ x, y, likelihood }), undefined, 200)
 *   print(likelihood, 'P(y = +1) at 0.5 and 3:', bayesPointMachinePredict(s, [[0.5, 1], [3, 1]]))
 * }
 */
export function bayesPointMachine(o: BayesPointMachineOptions): Algorithm<void, BayesPointMachineState> {
  const x = 'shape' in o.x ? toRows(o.x as Matrix) : (o.x as readonly (readonly number[])[]).map((r) => [...r])
  const y = Array.from(o.y)
  const n = x.length
  const priorVariance = o.priorVariance ?? 1
  const likelihood = o.likelihood ?? 'probit'
  const noiseVariance = o.noiseVariance ?? 0.1
  const damping = o.damping ?? 0
  const tolerance = o.tolerance ?? 1e-8
  return {
    name: 'ep.bayes-point-machine',
    init: () => {
      const zero = new Float64Array(n)
      const post = bpmPosterior(x, zero, zero, priorVariance)
      return {
        t: 0,
        likelihood,
        noiseVariance,
        sitePrecision: fromData(new Float64Array(n), [n]),
        siteShift: fromData(new Float64Array(n), [n]),
        mean: fromData(Float64Array.from(post.mean), [post.mean.length]),
        covariance: matrixOf(post.covariance),
        point: -1,
        cavity: { mean: NaN, variance: NaN },
        tilted: { logZ: NaN, mean: NaN, variance: NaN },
        ok: true,
        sweep: 0,
        position: 0,
        change: 0,
        sweepChange: 0,
        lastSweepChange: Infinity,
        converged: n === 0,
      }
    },
    step: (s) => {
      if (s.converged) return { ...s, t: s.t + 1 }
      const i = s.position
      const xi = x[i]
      const S = toRows(s.covariance)
      const m = Array.from(s.mean.data)
      const tau = Float64Array.from(s.sitePrecision.data)
      const nu = Float64Array.from(s.siteShift.data)
      // The marginal of sᵢ = wᵀxᵢ under q, then its cavity.
      const Sx = S.map((row) => row.reduce((acc, v, j) => acc + v * xi[j], 0))
      const vi = Sx.reduce((acc, v, j) => acc + v * xi[j], 0)
      const mi = m.reduce((acc, v, j) => acc + v * xi[j], 0)
      const ct = 1 / vi - tau[i]
      const cn = mi / vi - nu[i]
      let ok = ct > 0
      let change = 0
      let cavity = { mean: NaN, variance: NaN }
      let t: Tilted = { logZ: NaN, mean: NaN, variance: NaN }
      if (ok) {
        cavity = { mean: cn / ct, variance: 1 / ct }
        t =
          likelihood === 'step'
            ? stepTilted(y[i] * cavity.mean, cavity.variance)
            : probitTilted(cavity.mean, cavity.variance, y[i], { noiseVariance })
        // The step factor is written on y·s; map its mean back to s.
        if (likelihood === 'step') t = { ...t, mean: y[i] * t.mean }
        ok = t.variance > 0
      }
      if (ok) {
        const nt = 1 / t.variance - ct
        const nn = t.mean / t.variance - cn
        const dt = (1 - damping) * nt + damping * tau[i]
        const dn = (1 - damping) * nn + damping * nu[i]
        change = Math.max(Math.abs(dt - tau[i]), Math.abs(dn - nu[i]))
        tau[i] = dt
        nu[i] = dn
      }
      const post = bpmPosterior(x, tau, nu, priorVariance)
      let position = i + 1
      let { sweep, sweepChange, lastSweepChange } = s
      let converged: boolean = s.converged
      sweepChange = Math.max(sweepChange, change)
      if (position >= x.length) {
        converged = sweepChange < tolerance
        lastSweepChange = sweepChange
        sweepChange = 0
        position = 0
        sweep++
      }
      return {
        ...s,
        t: s.t + 1,
        sitePrecision: fromData(tau, [tau.length]),
        siteShift: fromData(nu, [nu.length]),
        mean: fromData(Float64Array.from(post.mean), [post.mean.length]),
        covariance: matrixOf(post.covariance),
        point: i,
        cavity,
        tilted: t,
        ok,
        sweep,
        position,
        change,
        sweepChange,
        lastSweepChange,
        converged,
      }
    },
  }
}

/**
 * Predictive probabilities $P(y = +1 \mid \xvec)$ under the BPM posterior:
 * $\Phi(\mvec^\top \xvec / \sqrt{\xvec^\top \Sigmamat \xvec + s^2})$ for the probit likelihood, and
 * $\Phi(\mvec^\top \xvec / \sqrt{\xvec^\top \Sigmamat \xvec})$ for the step.
 *
 * @param s A state of `bayesPointMachine`: its posterior, likelihood and noise are read.
 * @param points The points to predict, $m \times d$ (as a matrix or rows), with the same columns as the inputs.
 * @returns The probabilities, a vector of length $m$.
 *
 * @example Far from the boundary the prediction is confident, near it less so
 * const x = [[2, 1], [1, 1], [-1, 1], [-2, 1]]
 * const s = run(bayesPointMachine({ x, y: [1, 1, -1, -1] }), undefined, 200)
 * print(bayesPointMachinePredict(s, [[-3, 1], [0, 1], [0.5, 1], [3, 1]]))
 */
export function bayesPointMachinePredict(
  s: BayesPointMachineState,
  points: Matrix | readonly (readonly number[])[],
): Vector {
  const X = 'shape' in points ? toRows(points as Matrix) : (points as readonly (readonly number[])[])
  const S = toRows(s.covariance)
  const m = s.mean.data
  const noise = s.likelihood === 'probit' ? s.noiseVariance : 0
  const out = Float64Array.from(X, (x) => {
    let mu = 0
    let v = 0
    for (let p = 0; p < x.length; p++) {
      mu += m[p] * x[p]
      for (let q = 0; q < x.length; q++) v += x[p] * S[p][q] * x[q]
    }
    return normalCdf(mu / Math.sqrt(v + noise)) as number
  })
  return fromData(out, [out.length])
}
