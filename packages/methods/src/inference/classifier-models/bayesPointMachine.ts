/** The Bayes point machine by expectation propagation, part of `aifn-methods/inference/classifier-models`. */

import type { Status } from 'aifn-compute/foundation/contracts'
import { normalCdf } from 'aifn-compute/numerics/special'
import { fromData, toRows, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { inverse } from 'aifn-compute/numerics/linalg'
import { probitTilted, stepTilted, type Tilted } from 'aifn-compute/inference/expectation-propagation'

// ── Bayes point machine ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link bayesPointMachine}. */
export interface BayesPointMachineOptions {
  /** Inputs, n × d (append a column of ones for a bias). */
  x: Matrix | readonly (readonly number[])[]
  /** Labels ±1, length n. */
  y: ArrayLike<number>
  /** Prior w ~ N(0, priorVariance I). Default 1. */
  priorVariance?: number
  /** `step` (noise-free: 𝟙(y wᵀx > 0)) or `probit` (Φ(y wᵀx / s)). Default `probit`. */
  likelihood?: 'step' | 'probit'
  /** s² for the probit likelihood. Default 0.1. */
  noiseVariance?: number
  damping?: number
  tolerance?: number
}

/** The state of EP for the Bayes point machine. */
export interface BayesPointMachineState extends Status {
  /** Site updates done. */
  t: number
  /** The likelihood and its noise, for predictions (`bayesPointMachinePredict`). */
  likelihood: 'step' | 'probit'
  noiseVariance: number
  /** One-dimensional sites on the projections sᵢ = wᵀxᵢ: precision and shift (length n). */
  sitePrecision: Tensor
  siteShift: Tensor
  /** The posterior over w: mean (the Bayes point) and covariance. */
  mean: Vector
  covariance: Matrix
  /** The point updated last, its cavity on sᵢ and tilted moments. */
  point: number
  cavity: { mean: number; variance: number }
  tilted: Tilted
  ok: boolean
  /** Sweeps completed, and the next point's position in the sweep. */
  sweep: number
  position: number
  change: number
  sweepChange: number
  lastSweepChange: number
  converged: boolean
}

/** Σ = (I/σ₀² + Σᵢ τᵢ xᵢxᵢᵀ)⁻¹ and m = Σ Σᵢ νᵢ xᵢ. */
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

const matrixOf = (rows: number[][]): Matrix =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])

/**
 * The Bayes point machine by EP (Minka 2001, thesis §5.2): each point's likelihood acts on w only through
 * sᵢ = wᵀxᵢ, so its site is a one-dimensional Gaussian on sᵢ. An update projects q(w) onto sᵢ (mean mᵀxᵢ, variance
 * xᵢᵀΣxᵢ), removes the site, applies the step or probit factor, and refits the site. One point per step; no start.
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
 * Predictive probabilities P(y = +1 | x) under the BPM posterior: Φ(mᵀx / √(xᵀΣx + s²)) for the probit likelihood,
 * Φ(mᵀx / √(xᵀΣx)) for the step. `points` is n × d; returns a length-n vector.
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
