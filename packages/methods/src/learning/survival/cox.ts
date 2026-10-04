/**
 * The Cox proportional-hazards model (Cox, 1972): the hazard of subject i is h(t | xᵢ) = h₀(t) exp(xᵢᵀβ), with the
 * baseline hazard h₀ left unspecified. β maximises the partial likelihood (Cox, 1975), the product over event times of
 * the chance that the subject who failed was the one to fail among those still at risk:
 *
 *   ℓ(β) = Σ_events [xᵢᵀβ − log Σ_{j ∈ R(tᵢ)} exp(xⱼᵀβ)].
 *
 * With d tied events at one time, Breslow (1974) uses the full risk-set sum for each of the d, and Efron (1977) removes
 * the tied subjects' share in steps: the l-th of them (l = 0, …, d − 1) sees Σ_R − (l/d) Σ_D. The fit is Newton's method
 * on ℓ, with the exact gradient and Hessian (score and observed information), halving a step that lowers ℓ. Standard
 * errors are the square roots of the diagonal of the inverse information. The baseline cumulative hazard is Breslow's
 * estimator Ĥ₀(t) = Σ_{tⱼ ≤ t} dⱼ / Σ_{R(tⱼ)} exp(xᵀβ̂).
 */

import { dense, fromData, toFlat, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { inverse } from 'aifn-compute/numerics/linalg'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `coxPh`. */
export interface CoxOptions {
  /** How tied event times are handled: `efron` (default, as lifelines) or `breslow` (as R's `coxph(ties = 'breslow')`). */
  ties?: 'efron' | 'breslow'
  /** Most Newton steps (default 50). */
  maxIterations?: number
  /** Stop when the largest change in β is below this (default 1e-9). */
  tolerance?: number
  /** A ridge penalty ½λ‖β‖² subtracted from ℓ (default 0). */
  penalty?: number
}

/** A fitted Cox model. */
export interface CoxFit {
  readonly kind: 'cox-model'
  readonly coefficients: Float64Array
  /** exp(β): the multiplicative change in hazard per unit of each covariate. */
  readonly hazardRatios: Float64Array
  readonly standardErrors: Float64Array
  /** The partial log-likelihood at β̂ and at β = 0. */
  readonly logPartialLikelihood: number
  readonly nullLogPartialLikelihood: number
  /** The partial log-likelihood after each Newton step (entry 0 at β = 0). */
  readonly path: Float64Array
  /** β after each Newton step, row-major [(iterations + 1) × p] (row 0 is β = 0). */
  readonly coefficientPath: Float64Array
  readonly iterations: number
  readonly converged: boolean
  readonly ties: 'efron' | 'breslow'
  /** Breslow's baseline cumulative hazard at the distinct event times. */
  readonly baseline: { readonly time: Float64Array; readonly cumulativeHazard: Float64Array }
}

type Data = { n: number; p: number; x: Float64Array; t: Float64Array; e: Float64Array; order: number[] }

function prepare(x: MatrixLike, time: VectorLike, event: VectorLike): Data {
  const X = dense.toMatrixF64(x, 'coxPh')
  const t = dense.toF64(time, 'coxPh')
  const e = dense.toF64(event, 'coxPh')
  const n = t.length
  if (X.m !== n || e.length !== n) throw new ShapeError('coxPh', 'coxPh: x, time and event must have the same rows')
  // Ascending time; at a tie, events before censorings (a censoring at t is still at risk at t).
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => t[i] - t[j] || e[j] - e[i])
  return { n, p: X.n, x: X.data, t, e, order }
}

/** ℓ(β), its gradient and its Hessian, by one sweep from the latest time down (risk sets grow as time falls). */
function partial(d: Data, beta: Float64Array, ties: 'efron' | 'breslow') {
  const { n, p, x, t, e, order } = d
  const eta = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let k = 0; k < p; k++) s += x[i * p + k] * beta[k]
    eta[i] = s
  }
  // The weights exp(η) are taken relative to the largest η, c: every risk-set ratio is unchanged, log Σ w gains c back
  // below, and a covariate far from zero (a location shift, which Cox's model ignores) no longer overflows exp.
  const c = eta.reduce((a, v) => Math.max(a, v), -Infinity)
  const w = Float64Array.from(eta, (v) => Math.exp(v - c))
  let ll = 0
  const grad = new Float64Array(p)
  const hess = new Float64Array(p * p)
  // Risk-set sums: S0 = Σ w, S1 = Σ w x, S2 = Σ w x xᵀ over subjects with time ≥ the current one.
  let S0 = 0
  const S1 = new Float64Array(p)
  const S2 = new Float64Array(p * p)
  const times: number[] = []
  const hazard: number[] = []
  for (let k = n - 1; k >= 0;) {
    const tk = t[order[k]]
    let j = k
    // Everyone at this time joins the risk set; the events among them are the tied set D.
    let D0 = 0
    const D1 = new Float64Array(p)
    const D2 = new Float64Array(p * p)
    let deaths = 0
    for (; j >= 0 && t[order[j]] === tk; j--) {
      const i = order[j]
      S0 += w[i]
      for (let a = 0; a < p; a++) {
        S1[a] += w[i] * x[i * p + a]
        for (let b = 0; b < p; b++) S2[a * p + b] += w[i] * x[i * p + a] * x[i * p + b]
      }
      if (e[i] === 1) {
        deaths++
        D0 += w[i]
        ll += eta[i]
        for (let a = 0; a < p; a++) {
          D1[a] += w[i] * x[i * p + a]
          grad[a] += x[i * p + a]
          for (let b = 0; b < p; b++) D2[a * p + b] += w[i] * x[i * p + a] * x[i * p + b]
        }
      }
    }
    k = j
    if (deaths === 0) continue
    times.push(tk)
    for (let l = 0; l < deaths; l++) {
      const f = ties === 'efron' ? l / deaths : 0
      const phi = S0 - f * D0
      ll -= Math.log(phi) + c
      for (let a = 0; a < p; a++) {
        const ma = (S1[a] - f * D1[a]) / phi
        grad[a] -= ma
        for (let b = 0; b < p; b++) {
          const mb = (S1[b] - f * D1[b]) / phi
          hess[a * p + b] -= (S2[a * p + b] - f * D2[a * p + b]) / phi - ma * mb
        }
      }
    }
    // Breslow's increment uses the full risk set: d / S0.
    hazard.push((deaths / S0) * Math.exp(-c))
  }
  times.reverse()
  hazard.reverse()
  return { ll, grad, hess, times, hazard }
}

/**
 * Fit a Cox model to covariates x [n, p], times [n] and event flags [n] (1 an event, 0 right-censored) by Newton's
 * method on the partial likelihood.
 */
export function coxPh(x: MatrixLike, time: VectorLike, event: VectorLike, options: CoxOptions = {}): CoxFit {
  const { ties = 'efron', maxIterations = 50, tolerance = 1e-9, penalty = 0 } = options
  const d = prepare(x, time, event)
  const { p } = d
  const evaluate = (beta: Float64Array) => {
    const r = partial(d, beta, ties)
    if (penalty > 0)
      for (let a = 0; a < p; a++) {
        r.ll -= 0.5 * penalty * beta[a] * beta[a]
        r.grad[a] -= penalty * beta[a]
        r.hess[a * p + a] -= penalty
      }
    return r
  }
  let beta = new Float64Array(p)
  let current = evaluate(beta)
  const nullLl = current.ll
  const path = [current.ll]
  const betas: number[] = [...beta]
  let converged = false
  let iterations = 0
  for (; iterations < maxIterations && !converged;) {
    iterations++
    // Newton: β ← β + (−H)⁻¹ g.
    const info = fromData(
      Float64Array.from(current.hess, (v) => -v),
      [p, p],
    )
    const step = dense.matVec(toFlat(inverse(info)), current.grad, p, p)
    let scale = 1
    let next = beta
    let trial = current
    for (let halving = 0; halving < 30; halving++) {
      next = Float64Array.from(beta, (b, a) => b + scale * step[a])
      trial = evaluate(next)
      if (trial.ll >= current.ll - 1e-12) break
      scale /= 2
    }
    let change = 0
    for (let a = 0; a < p; a++) change = Math.max(change, Math.abs(next[a] - beta[a]))
    beta = next
    current = trial
    path.push(current.ll)
    betas.push(...beta)
    if (change < tolerance) converged = true
  }
  const cov = toFlat(
    inverse(
      fromData(
        Float64Array.from(current.hess, (v) => -v),
        [p, p],
      ),
    ) as Tensor,
  )
  const base = partial(d, beta, 'breslow')
  const cumulative = new Float64Array(base.hazard.length)
  let acc = 0
  base.hazard.forEach((h, j) => (cumulative[j] = acc += h))
  return {
    kind: 'cox-model',
    coefficients: beta,
    hazardRatios: Float64Array.from(beta, Math.exp),
    standardErrors: Float64Array.from({ length: p }, (_, a) => Math.sqrt(cov[a * p + a])),
    logPartialLikelihood: current.ll,
    nullLogPartialLikelihood: nullLl,
    path: Float64Array.from(path),
    coefficientPath: Float64Array.from(betas),
    iterations,
    converged,
    ties,
    baseline: { time: Float64Array.from(base.times), cumulativeHazard: cumulative },
  }
}

/** The partial log-likelihood ℓ(β) at given coefficients (Efron or Breslow ties), e.g. to draw its profile. */
export function coxPartialLikelihood(
  x: MatrixLike,
  time: VectorLike,
  event: VectorLike,
  coefficients: ArrayLike<number>,
  ties: 'efron' | 'breslow' = 'efron',
): number {
  return partial(prepare(x, time, event), Float64Array.from(coefficients), ties).ll
}

/**
 * The survival curve S(t | x) = exp(−Ĥ₀(t) exp(xᵀβ̂)) of a subject with covariates x, at the model's event times.
 */
export function coxSurvival(fit: CoxFit, x: ArrayLike<number>): Float64Array {
  let eta = 0
  for (let a = 0; a < fit.coefficients.length; a++) eta += x[a] * fit.coefficients[a]
  const r = Math.exp(eta)
  return Float64Array.from(fit.baseline.cumulativeHazard, (h) => Math.exp(-h * r))
}

/**
 * Harrell's concordance index: among comparable pairs (the shorter time is an event), the share where the subject who
 * failed first has the higher risk score (ties in score count ½).
 */
export function harrellConcordance(time: VectorLike, event: VectorLike, risk: VectorLike): number {
  const t = dense.toF64(time, 'harrellConcordance')
  const e = dense.toF64(event, 'harrellConcordance')
  const r = dense.toF64(risk, 'harrellConcordance')
  let concordant = 0
  let pairs = 0
  for (let i = 0; i < t.length; i++) {
    if (e[i] !== 1) continue
    for (let j = 0; j < t.length; j++) {
      if (!(t[j] > t[i])) continue
      pairs++
      concordant += r[i] > r[j] ? 1 : r[i] === r[j] ? 0.5 : 0
    }
  }
  return pairs > 0 ? concordant / pairs : NaN
}
