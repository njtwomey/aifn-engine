/**
 * ARMA(p, q) processes x_t − μ = Σ_{i≤p} φ_i (x_{t−i} − μ) + ε_t + Σ_{j≤q} θ_j ε_{t−j}, ε_t ~ N(0, σ²): stationarity
 * and invertibility from the roots of the AR and MA polynomials, ψ weights and the theoretical autocovariance,
 * simulation, conditional and exact likelihoods, fitting and forecasting. Sign conventions follow Box, Jenkins &
 * Reinsel (2008) for φ and statsmodels for θ (the MA polynomial is 1 + θ₁z + … + θ_q z^q).
 */

import { normals, type Stream } from 'aifn-compute/foundation/random'
import { roots } from 'aifn-compute/numerics/polynomial'
import { normalQuantile } from 'aifn-compute/numerics/special'
import {
  complexAbs,
  eye,
  fromData,
  mean as meanOf,
  outer,
  reshape,
  sub,
  tensor,
  toFlat,
  toRows,
  type Matrix,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { kron, luFactor, luSolve } from 'aifn-compute/numerics/linalg'
import { filterAll, type Model } from 'aifn-compute/inference/filtering'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { simplexFit, type FitState } from './fit'
import { toVec, type VectorLike } from './inputs'
import { DomainError } from 'aifn-compute/foundation/errors'

/** An ARMA model. Omitted parts are empty (no AR or MA terms), σ = 1 and μ = 0. */
export type ArmaSpec = {
  /** φ₁ … φ_p. */
  ar?: VectorLike
  /** θ₁ … θ_q. */
  ma?: VectorLike
  /** The innovation standard deviation σ. */
  sigma?: number
  /** The process mean μ. */
  mean?: number
}

type Parsed = { ar: number[]; ma: number[]; sigma: number; mean: number }
const parse = (m: ArmaSpec, where: string): Parsed => ({
  ar: m.ar ? toVec(m.ar, where) : [],
  ma: m.ma ? toVec(m.ma, where) : [],
  sigma: m.sigma ?? 1,
  mean: m.mean ?? 0,
})

/** Roots of a lag polynomial (complex128), with their moduli. */
export type LagRoots = { roots: Tensor; modulus: Vector; minModulus: number }

function lagRoots(c: number[]): LagRoots {
  // c₀ + c₁z + … + c_k z^k, highest degree first for `roots`; trailing zeros drop the degree.
  const coef = [...c]
  while (coef.length > 1 && coef[coef.length - 1] === 0) coef.pop()
  if (coef.length <= 1)
    return { roots: fromData(new Float64Array(0), [0], 'complex128'), modulus: tensor([]), minModulus: Infinity }
  const r = roots([...coef].reverse())
  const modulus = complexAbs(r)
  return { roots: r, modulus, minModulus: Math.min(...toFlat(modulus)) }
}

/**
 * The roots of the AR polynomial 1 − φ₁z − … − φ_p z^p and the MA polynomial 1 + θ₁z + … + θ_q z^q. The process is
 * stationary (causal) when every AR root lies outside the unit circle and invertible when every MA root does.
 */
export function armaRoots(model: ArmaSpec): { ar: LagRoots; ma: LagRoots } {
  const { ar, ma } = parse(model, 'armaRoots')
  return { ar: lagRoots([1, ...ar.map((v) => -v)]), ma: lagRoots([1, ...ma]) }
}

/** True when every root of 1 − Σ φ_i zⁱ lies strictly outside the unit circle. */
export const isStationary = (ar: VectorLike): boolean => armaRoots({ ar }).ar.minModulus > 1

/** True when every root of 1 + Σ θ_j z^j lies strictly outside the unit circle. */
export const isInvertible = (ma: VectorLike): boolean => armaRoots({ ma }).ma.minModulus > 1

/** ψ₀ … ψ_{count−1} (shared with `sarima.ts`; any AR polynomial, stationary or not). */
export function psi(ar: number[], ma: number[], count: number): number[] {
  const out = [1]
  for (let j = 1; j < count; j++) {
    let v = ma[j - 1] ?? 0
    for (let i = 0; i < ar.length && i < j; i++) v += ar[i] * out[j - 1 - i]
    out.push(v)
  }
  return out
}

/**
 * The ψ weights of the causal MA(∞) form x_t − μ = Σ_j ψ_j ε_{t−j}: ψ₀ = 1, ψ_j = θ_j + Σ_i φ_i ψ_{j−i}
 * (Brockwell & Davis, 1991, eq. 3.1.7). Returns ψ₀ … ψ_J.
 */
export function psiWeights(model: ArmaSpec, J: number): Vector {
  const { ar, ma } = parse(model, 'psiWeights')
  return tensor(psi(ar, ma, J + 1))
}

/**
 * The theoretical autocovariance γ(0) … γ(maxLag) of a stationary ARMA process, γ(h) = σ² Σ_j ψ_j ψ_{j+h}, with the ψ
 * weights summed until they fall below 1e-17 of the largest (they decay geometrically at the rate of the AR root
 * nearest the unit circle). Throws for a non-stationary AR polynomial, whose autocovariance does not exist.
 */
export function armaAutocovariance(model: ArmaSpec, maxLag: number): Vector {
  const { ar, ma, sigma } = parse(model, 'armaAutocovariance')
  const roots = armaRoots(model).ar
  if (!(roots.minModulus > 1))
    throw new DomainError('armaAutocovariance', 'armaAutocovariance: the AR polynomial is not stationary')
  // Enough terms that ρ^J < 1e-17 for the slowest root modulus ρ⁻¹ = minModulus.
  const decay = Number.isFinite(roots.minModulus) ? Math.log(1e-17) / -Math.log(roots.minModulus) : 0
  const J = Math.min(Math.ceil(decay) + ma.length + 50, 200000)
  const w = psi(ar, ma, J + maxLag + 1)
  const g = Array.from({ length: maxLag + 1 }, (_, h) => {
    let s = 0
    for (let j = 0; j + h < w.length; j++) s += w[j] * w[j + h]
    return sigma * sigma * s
  })
  return tensor(g)
}

/** The theoretical autocorrelation ρ(0) … ρ(maxLag) of a stationary ARMA process (see `armaAutocovariance`). */
export function armaAutocorrelation(model: ArmaSpec, maxLag: number): Vector {
  const g = toFlat(armaAutocovariance(model, maxLag))
  return tensor(g.map((v) => v / g[0]))
}

/** A simulated ARMA series. */
export type ArmaSimulation = {
  /** x₁ … x_n (after the burn-in). NaN from `divergedAt` on if the recursion overflowed. */
  x: Vector
  /** The innovations ε₁ … ε_n used for the kept values. */
  innovations: Vector
  /** Whether the AR polynomial is stationary; if not, the series grows without bound and is not a stationary sample. */
  stationary: boolean
  /** True if a value overflowed to a non-finite number. Nothing is clipped. */
  diverged: boolean
  /** The index in `x` of the first non-finite value, or −1. */
  divergedAt: number
}

/**
 * Simulate n values of an ARMA process from zero presample values, discarding `burn` values first so the start-up
 * transient is gone (default 200 for a stationary model and 0 otherwise, so an explosive model is seen from its
 * start). Innovations are σ·N(0, 1) draws from `s`. Values are never clipped: an explosive model is reported through
 * `stationary: false` and, once it overflows, `diverged` (the site's copy clipped at ±10⁶ instead).
 */
export function simulateArma(s: Stream, model: ArmaSpec, n: number, { burn }: { burn?: number } = {}): ArmaSimulation {
  const { ar, ma, sigma, mean } = parse(model, 'simulateArma')
  const stationary = ar.length === 0 || armaRoots({ ar }).ar.minModulus > 1
  const b = burn ?? (stationary ? 200 : 0)
  const total = n + b
  const e = toFlat(normals(s, total, 0, sigma))
  const d = new Array<number>(total).fill(0)
  let divergedAt = -1
  for (let t = 0; t < total; t++) {
    let v = e[t]
    for (let i = 0; i < ar.length && i < t; i++) v += ar[i] * d[t - 1 - i]
    for (let j = 0; j < ma.length && j < t; j++) v += ma[j] * e[t - 1 - j]
    d[t] = v
    if (!Number.isFinite(v)) {
      divergedAt = Math.max(t - b, 0)
      for (let k = t; k < total; k++) d[k] = NaN
      break
    }
  }
  return {
    x: tensor(d.slice(b).map((v) => v + mean)),
    innovations: tensor(e.slice(b)),
    stationary,
    diverged: divergedAt >= 0,
    divergedAt,
  }
}

/**
 * Conditional residuals e_t = (x_t − μ) − Σ φ_i (x_{t−i} − μ) − Σ θ_j e_{t−j} for t ≥ p, with presample residuals
 * zero; entries t < p are 0. Their sum of squares is the conditional sum of squares (CSS) of Box & Jenkins.
 */
export function armaResiduals(x: VectorLike, model: ArmaSpec): Vector {
  const { ar, ma, mean } = parse(model, 'armaResiduals')
  return tensor(residuals(toVec(x, 'armaResiduals'), ar, ma, mean))
}

/** Conditional residuals on arrays (shared with `sarima.ts`). */
export function residuals(x: number[], ar: number[], ma: number[], mean: number): number[] {
  const p = ar.length
  const e = new Array<number>(x.length).fill(0)
  for (let t = p; t < x.length; t++) {
    let v = x[t] - mean
    for (let i = 0; i < p; i++) v -= ar[i] * (x[t - 1 - i] - mean)
    for (let j = 0; j < ma.length && j < t; j++) v -= ma[j] * e[t - 1 - j]
    e[t] = v
  }
  return e
}

/** The stationary covariance of the Harvey state, P = T P Tᵀ + R Rᵀ (σ² = 1), by solving (I − T⊗T) vec P = vec RRᵀ. */
function stationaryStateCovariance(T: number[][], R: number[]): number[][] | null {
  const r = R.length
  const t = tensor(T)
  const f = luFactor(sub(eye(r * r), kron(t, t)))
  if (f.singular) return null
  return toRows(reshape(luSolve(f, reshape(outer(tensor(R), tensor(R)), [r * r])), [r, r]))
}

/** The exact Gaussian log-likelihood of an ARMA model and its prediction errors. */
export type ArmaLikelihood = {
  logLikelihood: number
  /** One-step prediction errors v_t = x_t − E[x_t | x_{<t}]. */
  innovations: Vector
  /** Their variances F_t (in units of σ²), which fall to 1 as the start-up uncertainty resolves. */
  innovationVariance: Vector
  /** The maximising σ² given φ and θ (σ² concentrated out), (1/n) Σ v_t²/F_t. */
  sigma2Hat: number
}

/**
 * The exact Gaussian log-likelihood of x under an ARMA model, by the Kalman filter on Harvey's state-space form
 * (Harvey, 1989, §3.4; Durbin & Koopman, 2012, §3.4): state dimension r = max(p, q + 1), transition with φ in its
 * first column and ones on the superdiagonal, disturbance loading (1, θ₁, …, θ_{r−1}), started from the stationary
 * state covariance. With `model.sigma` omitted, σ² is concentrated out and the profile likelihood is returned. The
 * log-likelihood is −∞ for a non-stationary AR polynomial.
 */
export function armaLogLikelihood(x: VectorLike, model: ArmaSpec): ArmaLikelihood {
  const { ar, ma, mean } = parse(model, 'armaLogLikelihood')
  return exactLikelihood(toVec(x, 'armaLogLikelihood'), ar, ma, mean, model.sigma)
}

/** `armaLogLikelihood` on arrays (shared with `sarima.ts`). */
export function exactLikelihood(x: number[], ar: number[], ma: number[], mean: number, sigma?: number): ArmaLikelihood {
  const n = x.length
  const r = Math.max(ar.length, ma.length + 1)
  const T: number[][] = Array.from({ length: r }, (_, i) =>
    Array.from({ length: r }, (_, j) => (j === 0 ? (ar[i] ?? 0) : +(j === i + 1))),
  )
  const R = Array.from({ length: r }, (_, i) => (i === 0 ? 1 : (ma[i - 1] ?? 0)))
  const bad = (): ArmaLikelihood => ({
    logLikelihood: -Infinity,
    innovations: tensor(new Array(n).fill(NaN)),
    innovationVariance: tensor(new Array(n).fill(NaN)),
    sigma2Hat: NaN,
  })
  if (ar.length && !(armaRoots({ ar }).ar.minModulus > 1)) return bad()
  const P0 = stationaryStateCovariance(T, R)
  if (!P0) return bad()
  // `aifn-compute/inference/filtering`'s Kalman filter on the Harvey form in units of σ²: observation Z α = α[0] without
  // noise, state noise R Rᵀ. The stationary P0 satisfies T P0 Tᵀ + R Rᵀ = P0, so the filter's first prediction is
  // the stationary start itself.
  const harvey: Model = {
    A: tensor(T) as Matrix,
    C: tensor([Array.from({ length: r }, (_, i) => +(i === 0))]) as Matrix,
    Q: tensor(R.map((ri) => R.map((rj) => ri * rj))) as Matrix,
    R: tensor([[0]]) as Matrix,
    m0: tensor(new Array<number>(r).fill(0)) as Vector,
    P0: tensor(P0) as Matrix,
  }
  const f = filterAll(
    harvey,
    x.map((xt) => [xt - mean]),
  )
  if (f.singularSteps.length) return bad()
  const v = f.steps.map((st) => st.innovation.data[st.innovation.offset] as number)
  const F = f.steps.map((st) => st.innovationCov.data[st.innovationCov.offset] as number)
  let sumSq = 0
  let sumLogF = 0
  for (let t = 0; t < n; t++) {
    sumSq += (v[t] * v[t]) / F[t]
    sumLogF += Math.log(F[t])
  }
  const sigma2Hat = sumSq / n
  const s2 = sigma === undefined ? sigma2Hat : sigma * sigma
  const logLikelihood = -0.5 * (n * Math.log(2 * Math.PI * s2) + sumLogF + sumSq / s2)
  return { logLikelihood, innovations: tensor(v), innovationVariance: tensor(F), sigma2Hat }
}

/**
 * Map unconstrained numbers to the coefficients of a stationary AR polynomial: tanh gives partial autocorrelations in
 * (−1, 1), and the Levinson step φ_kj = φ_{k−1,j} − r_k φ_{k−1,k−j} turns them into coefficients (Jones, 1980;
 * Monahan, 1984). Negated, the same map gives invertible MA coefficients.
 */
export function fromPacf(u: number[]): number[] {
  let phi: number[] = []
  for (let k = 0; k < u.length; k++) {
    const r = Math.tanh(u[k])
    phi = [...phi.map((p, j) => p - r * phi[k - 1 - j]), r]
  }
  return phi
}

/** The inverse of `fromPacf` (the backward Levinson step), with partial autocorrelations capped at ±0.99. */
export function toPacf(phi: number[]): number[] {
  let c = [...phi]
  const u = new Array<number>(phi.length)
  for (let k = phi.length; k >= 1; k--) {
    const r = Math.max(-0.99, Math.min(0.99, c[k - 1]))
    u[k - 1] = Math.atanh(r)
    c = c.slice(0, k - 1).map((p, j) => (p + r * c[k - 2 - j]) / (1 - r * r))
  }
  return u
}

/** The parameters of a fitted ARMA model. */
export type ArmaFit = {
  ar: Vector
  ma: Vector
  mean: number
  sigma2: number
  logLikelihood: number
  /** Akaike's criterion −2 log L + 2k, with k = p + q + 1 (σ²) + 1 if the mean was estimated. */
  aic: number
  method: 'css' | 'exact'
}

/** Options for `armaFitSteps` and `fitArma`. */
export type ArmaFitOptions = {
  p: number
  q: number
  /** `css` minimises the conditional sum of squares; `exact` maximises the exact likelihood. Default `exact`. */
  method?: 'css' | 'exact'
  /** Subtract the sample mean (default true); otherwise μ = 0. */
  demean?: boolean
  /** Starting coefficients; default the CSS fit for `exact` and zeros for `css`. */
  start?: { ar?: VectorLike; ma?: VectorLike }
}

/**
 * An ARMA(p, q) fitter as a traceable algorithm: Nelder–Mead over the partial-autocorrelation coordinates of the AR
 * and MA polynomials (so every iterate is stationary and invertible), minimising either the conditional sum of
 * squares or the negative exact profile log-likelihood (σ² concentrated out). The mean is the sample mean. Each
 * state's `params` is the fit at the best vertex. `init` takes `{}`.
 */
export function armaFitSteps(x: VectorLike, options: ArmaFitOptions): Algorithm<void, FitState<ArmaFit>> {
  const xs = toVec(x, 'armaFitSteps')
  const { p, q, method = 'exact', demean = true } = options
  const mean = demean ? meanOf(tensor(xs)) : 0
  const split = (u: number[]) => ({ ar: fromPacf(u.slice(0, p)), ma: fromPacf(u.slice(p)).map((v) => -v) })
  const k = p + q + 1 + (demean ? 1 : 0)
  const n = xs.length
  const css = (ar: number[], ma: number[]) => {
    const e = residuals(xs, ar, ma, mean)
    let ss = 0
    for (let t = p; t < n; t++) ss += e[t] * e[t]
    return ss
  }
  const decode = (u: number[]): ArmaFit => {
    const { ar, ma } = split(u)
    if (method === 'css') {
      const ss = css(ar, ma)
      const m = n - p
      const sigma2 = ss / m
      const logLikelihood = -0.5 * m * (Math.log(2 * Math.PI * sigma2) + 1)
      return { ar: tensor(ar), ma: tensor(ma), mean, sigma2, logLikelihood, aic: -2 * logLikelihood + 2 * k, method }
    }
    const L = exactLikelihood(xs, ar, ma, mean)
    return {
      ar: tensor(ar),
      ma: tensor(ma),
      mean,
      sigma2: L.sigma2Hat,
      logLikelihood: L.logLikelihood,
      aic: -2 * L.logLikelihood + 2 * k,
      method,
    }
  }
  const objective =
    method === 'css'
      ? (u: number[]) => {
          const { ar, ma } = split(u)
          return css(ar, ma)
        }
      : (u: number[]) => {
          const { ar, ma } = split(u)
          return -exactLikelihood(xs, ar, ma, mean).logLikelihood
        }
  let start: { ar: number[]; ma: number[] }
  if (options.start)
    start = { ar: toVec(options.start.ar ?? [], 'armaFitSteps'), ma: toVec(options.start.ma ?? [], 'armaFitSteps') }
  else if (method === 'exact' && p + q > 0) {
    const c = run(armaFitSteps(xs, { p, q, method: 'css', demean }), undefined, 2000).params
    start = { ar: toFlat(c.ar), ma: toFlat(c.ma) }
  } else start = { ar: new Array(p).fill(0), ma: new Array(q).fill(0) }
  const u0 = [...toPacf(start.ar), ...toPacf(start.ma.map((v) => -v))]
  if (u0.length === 0) {
    // ARMA(0, 0): nothing to optimise; a one-dimensional dummy keeps the protocol uniform.
    return simplexFit(
      `arma-${method}`,
      () => 0,
      () => decode([]),
      [0],
    )
  }
  return simplexFit(`arma-${method}`, objective, decode, u0)
}

/** Fit an ARMA(p, q) model (see `armaFitSteps`), running at most `maxSteps` Nelder–Mead steps (default 4000). */
export function fitArma(
  x: VectorLike,
  options: ArmaFitOptions & { maxSteps?: number },
): ArmaFit & { converged: boolean; steps: number } {
  const s = run(armaFitSteps(x, options), undefined, options.maxSteps ?? 4000)
  return { ...s.params, converged: s.converged, steps: s.t }
}

/** Point forecasts with prediction intervals. */
export type Forecast = { mean: Vector; lower: Vector; upper: Vector; se: Vector; level: number }

/**
 * h-step forecasts of an ARMA model from the observed x: the recursion x̂_{n+h} = μ + Σ φ_i (x̃_{n+h−i} − μ) +
 * Σ_{j≥h} θ_j e_{n+h−j}, with x̃ the observed or forecast value and e the conditional residuals (future shocks zero),
 * and standard errors σ √(Σ_{j<h} ψ_j²) (Box, Jenkins & Reinsel, 2008, §5.2). Intervals are ±z·se at `level`
 * (default 0.95).
 */
export function forecastArma(
  x: VectorLike,
  model: ArmaSpec,
  horizon: number,
  { level = 0.95 }: { level?: number } = {},
): Forecast {
  const { ar, ma, sigma, mean } = parse(model, 'forecastArma')
  const xs = toVec(x, 'forecastArma')
  const e = residuals(xs, ar, ma, mean)
  const n = xs.length
  const ext = xs.map((v) => v - mean)
  const out: number[] = []
  for (let h = 1; h <= horizon; h++) {
    let v = 0
    for (let i = 0; i < ar.length; i++) {
      const idx = n + h - 1 - (i + 1)
      if (idx >= 0) v += ar[i] * ext[idx]
    }
    for (let j = h; j <= ma.length; j++) {
      const idx = n + h - 1 - j
      if (idx >= 0 && idx < n) v += ma[j - 1] * e[idx]
    }
    ext.push(v)
    out.push(v + mean)
  }
  const w = psi(ar, ma, horizon)
  const z = normalQuantile(0.5 + level / 2) as number
  let acc = 0
  const se = w.map((wj) => {
    acc += wj * wj
    return sigma * Math.sqrt(acc)
  })
  return {
    mean: tensor(out),
    lower: tensor(out.map((m, i) => m - z * se[i])),
    upper: tensor(out.map((m, i) => m + z * se[i])),
    se: tensor(se),
    level,
  }
}
