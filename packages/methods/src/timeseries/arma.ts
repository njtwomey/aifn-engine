/**
 * ARMA($p$, $q$) processes
 * $x_t - \mu = \sum_{i=1}^{p} \phi_i (x_{t-i} - \mu) + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}$,
 * $\varepsilon_t \sim \Gauss(0, \sigma^2)$: stationarity and invertibility from the roots of the AR and MA
 * polynomials, $\psi$ weights and the theoretical autocovariance, simulation, conditional and exact likelihoods,
 * fitting and forecasting.
 *
 * Sign conventions follow Box, Jenkins & Reinsel (2008) for $\phi$ (the AR polynomial is
 * $1 - \phi_1 z - \dots - \phi_p z^p$) and statsmodels for $\theta$ (the MA polynomial is
 * $1 + \theta_1 z + \dots + \theta_q z^q$). Coefficients are given as `VectorLike` values, $\phi_1$ first; series are
 * indexed from 0 in code and from 1 in the formulas.
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

/** An ARMA model. Omitted parts are empty (no AR or MA terms), $\sigma = 1$ and $\mu = 0$. */
export type ArmaSpec = {
  /** The AR coefficients $\phi_1, \dots, \phi_p$. */
  ar?: VectorLike
  /** The MA coefficients $\theta_1, \dots, \theta_q$. */
  ma?: VectorLike
  /** The innovation standard deviation $\sigma$. */
  sigma?: number
  /** The process mean $\mu$. */
  mean?: number
}

/** An `ArmaSpec` with its defaults filled in: the coefficients as arrays of numbers, `sigma` and `mean` as numbers. */
type Parsed = { ar: number[]; ma: number[]; sigma: number; mean: number }
/**
 * Read an `ArmaSpec` into arrays, filling the defaults (no terms, $\sigma = 1$, $\mu = 0$).
 *
 * @param m The model as given by the caller.
 * @param where The caller's name, for error messages.
 * @returns The coefficients as arrays of numbers, with `sigma` and `mean`.
 */
const parse = (m: ArmaSpec, where: string): Parsed => ({
  ar: m.ar ? toVec(m.ar, where) : [],
  ma: m.ma ? toVec(m.ma, where) : [],
  sigma: m.sigma ?? 1,
  mean: m.mean ?? 0,
})

/**
 * The roots of a lag polynomial: `roots` as a complex128 vector, `modulus` their moduli, and `minModulus` the smallest
 * of them (Infinity when the polynomial is constant and has no roots).
 */
export type LagRoots = { roots: Tensor; modulus: Vector; minModulus: number }

/**
 * The roots of the lag polynomial $c_0 + c_1 z + \dots + c_k z^k$; trailing zero coefficients lower the degree.
 *
 * @param c The coefficients $c_0, c_1, \dots$, constant term first.
 * @returns The roots with their moduli; empty, with `minModulus` Infinity, for a constant polynomial.
 */
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
 * The roots of the AR polynomial $1 - \phi_1 z - \dots - \phi_p z^p$ and the MA polynomial
 * $1 + \theta_1 z + \dots + \theta_q z^q$. The process is stationary (causal) when every AR root lies outside the unit
 * circle and invertible when every MA root does.
 *
 * @param model The model; only `ar` and `ma` are read.
 * @returns The roots of each polynomial, `ar` and `ma`, with their moduli and the smallest modulus.
 *
 * @example The roots of an ARMA(2, 1) model
 * const r = armaRoots({ ar: [0.5, 0.3], ma: [0.4] })
 * print('AR roots (re, im):', r.ar.roots)
 * print('AR moduli:', r.ar.modulus)
 * print('MA modulus (the root is -1/0.4):', r.ma.modulus)
 */
export function armaRoots(model: ArmaSpec): { ar: LagRoots; ma: LagRoots } {
  const { ar, ma } = parse(model, 'armaRoots')
  return { ar: lagRoots([1, ...ar.map((v) => -v)]), ma: lagRoots([1, ...ma]) }
}

/**
 * True when every root of $1 - \sum_i \phi_i z^i$ lies strictly outside the unit circle, so the AR part is stationary.
 *
 * @param ar The AR coefficients $\phi_1, \dots, \phi_p$.
 * @returns Whether the AR polynomial is stationary (true for no coefficients).
 *
 * @example A random walk is not stationary
 * print('φ = 0.5:', isStationary([0.5]))
 * print('φ = 1 (random walk):', isStationary([1]))
 * print('φ = [0.5, 0.6]:', isStationary([0.5, 0.6]))
 */
export const isStationary = (ar: VectorLike): boolean => armaRoots({ ar }).ar.minModulus > 1

/**
 * True when every root of $1 + \sum_j \theta_j z^j$ lies strictly outside the unit circle, so the MA part is
 * invertible.
 *
 * @param ma The MA coefficients $\theta_1, \dots, \theta_q$.
 * @returns Whether the MA polynomial is invertible (true for no coefficients).
 *
 * @example An MA(1) is invertible when $\lvert \theta \rvert < 1$
 * print('θ = 0.5:', isInvertible([0.5]))
 * print('θ = 2:', isInvertible([2]))
 */
export const isInvertible = (ma: VectorLike): boolean => armaRoots({ ma }).ma.minModulus > 1

/**
 * The weights $\psi_0, \dots, \psi_{\mathrm{count}-1}$ on arrays (shared with `sarima.ts`): the recursion of
 * `psiWeights`, for any AR polynomial, stationary or not.
 *
 * @param ar The AR coefficients $\phi_1, \dots, \phi_p$.
 * @param ma The MA coefficients $\theta_1, \dots, \theta_q$.
 * @param count How many weights to return, from $\psi_0 = 1$.
 * @returns The weights, $\psi_0$ first.
 */
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
 * The $\psi$ weights of the causal MA($\infty$) form $x_t - \mu = \sum_j \psi_j \varepsilon_{t-j}$: $\psi_0 = 1$,
 * $\psi_j = \theta_j + \sum_{i=1}^{\min(j, p)} \phi_i \psi_{j-i}$ with $\theta_j = 0$ for $j > q$ (Brockwell & Davis,
 * 1991, eq. 3.1.7). The recursion runs for any AR polynomial; the weights decay only for a stationary one.
 *
 * @param model The model; only `ar` and `ma` are read.
 * @param J The last index wanted.
 * @returns $\psi_0, \dots, \psi_J$ ($J + 1$ values).
 *
 * @example An AR(1) has $\psi_j = \phi^j$
 * print('AR(1), φ = 0.5:', psiWeights({ ar: [0.5] }, 5))
 * print('ARMA(1, 1), φ = 0.5, θ = 0.4:', psiWeights({ ar: [0.5], ma: [0.4] }, 5))
 */
export function psiWeights(model: ArmaSpec, J: number): Vector {
  const { ar, ma } = parse(model, 'psiWeights')
  return tensor(psi(ar, ma, J + 1))
}

/**
 * The theoretical autocovariance $\gamma(0), \dots, \gamma(\mathrm{maxLag})$ of a stationary ARMA process,
 * $\gamma(h) = \sigma^2 \sum_j \psi_j \psi_{j+h}$. The $\psi$ weights decay geometrically at the rate $\rho$, the
 * reciprocal of the smallest AR root modulus, so the sum is cut after about $J$ terms with $\rho^J < 10^{-17}$, plus
 * $q + 50$ (at most 200000). Throws `DomainError` for a non-stationary AR polynomial, whose autocovariance does not
 * exist.
 *
 * @param model The model: `ar`, `ma` and `sigma` are read (the mean does not enter).
 * @param maxLag The largest lag $h$ wanted.
 * @returns $\gamma(0), \dots, \gamma(\mathrm{maxLag})$.
 *
 * @example An AR(1) has $\gamma(h) = \phi^h \sigma^2 / (1 - \phi^2)$
 * print('γ(0), …, γ(3):', armaAutocovariance({ ar: [0.5] }, 3))
 * print('1 / (1 - φ²):', 1 / (1 - 0.5 ** 2))
 *
 * @example A unit root has no autocovariance
 * try {
 *   armaAutocovariance({ ar: [1] }, 3)
 * } catch (e) {
 *   print(e.message)
 * }
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

/**
 * The theoretical autocorrelation $\rho(h) = \gamma(h) / \gamma(0)$, $h = 0, \dots, \mathrm{maxLag}$, of a stationary
 * ARMA process (see `armaAutocovariance`, which throws for a non-stationary one).
 *
 * @param model The model; `ar` and `ma` are read (`sigma` cancels).
 * @param maxLag The largest lag $h$ wanted.
 * @returns $\rho(0) = 1, \rho(1), \dots, \rho(\mathrm{maxLag})$.
 *
 * @example An MA(1) is correlated at lag 1 only, by $\theta / (1 + \theta^2)$
 * print('MA(1), θ = 0.5:', armaAutocorrelation({ ma: [0.5] }, 3))
 * print('θ / (1 + θ²):', 0.5 / (1 + 0.5 ** 2))
 *
 * @example An AR(1) decays geometrically
 * print('AR(1), φ = 0.8:', armaAutocorrelation({ ar: [0.8] }, 4))
 */
export function armaAutocorrelation(model: ArmaSpec, maxLag: number): Vector {
  const g = toFlat(armaAutocovariance(model, maxLag))
  return tensor(g.map((v) => v / g[0]))
}

/** A simulated ARMA series. */
export type ArmaSimulation = {
  /** The series $x_1, \dots, x_n$ (after the burn-in). NaN from `divergedAt` on if the recursion overflowed. */
  x: Vector
  /** The innovations $\varepsilon_1, \dots, \varepsilon_n$ used for the kept values. */
  innovations: Vector
  /** Whether the AR polynomial is stationary; if not, the series grows without bound and is not a stationary sample. */
  stationary: boolean
  /** True if a value overflowed to a non-finite number. Nothing is clipped. */
  diverged: boolean
  /** The index in `x` of the first non-finite value (0 if it overflowed during the burn-in), or $-1$. */
  divergedAt: number
}

/**
 * Simulate $n$ values of an ARMA process from zero presample values and innovations, discarding `burn` values first
 * so the start-up transient is gone. Innovations are $\Gauss(0, \sigma^2)$ draws from `s`. Values are never clipped:
 * an explosive model is reported through `stationary: false` and, once it overflows, `diverged` (the site's copy
 * clipped at $\pm 10^6$ instead).
 *
 * @param s The random stream the innovations are drawn from; advanced in place.
 * @param model The model: `ar`, `ma`, `sigma` and `mean` are all read.
 * @param n The number of values returned.
 * @param options Simulation options.
 * @param options.burn Values simulated and discarded before the $n$ kept (default 200 for a stationary model and 0
 *   otherwise, so an explosive model is seen from its start).
 * @returns The series, its innovations, and whether the model is stationary and the series diverged.
 *
 * @example Simulate an AR(1) and check its moments
 * const sim = simulateArma(stream(1), { ar: [0.7], mean: 5 }, 2000)
 * print('first values:', toFlat(sim.x).slice(0, 4))
 * print('sample mean:', mean(sim.x))
 * print('sample variance:', variance(sim.x))
 * print('σ² / (1 - φ²):', 1 / (1 - 0.7 ** 2))
 *
 * @example An explosive model is reported, not clipped
 * const sim = simulateArma(stream(1), { ar: [1.5] }, 2000)
 * print('stationary:', sim.stationary)
 * print('diverged:', sim.diverged, 'at index', sim.divergedAt)
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
 * Conditional residuals $e_t = (x_t - \mu) - \sum_i \phi_i (x_{t-i} - \mu) - \sum_j \theta_j e_{t-j}$ for $t > p$,
 * with presample residuals zero; the first $p$ entries are 0. Their sum of squares is the conditional sum of squares
 * (CSS) of Box & Jenkins.
 *
 * @param x The observed series.
 * @param model The model: `ar`, `ma` and `mean` are read (`sigma` is not).
 * @returns The residuals, one per observation, the first $p$ zero.
 *
 * @example Under the true AR model the residuals are the innovations
 * const sim = simulateArma(stream(1), { ar: [0.7] }, 6)
 * print('residuals:  ', armaResiduals(sim.x, { ar: [0.7] }))
 * print('innovations:', sim.innovations)
 */
export function armaResiduals(x: VectorLike, model: ArmaSpec): Vector {
  const { ar, ma, mean } = parse(model, 'armaResiduals')
  return tensor(residuals(toVec(x, 'armaResiduals'), ar, ma, mean))
}

/**
 * `armaResiduals` on arrays (shared with `sarima.ts`).
 *
 * @param x The observed series.
 * @param ar The AR coefficients $\phi_1, \dots, \phi_p$.
 * @param ma The MA coefficients $\theta_1, \dots, \theta_q$.
 * @param mean The process mean $\mu$, subtracted from `x`.
 * @returns The conditional residuals, one per observation, the first $p$ zero.
 */
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

/**
 * The stationary covariance of the Harvey state, $\Pmat = \Tmat\Pmat\Tmat^\top + \rvec\rvec^\top$ (in units of
 * $\sigma^2$), by solving $(\Imat - \Tmat \otimes \Tmat) \operatorname{vec}\Pmat = \operatorname{vec}\rvec\rvec^\top$.
 *
 * @param T The $r \times r$ transition matrix $\Tmat$, as rows.
 * @param R The disturbance loading $\rvec$, $r$ values.
 * @returns $\Pmat$ as rows, or null when $\Imat - \Tmat \otimes \Tmat$ is singular (a unit root).
 */
function stationaryStateCovariance(T: number[][], R: number[]): number[][] | null {
  const r = R.length
  const t = tensor(T)
  const f = luFactor(sub(eye(r * r), kron(t, t)))
  if (f.singular) return null
  return toRows(reshape(luSolve(f, reshape(outer(tensor(R), tensor(R)), [r * r])), [r, r]))
}

/** The exact Gaussian log-likelihood of an ARMA model and its prediction errors. */
export type ArmaLikelihood = {
  /** The log-likelihood; $-\infty$ for a non-stationary AR polynomial. */
  logLikelihood: number
  /** One-step prediction errors $v_t = x_t - \expect[x_t \mid x_{<t}]$ (NaN when the likelihood is $-\infty$). */
  innovations: Vector
  /** Their variances $F_t$ (in units of $\sigma^2$), which fall to 1 as the start-up uncertainty resolves. */
  innovationVariance: Vector
  /**
   * The maximising $\sigma^2$ given $\phi$ and $\theta$ ($\sigma^2$ concentrated out),
   * $\frac{1}{n} \sum_t v_t^2 / F_t$.
   */
  sigma2Hat: number
}

/**
 * The exact Gaussian log-likelihood of $\xvec$ under an ARMA model, by the Kalman filter on Harvey's state-space form
 * (Harvey, 1989, §3.4; Durbin & Koopman, 2012, §3.4): state dimension $r = \max(p, q + 1)$, transition with $\phi$ in
 * its first column and ones on the superdiagonal, disturbance loading $(1, \theta_1, \dots, \theta_{r-1})$, started
 * from the stationary state covariance. With the prediction errors $v_t$ and their variances $\sigma^2 F_t$,
 * $\log L = -\frac{1}{2} \left(n \log 2\pi\sigma^2 + \sum_t \log F_t + \sum_t v_t^2 / (\sigma^2 F_t)\right)$. With
 * `model.sigma` omitted, $\sigma^2$ is concentrated out ($\hat\sigma^2$ in its place) and the profile likelihood is
 * returned. The log-likelihood is $-\infty$ for a non-stationary AR polynomial.
 *
 * @param x The observed series, $n$ values.
 * @param model The model: `ar`, `ma` and `mean` are read, and `sigma` if given.
 * @returns The log-likelihood, the prediction errors and their variances, and $\hat\sigma^2$.
 *
 * @example The likelihood peaks near the true coefficient
 * const { x } = simulateArma(stream(1), { ar: [0.7] }, 200)
 * for (const phi of [0.3, 0.7, 0.9]) print('φ =', phi, 'log L =', armaLogLikelihood(x, { ar: [phi] }).logLikelihood)
 *
 * @example The first prediction is the most uncertain
 * const L = armaLogLikelihood([1, 0.5, -0.2, 0.3], { ar: [0.5] })
 * print('F_t:', L.innovationVariance)
 * print('σ̂²:', L.sigma2Hat)
 */
export function armaLogLikelihood(x: VectorLike, model: ArmaSpec): ArmaLikelihood {
  const { ar, ma, mean } = parse(model, 'armaLogLikelihood')
  return exactLikelihood(toVec(x, 'armaLogLikelihood'), ar, ma, mean, model.sigma)
}

/**
 * `armaLogLikelihood` on arrays (shared with `sarima.ts`).
 *
 * @param x The observed series, $n$ values.
 * @param ar The AR coefficients $\phi_1, \dots, \phi_p$.
 * @param ma The MA coefficients $\theta_1, \dots, \theta_q$.
 * @param mean The process mean $\mu$, subtracted from `x`.
 * @param sigma The innovation standard deviation $\sigma$; when left out, $\sigma^2$ is concentrated out.
 * @returns The log-likelihood ($-\infty$, with NaN innovations, for a non-stationary AR polynomial or a singular
 *   filter step), the prediction errors and their variances, and $\hat\sigma^2$.
 */
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
 * Map unconstrained numbers to the coefficients of a stationary AR polynomial: $\tanh$ gives partial autocorrelations
 * $r_k$ in $(-1, 1)$, and the Levinson step $\phi_{kj} = \phi_{k-1,j} - r_k \phi_{k-1,k-j}$, $\phi_{kk} = r_k$, turns
 * them into coefficients (Jones, 1980; Monahan, 1984). Negated, the same map gives invertible MA coefficients.
 *
 * @param u The unconstrained coordinates, one per coefficient; $r_k = \tanh u_k$.
 * @returns The coefficients $\phi_1, \dots, \phi_p$ of a stationary AR polynomial, as many as `u` has entries.
 */
export function fromPacf(u: number[]): number[] {
  let phi: number[] = []
  for (let k = 0; k < u.length; k++) {
    const r = Math.tanh(u[k])
    phi = [...phi.map((p, j) => p - r * phi[k - 1 - j]), r]
  }
  return phi
}

/**
 * The inverse of `fromPacf` (the backward Levinson step), with partial autocorrelations capped at $\pm 0.99$ so that
 * a start on the boundary maps to finite coordinates.
 *
 * @param phi The AR coefficients $\phi_1, \dots, \phi_p$ (they should be stationary for the map to invert).
 * @returns The unconstrained coordinates $u_k = \operatorname{atanh} r_k$, one per coefficient.
 */
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
  /** The fitted AR coefficients $\phi_1, \dots, \phi_p$ (stationary). */
  ar: Vector
  /** The fitted MA coefficients $\theta_1, \dots, \theta_q$ (invertible). */
  ma: Vector
  /** The mean $\mu$: the sample mean, or 0 without `demean`. */
  mean: number
  /** The innovation variance $\hat\sigma^2$: the concentrated estimate (`exact`), or the CSS over $n - p$ (`css`). */
  sigma2: number
  /**
   * The exact profile log-likelihood (`exact`), or the conditional one
   * $-\frac{m}{2}(\log 2\pi\hat\sigma^2 + 1)$ with $m = n - p$ (`css`).
   */
  logLikelihood: number
  /** Akaike's criterion $-2 \log L + 2k$, with $k = p + q + 1$ ($\sigma^2$) $+ 1$ if the mean was estimated. */
  aic: number
  /** The objective that was optimised. */
  method: 'css' | 'exact'
}

/** Options for `armaFitSteps` and `fitArma`. */
export type ArmaFitOptions = {
  /** The AR order $p$. */
  p: number
  /** The MA order $q$. */
  q: number
  /** `css` minimises the conditional sum of squares; `exact` maximises the exact likelihood. Default `exact`. */
  method?: 'css' | 'exact'
  /** Subtract the sample mean (default true); otherwise $\mu = 0$. */
  demean?: boolean
  /** Starting coefficients; default the CSS fit for `exact` and zeros for `css`. */
  start?: { ar?: VectorLike; ma?: VectorLike }
}

/**
 * An ARMA($p$, $q$) fitter as a traceable algorithm: Nelder–Mead over the partial-autocorrelation coordinates of the AR
 * and MA polynomials (`fromPacf`, so every iterate is stationary and invertible), minimising either the conditional sum
 * of squares or the negative exact profile log-likelihood ($\sigma^2$ concentrated out). The mean is fixed at the
 * sample mean (or 0 without `demean`) and not optimised. Without `start`, the `exact` fit starts from a CSS fit of up
 * to 2000 steps, run when the algorithm is built. Each state's `params` is the fit at the best vertex. `init` takes no
 * start (`run(alg, undefined, steps)`).
 *
 * @param x The observed series.
 * @param options The orders, the objective, whether to demean and the starting coefficients (see `ArmaFitOptions`).
 * @returns The algorithm; its state is a `FitState` whose `params` is an `ArmaFit`.
 *
 * @example Watch a CSS fit approach the true coefficient
 * const { x } = simulateArma(stream(1), { ar: [0.7] }, 100)
 * const alg = armaFitSteps(x, { p: 1, q: 0, method: 'css' })
 * for (const steps of [1, 5, 40]) print(steps, 'steps: φ =', run(alg, undefined, steps).params.ar)
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

/**
 * Fit an ARMA($p$, $q$) model by running `armaFitSteps` until Nelder–Mead converges or `maxSteps` steps (default 4000)
 * have run. Check `converged` before trusting the result.
 *
 * @param x The observed series.
 * @param options The options of `armaFitSteps` (see `ArmaFitOptions`), and `maxSteps`, the most Nelder–Mead steps to
 *   run (default 4000).
 * @returns The fit at the best vertex, with whether the optimiser converged and the number of steps taken.
 *
 * @example Recover a known AR coefficient
 * const { x } = simulateArma(stream(1), { ar: [0.7] }, 100)
 * const fit = fitArma(x, { p: 1, q: 0 })
 * print('φ̂ =', fit.ar, '(true 0.7)')
 * print('σ̂² =', fit.sigma2, '(true 1)')
 * print('converged:', fit.converged, 'in', fit.steps, 'steps')
 */
export function fitArma(
  x: VectorLike,
  options: ArmaFitOptions & { maxSteps?: number },
): ArmaFit & { converged: boolean; steps: number } {
  const s = run(armaFitSteps(x, options), undefined, options.maxSteps ?? 4000)
  return { ...s.params, converged: s.converged, steps: s.t }
}

/**
 * Point forecasts with prediction intervals: `mean` the forecasts $\hat{x}_{n+h}$, `lower` and `upper` the interval
 * ends $\hat{x}_{n+h} \mp z \cdot \mathrm{se}_h$, `se` the standard errors, and `level` the coverage of the intervals
 * (such as 0.95), one entry per step $h = 1, \dots, \mathrm{horizon}$.
 */
export type Forecast = { mean: Vector; lower: Vector; upper: Vector; se: Vector; level: number }

/**
 * Forecasts of an ARMA model $h = 1, \dots, \mathrm{horizon}$ steps past the observed $x_1, \dots, x_n$: the recursion
 * $\hat{x}_{n+h} = \mu + \sum_i \phi_i (\tilde{x}_{n+h-i} - \mu) + \sum_{j \ge h} \theta_j e_{n+h-j}$, with
 * $\tilde{x}$ the observed or forecast value and $e$ the conditional residuals (future shocks zero), and standard
 * errors $\mathrm{se}_h = \sigma \sqrt{\sum_{j<h} \psi_j^2}$ (Box, Jenkins & Reinsel, 2008, §5.2). Intervals are
 * $\pm z \cdot \mathrm{se}_h$ with $z$ the normal quantile for `level`. The standard errors use the model's `sigma`
 * (default 1), not one estimated from `x`: pass $\sqrt{\hat\sigma^2}$ from a fit.
 *
 * @param x The observed series the forecasts continue.
 * @param model The model: `ar`, `ma`, `sigma` and `mean` are all read.
 * @param horizon The number of steps ahead to forecast.
 * @param options Forecast options.
 * @param options.level The coverage of the prediction intervals (default 0.95).
 * @returns The forecasts with their standard errors and intervals.
 *
 * @example Forecasts revert to the mean and the interval widens
 * const { x } = simulateArma(stream(1), { ar: [0.7], mean: 10 }, 100)
 * const f = forecastArma(x, { ar: [0.7], mean: 10 }, 4)
 * print('last value:', toFlat(x)[99])
 * print('forecast:', f.mean)
 * print('95% interval:', f.lower, f.upper)
 *
 * @example Forecast from a fitted model, with an 80% interval
 * const { x } = simulateArma(stream(1), { ar: [0.7] }, 100)
 * const fit = fitArma(x, { p: 1, q: 0 })
 * const f = forecastArma(x, { ar: fit.ar, mean: fit.mean, sigma: Math.sqrt(fit.sigma2) }, 3, { level: 0.8 })
 * print('forecast:', f.mean)
 * print('se:', f.se)
 * print('80% interval:', f.lower, f.upper)
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
