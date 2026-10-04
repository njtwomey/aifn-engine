/**
 * Standard target log-densities (`kind: 'log-density'`) for samplers and variational inference: a banana (twisted
 * Gaussian), a Gaussian, an isotropic Gaussian mixture and Neal's funnel. Each log-density is written with the
 * registered distributions of `aifn-compute/probability/distributions`, so it is differentiable; `grad` gives the closed form,
 * and `truth` the known moments where they exist.
 */

import type { LogDensity, Value, VectorLike } from 'aifn-compute/foundation/contracts'
import {
  add,
  concat,
  exp,
  fromData,
  get,
  isTensor,
  logsumexp,
  mul,
  slice,
  sub,
  sum,
  toFlat,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { inverse } from 'aifn-compute/numerics/linalg'
import { MultivariateNormal, Normal } from 'aifn-compute/probability/distributions'
import { transformLogDensity, type TransformedLogDensity } from 'aifn-compute/probability/bijectors'
import type { LogDensityInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A vector input as a fresh Float64Array. */
function vec(v: Tensor | VectorLike): Float64Array {
  if (isTensor(v)) {
    const n = v.shape.length === 0 ? 1 : v.shape[0]
    return Float64Array.from({ length: n }, (_, i) => Number(v.data[v.offset + i * (v.strides[0] ?? 0)]))
  }
  return Float64Array.from(v as ArrayLike<number>)
}

const matrixOf = (rows: readonly (readonly number[])[]): Tensor =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])

/**
 * The banana (twisted Gaussian of Haario, Saksman and Tamminen, 1999, "Adaptive proposal distribution for random walk
 * Metropolis algorithm", Computational Statistics 14) in two dimensions: x ~ N(0, a²) and y | x ~ N(b(x² − a²), 1), so
 * the mean is (0, 0), Var x = a², Var y = 1 + 2b²a⁴, and b bends the ridge (b = 0 is a Gaussian). Default a = 1, b = 1.
 */
export function banana(options: { a?: number; b?: number } = {}): LogDensity & { mean: Tensor; variance: Tensor } {
  const { a = 1, b = 1 } = options
  const mean = fromData(Float64Array.of(0, 0), [2])
  const variance = fromData(Float64Array.of(a * a, 1 + 2 * b * b * a ** 4), [2])
  return {
    kind: 'log-density',
    name: 'banana',
    dim: 2,
    normalised: true,
    logDensity: (theta: Value) => {
      const x = get(theta, 0)
      const y = get(theta, 1)
      const ridge = mul(b, sub(mul(x, x), a * a))
      return add(Normal(0, a).logProb(x), Normal(ridge, 1).logProb(y))
    },
    grad: (theta) => {
      const [x, y] = vec(theta)
      const r = y - b * (x * x - a * a)
      return [-x / (a * a) + 2 * b * x * r, -r]
    },
    mean,
    variance,
    truth: { mean },
  }
}

/**
 * A normalised multivariate Gaussian target N(mean, covariance): log π(θ) = −½(θ − μ)ᵀΣ⁻¹(θ − μ) − ½ log|2πΣ|, with
 * gradient −Σ⁻¹(θ − μ). `precision` is Σ⁻¹ (d × d).
 */
export function gaussianTarget(
  mean: Tensor | VectorLike,
  covariance: Tensor | readonly (readonly number[])[],
): LogDensity & { mean: Tensor; covariance: Tensor; precision: Tensor } {
  const mu = vec(mean)
  const d = mu.length
  const cov = isTensor(covariance) ? covariance : matrixOf(covariance as readonly (readonly number[])[])
  if (cov.shape.length !== 2 || cov.shape[0] !== d || cov.shape[1] !== d)
    throw new ShapeError('gaussianTarget', `gaussianTarget: covariance must be ${d}×${d}`)
  const law = MultivariateNormal(fromData(mu, [d]), { covariance: cov })
  const precision = inverse(cov)
  const P = Float64Array.from(toFlat(precision))
  const meanT = fromData(Float64Array.from(mu), [d])
  return {
    kind: 'log-density',
    name: 'gaussian',
    dim: d,
    normalised: true,
    logDensity: (theta: Value) => law.logProb(theta),
    grad: (theta) => {
      const x = vec(theta)
      const g = new Float64Array(d)
      for (let i = 0; i < d; i++) {
        let s = 0
        for (let j = 0; j < d; j++) s += P[i * d + j] * (x[j] - mu[j])
        g[i] = -s
      }
      return g
    },
    mean: meanT,
    covariance: cov,
    precision,
    truth: { mean: meanT, cov },
  }
}

/**
 * An isotropic Gaussian mixture Σₖ wₖ N(θ | mₖ, σ²I) (normalised): the log of the weighted sum of a batch of K
 * multivariate normals. `means` is K × d; `weights` default to equal.
 */
export function gaussianMixtureTarget(
  means: readonly (readonly number[])[],
  sd: number,
  weights?: readonly number[],
): LogDensity & { means: Tensor; weights: Tensor } {
  const K = means.length
  const d = means[0].length
  const total = weights?.reduce((a, b) => a + b, 0) ?? K
  const w = weights ? weights.map((v) => v / total) : new Array<number>(K).fill(1 / K)
  const logW = fromData(Float64Array.from(w, Math.log), [K])
  const components = MultivariateNormal(matrixOf(means), {
    covariance: fromData(
      Float64Array.from({ length: d * d }, (_, e) => (e % (d + 1) === 0 ? sd * sd : 0)),
      [d, d],
    ),
  })
  const componentLogs = (theta: Value) => add(logW, components.logProb(theta))
  const meanOf = Float64Array.from({ length: d }, (_, i) => w.reduce((a, wk, k) => a + wk * means[k][i], 0))
  return {
    kind: 'log-density',
    name: 'gaussian-mixture',
    dim: d,
    normalised: true,
    logDensity: (theta: Value) => logsumexp(componentLogs(theta)),
    grad: (theta) => {
      const x = vec(theta)
      const logs = vec(componentLogs(fromData(x, [d])) as Tensor)
      const m = Math.max(...logs)
      const r = logs.map((l) => Math.exp(l - m))
      const z = r.reduce((a, b) => a + b, 0)
      const g = new Float64Array(d)
      means.forEach((mk, k) => {
        for (let i = 0; i < d; i++) g[i] -= ((r[k] / z) * (x[i] - mk[i])) / (sd * sd)
      })
      return g
    },
    means: matrixOf(means),
    weights: fromData(Float64Array.from(w), [K]),
    truth: { mean: fromData(meanOf, [d]) },
  }
}

/**
 * Neal's funnel (Neal, 2003, "Slice sampling", Annals of Statistics 31(3), §8) in d dimensions: v ~ N(0, s²) and
 * xᵢ | v ~ N(0, eᵛ) for i = 1 … d − 1, with θ = (v, x₁, …). Its neck (v ≪ 0) needs small steps and its mouth large
 * ones, so fixed-step HMC diverges there. Default d = 2, s = 3.
 */
export function funnel(options: { dim?: number; scale?: number } = {}): LogDensity {
  const { dim = 2, scale = 3 } = options
  return {
    kind: 'log-density',
    name: 'funnel',
    dim,
    normalised: true,
    logDensity: (theta: Value) => {
      const v = get(theta, 0)
      const rest = slice(theta, [1, null])
      return add(Normal(0, scale).logProb(v), sum(Normal(0, exp(mul(0.5, v))).logProb(rest)))
    },
    grad: (theta) => {
      const x = vec(theta)
      const v = x[0]
      const g = new Float64Array(dim)
      let q = 0
      for (let i = 1; i < dim; i++) {
        q += x[i] * x[i]
        g[i] = -x[i] * Math.exp(-v)
      }
      g[0] = -v / (scale * scale) + 0.5 * q * Math.exp(-v) - (dim - 1) / 2
      return g
    },
    truth: { mean: fromData(new Float64Array(dim), [dim]) },
  }
}

/**
 * Neal's funnel in its non-centred parameterisation: θ = (v, z₁, …) with v ~ N(0, s²) and zᵢ ~ N(0, 1) independent.
 * It is `transformLogDensity(funnel, T)` for the map T(v, z) = (v, z e^{v/2}) onto the funnel's (v, x), whose
 * log |det J_T| = (d − 1) v/2; `toOriginal` maps a draw back to the funnel and
 * `fromOriginal` the other way. The density is an axis-aligned Gaussian, so one step size suits it everywhere and HMC
 * does not diverge (Papaspiliopoulos, Roberts & Sköld, 2007, "A general framework for the parametrization of
 * hierarchical models", Statistical Science 22(1)). Default d = 2, s = 3.
 */
export function nonCentredFunnel(options: { dim?: number; scale?: number } = {}): TransformedLogDensity {
  const { dim = 2, scale = 3 } = options
  const scaleRest = (t: Value, sign: 1 | -1) =>
    concat([slice(t, [0, 1]), mul(slice(t, [1, null]), exp(mul(0.5 * sign, get(t, 0))))])
  const nc = transformLogDensity(funnel({ dim, scale }), {
    name: 'non-centring',
    forward: (u) => scaleRest(u, 1),
    inverse: (theta) => scaleRest(theta, -1),
    logAbsDetJacobian: (u) => mul((dim - 1) / 2, get(u, 0)),
  })
  return {
    ...nc,
    name: 'non-centred funnel',
    // The closed form in u: the density is N(0, s²) × N(0, 1)^(d − 1).
    grad: (theta) => {
      const x = vec(theta)
      const g = new Float64Array(dim)
      g[0] = -x[0] / (scale * scale)
      for (let i = 1; i < dim; i++) g[i] = -x[i]
      return g
    },
    truth: { mean: fromData(new Float64Array(dim), [dim]) },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const logDensity = definer<LogDensityInfo>('log-density', 'data/targets')

logDensity(
  {
    key: 'banana',
    name: 'Banana',
    summary: 'A Gaussian bent into a crescent by a quadratic shift of one coordinate, with exact moments.',
    params: space({ a: real(0.1, 5, { default: 1 }), b: real(0, 2, { default: 1 }) }),
    dim: 2,
    truth: true,
    notes: ['metropolis-hastings', 'hamiltonian-monte-carlo'],
  },
  banana,
)

logDensity(
  {
    key: 'gaussianTarget',
    name: 'Gaussian target',
    summary: 'A multivariate Gaussian N(μ, Σ); the mean and covariance are required arguments.',
    params: space({}),
    dim: null,
    truth: true,
    notes: ['markov-chain-monte-carlo'],
  },
  gaussianTarget,
)

logDensity(
  {
    key: 'gaussianMixtureTarget',
    name: 'Gaussian mixture target',
    summary: 'An isotropic Gaussian mixture, a multimodal test for samplers; the means and sd are required arguments.',
    params: space({}),
    dim: null,
    truth: false,
    notes: ['markov-chain-monte-carlo'],
  },
  gaussianMixtureTarget,
)

logDensity(
  {
    key: 'funnel',
    name: "Neal's funnel",
    summary: 'A hierarchical funnel whose scale varies over orders of magnitude, hard for fixed-step samplers.',
    params: space({ dim: int(2, 20, { default: 2 }), scale: real(0.5, 10, { default: 3 }) }),
    dim: null,
    truth: false,
    notes: ['hamiltonian-monte-carlo', 'reparameterisation-trick'],
  },
  funnel,
)

logDensity(
  {
    key: 'nonCentredFunnel',
    name: 'Non-centred funnel',
    summary: 'The funnel in its non-centred parametrisation, an isotropic Gaussian, with the map back to the funnel.',
    params: space({ dim: int(2, 20, { default: 2 }), scale: real(0.5, 10, { default: 3 }) }),
    dim: null,
    truth: false,
    notes: ['hamiltonian-monte-carlo', 'reparameterisation-trick'],
  },
  nonCentredFunnel,
)
