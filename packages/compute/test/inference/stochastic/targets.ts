/**
 * Target densities for the sampler tests, written here so the compute tests do not depend on `aifn-methods`'s targets:
 * a correlated Gaussian, Haario's banana and Neal's funnel, each normalised and with its gradient in closed form.
 */
import type { LogDensity } from 'aifn-compute/foundation/contracts'
import { MultivariateNormal, Normal } from 'aifn-compute/probability/distributions'
import { inverse } from 'aifn-compute/numerics/linalg'
import {
  add,
  exp,
  fromRows,
  get,
  mul,
  slice,
  sub,
  sum,
  tensor,
  toFlat,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'

const vec = (v: Vector) => toFlat(v)

/** N(mean, covariance), with gradient −Σ⁻¹(θ − μ). */
export function gaussianTarget(mean: number[], covariance: number[][]): LogDensity {
  const d = mean.length
  const law = MultivariateNormal(tensor(mean), { covariance: fromRows(covariance) })
  const P = toFlat(inverse(fromRows(covariance)))
  return {
    kind: 'log-density',
    name: 'gaussian',
    dim: d,
    normalised: true,
    logDensity: (theta: Value) => law.logProb(theta),
    grad: (theta) => {
      const x = vec(theta)
      return Array.from({ length: d }, (_, i) => -mean.reduce((s, m, j) => s + P[i * d + j] * (x[j] - m), 0))
    },
  }
}

/** The banana: x ~ N(0, a²), y | x ~ N(b(x² − a²), 1); Var y = 1 + 2b²a⁴. */
export function banana({ a = 1, b = 1 }: { a?: number; b?: number } = {}): LogDensity {
  return {
    kind: 'log-density',
    name: 'banana',
    dim: 2,
    normalised: true,
    logDensity: (theta: Value) => {
      const x = get(theta, 0)
      const y = get(theta, 1)
      return add(Normal(0, a).logProb(x), Normal(mul(b, sub(mul(x, x), a * a)), 1).logProb(y))
    },
    grad: (theta) => {
      const [x, y] = vec(theta)
      const r = y - b * (x * x - a * a)
      return [-x / (a * a) + 2 * b * x * r, -r]
    },
  }
}

/** Neal's funnel: v ~ N(0, scale²), xᵢ | v ~ N(0, eᵛ). */
export function funnel({ dim = 2, scale = 3 }: { dim?: number; scale?: number } = {}): LogDensity {
  return {
    kind: 'log-density',
    name: 'funnel',
    dim,
    normalised: true,
    logDensity: (theta: Value) =>
      add(
        Normal(0, scale).logProb(get(theta, 0)),
        sum(Normal(0, exp(mul(0.5, get(theta, 0)))).logProb(slice(theta, [1, null]))),
      ),
    grad: (theta) => {
      const x = vec(theta)
      const v = x[0]
      const g = new Array<number>(dim).fill(0)
      let q = 0
      for (let i = 1; i < dim; i++) {
        q += x[i] * x[i]
        g[i] = -x[i] * Math.exp(-v)
      }
      g[0] = -v / (scale * scale) + 0.5 * q * Math.exp(-v) - (dim - 1) / 2
      return g
    },
  }
}
