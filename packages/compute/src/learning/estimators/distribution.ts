/**
 * Predictive distributions: every `predictive(x)` returns a contract `Distribution` (`kind: 'distribution'`), usually a
 * family of `aifn-compute/probability/distributions`. This file holds the helpers that read class probabilities and
 * expectations from any distribution, and the predictive constructors the estimators use.
 */

import type { AnyUnivariate, Distribution, Size, Univariate } from 'aifn-compute/foundation/contracts'
import { dense, fromData, isTensor, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { gaussHermite } from 'aifn-compute/numerics/quadrature'
import { normalCdf } from 'aifn-compute/numerics/special'
import { Bernoulli, Categorical, Normal } from 'aifn-compute/probability/distributions'
import { sizeOf } from './util'
import { DomainError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { AnyUnivariate, Distribution } from 'aifn-compute/foundation/contracts'

/** A distribution over classes 0 … K−1: `aifn-compute/probability/distributions`' Bernoulli (K = 2) or Categorical. */
export interface ClassDistribution extends Distribution {
  readonly name: 'Bernoulli' | 'Categorical'
}

/** An untraced `Value` as a tensor (a number becomes a scalar tensor). */
export function asTensor(v: Value): Tensor {
  if (typeof v === 'number') return fromData(Float64Array.of(v), [])
  if (isTensor(v)) return v
  throw new DomainError('asTensor', 'asTensor: a traced value; evaluate distributions outside a gradient tape here')
}

/** True when `d` is univariate (scalar events) with a CDF and a quantile function. */
export function isUnivariate(d: Distribution): d is AnyUnivariate {
  const u = d as Partial<AnyUnivariate>
  return d.eventShape.length === 0 && typeof u.cdf === 'function' && typeof u.quantile === 'function'
}

/** True when `d` is a Bernoulli or categorical law (by name). */
export function isClassDistribution(d: Distribution): d is ClassDistribution {
  return d.name === 'Bernoulli' || d.name === 'Categorical'
}

/**
 * Class probabilities [N, K] from a Bernoulli (K = 2: [1 − p, p]) or categorical distribution over a batch of N, read
 * through `logProb` so that either parameterisation (probabilities or logits) works. The number of classes comes from
 * the support. Throws for any other distribution.
 */
export function classProbabilities(d: Distribution): Tensor {
  if (!isClassDistribution(d))
    throw new DomainError('classProbabilities', `classProbabilities: ${d.name} has no class probabilities`)
  const n = sizeOf(d.batchShape)
  const upper = d.support.type === 'integers' ? d.support.upper : undefined
  const k = d.name === 'Bernoulli' ? 2 : typeof upper === 'number' ? upper + 1 : NaN
  if (!Number.isInteger(k)) throw new DomainError('classProbabilities', 'classProbabilities: unknown number of classes')
  const out = new Float64Array(n * k)
  for (let c = 0; c < k; c++) {
    const lp = dense.data(asTensor(d.logProb(c)))
    for (let i = 0; i < n; i++) out[i * k + c] = Math.exp(lp.length === 1 ? lp[0] : lp[i])
  }
  return fromData(out, [n, k])
}

// ── Predictive constructors ──────────────────────────────────────────────────────────────────────────────────────

/** A batch of Gaussians with means `loc` and standard deviations `scale` (same shape): `Normal(loc, scale)`. */
export function gaussianPredictive(loc: Tensor, scale: Tensor): Univariate<Tensor> {
  return Normal(loc, scale)
}

/** Bernoulli laws with P(y = 1) = `probs`: `Bernoulli(probs)`. */
export function bernoulliPredictive(probs: Tensor): Univariate<Tensor> {
  return Bernoulli(probs)
}

/** Categorical laws with class probabilities `probs` [..., K]: `Categorical(probs)`. */
export function categoricalPredictive(probs: Tensor): Univariate<Tensor> {
  return Categorical(probs)
}

// ── Expectations ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Nodes of the probabilists' Gauss–Hermite rule `expectation` uses. */
const HERMITE_NODES: Size = 32

let hermite: { nodes: Float64Array; weights: Float64Array } | undefined

/** The rule with weights normalised to sum to 1, so Σ wⱼ g(zⱼ) ≈ E g(Z) for Z ~ N(0, 1) (computed once). */
function normalRule(): { nodes: Float64Array; weights: Float64Array } {
  if (hermite) return hermite
  const rule = gaussHermite(HERMITE_NODES, { probabilists: true })
  const w = dense.data(rule.weights)
  const total = w.reduce((a, b) => a + b, 0)
  hermite = { nodes: dense.data(rule.nodes), weights: Float64Array.from(w, (v) => v / total) }
  return hermite
}

/**
 * E[f(y)] under `d`, elementwise over its batch (shape `batchShape`). Without `f`, the mean. For class distributions,
 * the finite sum Σₖ f(k) pₖ. For other univariate laws, 32-point probabilists' Gauss–Hermite quadrature on normal
 * scores (Golub and Welsch, 1969, "Calculation of Gauss quadrature rules", Math. Comp. 23; the rule from
 * `aifn-compute/numerics/quadrature`): E f(y) = E f(Q(Φ(Z))) with Z ~ N(0, 1) and Q the quantile function, exact for a
 * Gaussian and f a polynomial of degree < 64 up to the skipped outermost nodes (|z| > 8.1, total weight below 1e-15).
 */
export function expectation(d: Distribution, f?: (y: number) => number): Tensor {
  if (!f) return asTensor(d.mean())
  if (isClassDistribution(d)) {
    const p = classProbabilities(d)
    const [n, k] = p.shape
    const probs = dense.data(p)
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) out[i] += f(c) * probs[i * k + c]
    return fromData(out, d.batchShape)
  }
  if (!isUnivariate(d))
    throw new DomainError('expectation', 'expectation: needs a class distribution or a univariate one with a quantile')
  const { nodes, weights } = normalRule()
  const n = sizeOf(d.batchShape)
  const out = new Float64Array(n)
  for (let j = 0; j < nodes.length; j++) {
    // Skip the outermost nodes, where Φ(z) or 1 − Φ(z) is below the float64 resolution of 1 (their weights are
    // below 1e-17), so that quantiles are never asked for at 0 or 1.
    if (normalCdf(-Math.abs(nodes[j])) < Number.EPSILON) continue
    const q = dense.data(asTensor(d.quantile(normalCdf(nodes[j]))))
    for (let i = 0; i < n; i++) out[i] += weights[j] * f(q.length === 1 ? q[0] : q[i])
  }
  return fromData(out, d.batchShape)
}
