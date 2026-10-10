/**
 * Predictive distributions: every `predictive(x)` returns a contract `Distribution` (`kind: 'distribution'`), usually a
 * family of `aifn-compute/probability/distributions`. This file holds the helpers that read class probabilities and
 * expectations from any distribution, and the predictive constructors the estimators use.
 *
 * Expectations of a continuous univariate law are computed on normal scores:
 * $\expect f(y) = \expect f(Q(\Phi(Z)))$ for $Z \sim \Gauss(0, 1)$, with $Q$ the law's quantile function and $\Phi$
 * the standard normal CDF, by Gauss–Hermite quadrature (Golub and Welsch, 1969). The helpers work on untraced values:
 * they are for evaluation and figures, not for gradients.
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

/**
 * A distribution over classes $0, \dots, K - 1$: `aifn-compute/probability/distributions`' Bernoulli ($K = 2$) or
 * Categorical.
 */
export interface ClassDistribution extends Distribution {
  /** The family's name, by which `isClassDistribution` recognises it. */
  readonly name: 'Bernoulli' | 'Categorical'
}

/**
 * An untraced `Value` as a tensor (a number becomes a scalar tensor). Throws `DomainError` for a traced value (inside
 * `grad` and the like).
 *
 * @param v A number or a tensor, as a distribution's methods return them.
 * @returns `v` itself when it is a tensor, otherwise a new scalar float64 tensor.
 *
 * @example Numbers and tensors
 * print('number:', asTensor(2).shape, asTensor(2))
 * print('tensor:', asTensor(tensor([1, 2])))
 */
export function asTensor(v: Value): Tensor {
  if (typeof v === 'number') return fromData(Float64Array.of(v), [])
  if (isTensor(v)) return v
  throw new DomainError('asTensor', 'asTensor: a traced value; evaluate distributions outside a gradient tape here')
}

/**
 * True when `d` is univariate (scalar events) with a CDF and a quantile function, which `expectation` and the target
 * transforms need.
 *
 * @param d The distribution.
 * @returns Whether `d` has an empty event shape and `cdf` and `quantile` methods.
 *
 * @example Laws with and without a quantile function
 * print('Gaussian:', isUnivariate(gaussianPredictive(tensor([0]), tensor([1]))))
 * print('categorical:', isUnivariate(categoricalPredictive(tensor([[0.2, 0.8]]))))
 * print('a CDF only:', isUnivariate({ eventShape: [], cdf: (y) => y }))
 */
export function isUnivariate(d: Distribution): d is AnyUnivariate {
  const u = d as Partial<AnyUnivariate>
  return d.eventShape.length === 0 && typeof u.cdf === 'function' && typeof u.quantile === 'function'
}

/**
 * True when `d` is a Bernoulli or categorical law (by its `name`).
 *
 * @param d The distribution.
 * @returns Whether `d` is a `ClassDistribution`.
 *
 * @example Class laws and others
 * print('Bernoulli:', isClassDistribution(bernoulliPredictive(tensor([0.3]))))
 * print('Gaussian:', isClassDistribution(gaussianPredictive(tensor([0]), tensor([1]))))
 */
export function isClassDistribution(d: Distribution): d is ClassDistribution {
  return d.name === 'Bernoulli' || d.name === 'Categorical'
}

/**
 * Class probabilities, an $N \times K$ matrix, from a Bernoulli ($K = 2$, rows $[1 - p, p]$) or categorical
 * distribution over a batch of $N$, read through `logProb` so that either parameterisation (probabilities or logits)
 * works. The number of classes comes from the support. Throws `DomainError` for any other distribution, or a support
 * that does not give the number of classes.
 *
 * @param d A Bernoulli or categorical distribution; a scalar batch counts as $N = 1$.
 * @returns A new $N \times K$ float64 matrix whose rows sum to 1.
 *
 * @example A Bernoulli batch as two-column probabilities
 * print(classProbabilities(bernoulliPredictive(tensor([0.1, 0.75]))))
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

/**
 * A batch of Gaussians with means `loc` and standard deviations `scale` (same shape): `Normal(loc, scale)` of
 * `aifn-compute/probability/distributions`.
 *
 * @param loc The means $\mu$, one per example.
 * @param scale The standard deviations $\sigma > 0$, of the same shape.
 * @returns The batch of normal laws $\Gauss(\mu, \sigma^2)$.
 *
 * @example A Gaussian predictive and its 95% interval
 * const d = gaussianPredictive(tensor([0, 10]), tensor([1, 2]))
 * print('mean:', d.mean())
 * print('2.5%:', d.quantile(0.025))
 * print('97.5%:', d.quantile(0.975))
 */
export function gaussianPredictive(loc: Tensor, scale: Tensor): Univariate<Tensor> {
  return Normal(loc, scale)
}

/**
 * Bernoulli laws with $\Pr(y = 1) = p$: `Bernoulli(probs)` of `aifn-compute/probability/distributions`.
 *
 * @param probs The probabilities $p$ of class 1, one per example, in $[0, 1]$.
 * @returns The batch of Bernoulli laws.
 *
 * @example The mean of a Bernoulli is its probability
 * const d = bernoulliPredictive(tensor([0.2, 0.9]))
 * print('mean:', d.mean())
 * print('P(y = 0):', d.prob(0))
 */
export function bernoulliPredictive(probs: Tensor): Univariate<Tensor> {
  return Bernoulli(probs)
}

/**
 * Categorical laws with class probabilities `probs`: `Categorical(probs)` of `aifn-compute/probability/distributions`.
 *
 * @param probs The class probabilities, the last axis over the $K$ classes (each row summing to 1).
 * @returns The batch of categorical laws over $0, \dots, K - 1$.
 *
 * @example Two examples over three classes
 * const d = categoricalPredictive(tensor([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]]))
 * print('P(class 1):', d.prob(1))
 * print('class probabilities:', classProbabilities(d))
 */
export function categoricalPredictive(probs: Tensor): Univariate<Tensor> {
  return Categorical(probs)
}

// ── Expectations ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Nodes of the probabilists' Gauss–Hermite rule `expectation` uses. */
const HERMITE_NODES: Size = 32

let hermite: { nodes: Float64Array; weights: Float64Array } | undefined

/**
 * The probabilists' Gauss–Hermite rule of `HERMITE_NODES` nodes with weights normalised to sum to 1, so that
 * $\sum_j w_j g(z_j) \approx \expect g(Z)$ for $Z \sim \Gauss(0, 1)$. Computed once and cached.
 *
 * @returns The nodes $z_j$ and the normalised weights $w_j$.
 */
function normalRule(): { nodes: Float64Array; weights: Float64Array } {
  if (hermite) return hermite
  const rule = gaussHermite(HERMITE_NODES, { probabilists: true })
  const w = dense.data(rule.weights)
  const total = w.reduce((a, b) => a + b, 0)
  hermite = { nodes: dense.data(rule.nodes), weights: Float64Array.from(w, (v) => v / total) }
  return hermite
}

/**
 * $\expect[f(y)]$ under `d`, elementwise over its batch (shape `batchShape`). Without `f`, the mean (`d.mean()`). For
 * class distributions, the finite sum $\sum_k f(k) p_k$. For other univariate laws, 32-point probabilists'
 * Gauss–Hermite quadrature on normal scores (Golub and Welsch, 1969, "Calculation of Gauss quadrature rules", Math.
 * Comp. 23; the rule from `aifn-compute/numerics/quadrature`): $\expect f(y) = \expect f(Q(\Phi(Z)))$ with
 * $Z \sim \Gauss(0, 1)$ and $Q$ the quantile function, exact for a Gaussian and $f$ a polynomial of degree below 64 up
 * to the skipped outermost nodes ($\lvert z \rvert > 8.1$, total weight below $10^{-15}$). Throws `DomainError` for a
 * law that is neither.
 *
 * @param d The distribution: a class distribution, or a univariate law with a quantile function.
 * @param f The function of $y$ whose expectation is taken. Left out, the mean.
 * @returns A new tensor of shape `batchShape` (the mean as `d` returns it when `f` is left out).
 *
 * @example The second moment of a Gaussian, and the expected cost of a class
 * const g = gaussianPredictive(tensor([1]), tensor([2]))
 * print('E[y^2] = 1 + 4:', expectation(g, (y) => y * y))
 * const c = categoricalPredictive(tensor([[0.2, 0.5, 0.3]]))
 * print('E[cost] with costs 0, 10, 100:', expectation(c, (k) => [0, 10, 100][k]))
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
