/**
 * Kullback–Leibler divergences KL(p ‖ q) = E_p[log p(x) − log q(x)] in closed form for pairs of families, dispatched
 * on the families' names, and a Monte Carlo estimate for any pair. Closed forms are compositions of primitives, so
 * they are differentiable in both distributions' parameters; each batches over the broadcast of the two batches.
 *
 * Sources: Kullback and Leibler (1951), "On information and sufficiency", Ann. Math. Statist. 22(1); the closed forms
 * for the normal, gamma, Dirichlet and Wishart families as in Penny (2001), "KL-divergences of Normal, Gamma,
 * Dirichlet and Wishart densities" (Wellcome Department of Imaging Neuroscience, UCL), and for exponential families in
 * general through the log-partition function (Nielsen and Nock 2010, "Entropies and cross-entropies of exponential
 * families", ICIP).
 */

import { AifnError } from 'aifn-compute/foundation/errors'
import { definer, entries, type Entry, type KlRuleInfo, type Spec } from 'aifn-compute/foundation/registry'
import { choleskyLogDet, solveTriangular } from 'aifn-compute/numerics/linalg'
import type { Stream } from 'aifn-compute/foundation/random'
import { digamma, logBeta, logGamma, logSigmoid, logSoftmax, sigmoid } from 'aifn-compute/numerics/special'
import {
  add,
  div,
  exp,
  log,
  log1p,
  mean,
  mul,
  neg,
  shapeOfValue,
  square,
  sub,
  sum,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { MultivariateNormal } from './multivariate'
import type { Distribution } from './types'
import { guard, mask, outside, sumLast } from './util'
import { xlogy } from 'aifn-compute/numerics/special'

/** A closed-form KL divergence between two distributions of given families. */
export type KlRule = (p: Distribution, q: Distribution) => Value

const define = definer<KlRuleInfo>('kl-rule', 'probability/distributions')
const NOTES = ['kullback-leibler-divergence']
const CITE = ['kullback1951']

/** A closed form for KL(p ‖ q) between two families of the same name (keyed `p|q`, e.g. `Normal|Normal`). */
function rule(family: string, f: KlRule, spec: Partial<Spec<KlRuleInfo>> = {}): Entry<KlRule, KlRuleInfo> {
  return define(
    {
      key: `${family}|${family}`,
      name: `KL(${family} ‖ ${family})`,
      p: family,
      q: family,
      notes: NOTES,
      cite: CITE,
      ...spec,
    },
    f,
  )
}

/** Whether `kl` has a closed form for the pair (p's family, q's family). */
export function hasKl(p: Distribution, q: Distribution): boolean {
  return `${p.name}|${q.name}` in klRegistry
}

/**
 * KL(p ‖ q) in nats, in closed form, with the batch shape of the broadcast batches. Pairs covered: Normal, LogNormal,
 * MultivariateNormal (unbatched), Beta, Gamma, Exponential, Poisson, Dirichlet, Categorical and Bernoulli (each with
 * itself); the table is `klRegistry`. Other pairs throw; use `klMonteCarlo`, or `klNumerical` / `klAuto` for
 * continuous univariate pairs.
 */
export function kl(p: Distribution, q: Distribution): Value {
  const r = klRegistry[`${p.name}|${q.name}`]
  if (!r) throw new AifnError('kl', `kl: no closed form for ${p.name} ‖ ${q.name}; use klMonteCarlo`)
  return r(p, q)
}

/**
 * A Monte Carlo estimate of KL(p ‖ q): the mean of log p(x) − log q(x) over n draws from p (stream `s`). Unbiased; its
 * standard error falls as 1/√n. Not differentiable in p's parameters (the draws are not reparameterised).
 */
export function klMonteCarlo(s: Stream, p: Distribution, q: Distribution, n: number): Value {
  const x = p.sample(s, { shape: [n] })
  const diff = sub(p.logProb(x) as Value, q.logProb(x) as Value)
  return shapeOfValue(diff).length === 1 ? mean(diff) : mean(diff, 0)
}

const P = (d: Distribution, name: string): Value => d.params[name]

// Normal: log(σ₂/σ₁) + (σ₁² + (μ₁ − μ₂)²)/(2σ₂²) − 1/2.
const normal: KlRule = (p, q) => {
  const [m1, s1, m2, s2] = [P(p, 'loc'), P(p, 'scale'), P(q, 'loc'), P(q, 'scale')]
  return sub(add(log(div(s2, s1)), div(add(square(s1), square(sub(m1, m2))), mul(2, square(s2)))), 0.5)
}
export const klNormal = rule('Normal', normal)
// KL is invariant under the shared bijection exp, so the log-normal pair is the normal pair of the logs.
export const klLogNormal = rule('LogNormal', (p, q) => {
  const [m1, s1, m2, s2] = [P(p, 'mu'), P(p, 'sigma'), P(q, 'mu'), P(q, 'sigma')]
  return sub(add(log(div(s2, s1)), div(add(square(s1), square(sub(m1, m2))), mul(2, square(s2)))), 0.5)
})

// Multivariate normal: ½[tr(Σ₂⁻¹Σ₁) + (μ₂ − μ₁)ᵀΣ₂⁻¹(μ₂ − μ₁) − d + log|Σ₂|/|Σ₁|], through the Cholesky factors.
export const klMultivariateNormal = rule('MultivariateNormal', (p, q) => {
  const a = p as MultivariateNormal
  const b = q as MultivariateNormal
  if (a.batchShape.length > 0 || b.batchShape.length > 0) throw new AifnError('kl', 'kl: batched multivariate normals')
  const d = a.eventShape[0]
  const M = solveTriangular(b.scaleTril, a.scaleTril)
  const z = solveTriangular(b.scaleTril, sub(b.loc, a.loc))
  const logDetRatio = sub(choleskyLogDet(b.scaleTril), choleskyLogDet(a.scaleTril))
  return mul(0.5, add(sub(add(sum(square(M)), sum(square(z))), d), logDetRatio))
})

// Beta: log B(a₂, b₂) − log B(a₁, b₁) + (a₁ − a₂)ψ(a₁) + (b₁ − b₂)ψ(b₁) + (a₂ − a₁ + b₂ − b₁)ψ(a₁ + b₁).
export const klBeta = rule('Beta', (p, q) => {
  const [a1, b1, a2, b2] = [P(p, 'a'), P(p, 'b'), P(q, 'a'), P(q, 'b')]
  return add(
    add(sub(logBeta(a2, b2), logBeta(a1, b1)), add(mul(sub(a1, a2), digamma(a1)), mul(sub(b1, b2), digamma(b1)))),
    mul(sub(add(a2, b2), add(a1, b1)), digamma(add(a1, b1))),
  )
})

// Gamma (shape α, rate β): (α₁ − α₂)ψ(α₁) − log Γ(α₁) + log Γ(α₂) + α₂ log(β₁/β₂) + α₁(β₂ − β₁)/β₁.
export const klGamma = rule('Gamma', (p, q) => {
  const [a1, r1, a2, r2] = [P(p, 'shape'), P(p, 'rate'), P(q, 'shape'), P(q, 'rate')]
  return add(
    add(sub(mul(sub(a1, a2), digamma(a1)), logGamma(a1)), logGamma(a2)),
    add(mul(a2, log(div(r1, r2))), div(mul(a1, sub(r2, r1)), r1)),
  )
})

// Exponential: log(λ₁/λ₂) + λ₂/λ₁ − 1.
export const klExponential = rule('Exponential', (p, q) => {
  const [l1, l2] = [P(p, 'rate'), P(q, 'rate')]
  return sub(add(log(div(l1, l2)), div(l2, l1)), 1)
})

// Poisson: λ₁ log(λ₁/λ₂) − λ₁ + λ₂.
export const klPoisson = rule('Poisson', (p, q) => {
  const [l1, l2] = [P(p, 'rate'), P(q, 'rate')]
  return add(sub(mul(l1, log(div(l1, l2))), l1), l2)
})

// Dirichlet: log Γ(α₀) − Σ log Γ(αᵢ) − log Γ(β₀) + Σ log Γ(βᵢ) + Σ (αᵢ − βᵢ)(ψ(αᵢ) − ψ(α₀)).
export const klDirichlet = rule('Dirichlet', (p, q) => {
  const [a, b] = [P(p, 'concentration'), P(q, 'concentration')]
  const a0 = sum(a, -1, true)
  const b0 = sumLast(b)
  return add(
    add(sub(logGamma(sumLast(a)), sumLast(logGamma(a))), sub(sumLast(logGamma(b)), logGamma(b0))),
    sumLast(mul(sub(a, b), sub(digamma(a), digamma(a0)))),
  )
})

/** Log-probabilities of a categorical from its parameters (probabilities or logits). */
function categoricalLogProbs(d: Distribution): Value {
  if ('logits' in d.params) return logSoftmax(d.params.logits)
  const probs = d.params.probs
  return log(div(probs, sum(probs, -1, true)))
}

/** x · log y given log y, with 0 · log y = 0 even where log y = −∞. */
function times(x: Value, logY: Value): Value {
  const ok = mask([x], (v) => v !== 0)
  return outside(ok, mul(x, guard(logY, ok, 0)), 0)
}

// Categorical: Σ pₖ (log pₖ − log qₖ), with 0 · log 0 = 0.
export const klCategorical = rule('Categorical', (p, q) => {
  const lp = categoricalLogProbs(p)
  const w = exp(lp)
  return sumLast(sub(times(w, lp), times(w, categoricalLogProbs(q))))
})

// Bernoulli: p log(p/q) + (1 − p) log((1 − p)/(1 − q)), with 0 · log 0 = 0.
export const klBernoulli = rule('Bernoulli', (p, q) => {
  const probs = (d: Distribution) => ('logits' in d.params ? sigmoid(d.params.logits) : d.params.probs)
  const logQ = (d: Distribution) => ('logits' in d.params ? logSigmoid(d.params.logits) : log(d.params.probs))
  const log1mQ = (d: Distribution) =>
    'logits' in d.params ? logSigmoid(neg(d.params.logits)) : log1p(neg(d.params.probs))
  const a = probs(p)
  const b = sub(1, a)
  return sub(add(xlogy(a, a), xlogy(b, b)), add(times(a, logQ(q)), times(b, log1mQ(q))))
})

/** Every closed-form KL rule, keyed `p|q` by family names. */
export const klRegistry: Readonly<Record<string, Entry<KlRule, KlRuleInfo>>> = entries<KlRuleInfo>('kl-rule', {
  klNormal,
  klLogNormal,
  klMultivariateNormal,
  klBeta,
  klGamma,
  klExponential,
  klPoisson,
  klDirichlet,
  klCategorical,
  klBernoulli,
}) as Readonly<Record<string, Entry<KlRule, KlRuleInfo>>>
