/**
 * Kullback–Leibler divergences $\KL(p \,\Vert\, q) = \expect_p[\log p(x) - \log q(x)]$ in closed form for pairs of
 * families, dispatched on the families' names, and a Monte Carlo estimate for any pair. Closed forms are compositions
 * of primitives, so they are differentiable in both distributions' parameters; each batches over the broadcast of the
 * two batches (the multivariate normal's excepted). The closed forms are those of one family with itself; each rule
 * below gives its formula, with subscript 1 for $p$ and 2 for $q$.
 *
 * Sources: Kullback and Leibler (1951), "On information and sufficiency", Ann. Math. Statist. 22(1); the closed forms
 * for the normal, gamma and Dirichlet families as in Penny (2001), "KL-divergences of Normal, Gamma, Dirichlet and
 * Wishart densities" (Wellcome Department of Imaging Neuroscience, UCL).
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

/**
 * A closed-form KL divergence between two distributions of given families: given $p$ and $q$ it returns
 * $\KL(p \,\Vert\, q)$ in nats, batched over the broadcast of their batches.
 */
export type KlRule = (p: Distribution, q: Distribution) => Value

const define = definer<KlRuleInfo>('kl-rule', 'probability/distributions')
const NOTES = ['kullback-leibler-divergence']
const CITE = ['kullback1951']

/**
 * Register a closed form for $\KL(p \,\Vert\, q)$ between two distributions of the same family (keyed `p|q`, e.g.
 * `Normal|Normal`), citing Kullback and Leibler (1951).
 *
 * @param family The family's name, as the distributions' `name` gives it.
 * @param f The closed form.
 * @param spec Registry fields that override the defaults (notes, citations, name).
 * @returns The registry entry, callable as the rule.
 */
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

/**
 * Whether `kl` has a closed form for the pair ($p$'s family, $q$'s family), looked up in `klRegistry` by name. It
 * does not check the batches: a pair of batched multivariate normals has an entry but `kl` throws for it.
 *
 * @param p The first distribution, the one the expectation is under.
 * @param q The second distribution.
 * @returns True when `kl(p, q)` has a registered closed form.
 *
 * @example Which pairs have a closed form
 * print('Normal, Normal:', hasKl(Normal(0, 1), Normal(1, 2)))
 * print('Normal, Laplace:', hasKl(Normal(0, 1), Laplace(0, 1)))
 */
export function hasKl(p: Distribution, q: Distribution): boolean {
  return `${p.name}|${q.name}` in klRegistry
}

/**
 * $\KL(p \,\Vert\, q) = \expect_p[\log p(x) - \log q(x)]$ in nats, in closed form, with the batch shape of the
 * broadcast batches; differentiable in both distributions' parameters. Pairs covered: Normal, LogNormal,
 * MultivariateNormal (unbatched), Beta, Gamma, Exponential, Poisson, Dirichlet, Categorical and Bernoulli (each with
 * itself); the table is `klRegistry`. Other pairs throw `AifnError`; use `klMonteCarlo`, or `klNumerical` / `klAuto`
 * for continuous univariate pairs.
 *
 * @param p The first distribution, the one the expectation is under.
 * @param q The second distribution, of the same family.
 * @returns The divergence: a number, or a tensor with the broadcast batch shape.
 *
 * @example Two normals, against the closed form
 * print('kl =', kl(Normal(0, 1), Normal(1, 2)))
 * print('log 2 + 2/8 - 1/2 =', Math.log(2) + 2 / 8 - 0.5)
 *
 * @example It is not symmetric
 * print('kl(p, q) =', kl(Bernoulli(0.5), Bernoulli(0.25)), ' log(4/3) / 2 =', 0.5 * Math.log(4 / 3))
 * print('kl(q, p) =', kl(Bernoulli(0.25), Bernoulli(0.5)))
 *
 * @example A batch of divergences at once
 * // Unit-variance normals: the divergence is half the squared distance between the means.
 * print('kl =', kl(Normal(tensor([0, 1, 2]), 1), Normal(0, 1)))
 */
export function kl(p: Distribution, q: Distribution): Value {
  const r = klRegistry[`${p.name}|${q.name}`]
  if (!r) throw new AifnError('kl', `kl: no closed form for ${p.name} ‖ ${q.name}; use klMonteCarlo`)
  return r(p, q)
}

/**
 * A Monte Carlo estimate of $\KL(p \,\Vert\, q)$: the mean of $\log p(x) - \log q(x)$ over $n$ draws from $p$.
 * Unbiased; its standard error falls as $1/\sqrt{n}$ (`klMonteCarloWithError` reports it). Works for any pair whose
 * log-densities are defined at $p$'s draws. Not differentiable in $p$'s parameters (the draws are not
 * reparameterised).
 *
 * @param s The random stream the draws come from.
 * @param p The distribution the draws come from, the one the expectation is under.
 * @param q The second distribution.
 * @param n The number of draws.
 * @returns The estimate: a number, or a tensor with the batch shape of the log-density difference.
 *
 * @example The estimate beside the closed form
 * const p = Normal(0, 1)
 * const q = Normal(1, 2)
 * print('estimate =', klMonteCarlo(stream(0), p, q, 10000))
 * print('closed form =', kl(p, q))
 */
export function klMonteCarlo(s: Stream, p: Distribution, q: Distribution, n: number): Value {
  const x = p.sample(s, { shape: [n] })
  const diff = sub(p.logProb(x) as Value, q.logProb(x) as Value)
  return shapeOfValue(diff).length === 1 ? mean(diff) : mean(diff, 0)
}

/**
 * A distribution's parameter by name.
 *
 * @param d The distribution.
 * @param name The parameter's name in `d.params`, such as `'loc'`.
 */
const P = (d: Distribution, name: string): Value => d.params[name]

/**
 * Normal:
 * $\log(\sigma_2/\sigma_1) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$.
 *
 * @param p The first normal, $\Gauss(\mu_1, \sigma_1^2)$.
 * @param q The second normal, $\Gauss(\mu_2, \sigma_2^2)$.
 */
const normal: KlRule = (p, q) => {
  const [m1, s1, m2, s2] = [P(p, 'loc'), P(p, 'scale'), P(q, 'loc'), P(q, 'scale')]
  return sub(add(log(div(s2, s1)), div(add(square(s1), square(sub(m1, m2))), mul(2, square(s2)))), 0.5)
}
/** The rule for two normals. */
export const klNormal = rule('Normal', normal)
/**
 * Log-normal: KL is invariant under the shared bijection $\exp$, so the log-normal pair is the normal pair of the
 * logs, $\log(\sigma_2/\sigma_1) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$.
 */
export const klLogNormal = rule('LogNormal', (p, q) => {
  const [m1, s1, m2, s2] = [P(p, 'mu'), P(p, 'sigma'), P(q, 'mu'), P(q, 'sigma')]
  return sub(add(log(div(s2, s1)), div(add(square(s1), square(sub(m1, m2))), mul(2, square(s2)))), 0.5)
})

/**
 * Multivariate normal:
 * $\frac{1}{2}\left[\trace(\Sigmamat_2^{-1}\Sigmamat_1) + (\muvec_2 - \muvec_1)^\top \Sigmamat_2^{-1}
 * (\muvec_2 - \muvec_1) - d + \log\frac{\det\Sigmamat_2}{\det\Sigmamat_1}\right]$, through the Cholesky factors
 * (the trace is $\lVert \Lmat_2^{-1}\Lmat_1 \rVert_F^2$). Unbatched only: a batch throws `AifnError`.
 */
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

/**
 * Beta: $\log B(a_2, b_2) - \log B(a_1, b_1) + (a_1 - a_2)\psi(a_1) + (b_1 - b_2)\psi(b_1)
 * + (a_2 - a_1 + b_2 - b_1)\psi(a_1 + b_1)$.
 */
export const klBeta = rule('Beta', (p, q) => {
  const [a1, b1, a2, b2] = [P(p, 'a'), P(p, 'b'), P(q, 'a'), P(q, 'b')]
  return add(
    add(sub(logBeta(a2, b2), logBeta(a1, b1)), add(mul(sub(a1, a2), digamma(a1)), mul(sub(b1, b2), digamma(b1)))),
    mul(sub(add(a2, b2), add(a1, b1)), digamma(add(a1, b1))),
  )
})

/**
 * Gamma (shape $\alpha$, rate $\beta$): $(\alpha_1 - \alpha_2)\psi(\alpha_1) - \log\Gamma(\alpha_1)
 * + \log\Gamma(\alpha_2) + \alpha_2 \log(\beta_1/\beta_2) + \alpha_1(\beta_2 - \beta_1)/\beta_1$.
 */
export const klGamma = rule('Gamma', (p, q) => {
  const [a1, r1, a2, r2] = [P(p, 'shape'), P(p, 'rate'), P(q, 'shape'), P(q, 'rate')]
  return add(
    add(sub(mul(sub(a1, a2), digamma(a1)), logGamma(a1)), logGamma(a2)),
    add(mul(a2, log(div(r1, r2))), div(mul(a1, sub(r2, r1)), r1)),
  )
})

/** Exponential: $\log(\lambda_1/\lambda_2) + \lambda_2/\lambda_1 - 1$. */
export const klExponential = rule('Exponential', (p, q) => {
  const [l1, l2] = [P(p, 'rate'), P(q, 'rate')]
  return sub(add(log(div(l1, l2)), div(l2, l1)), 1)
})

/** Poisson: $\lambda_1 \log(\lambda_1/\lambda_2) - \lambda_1 + \lambda_2$. */
export const klPoisson = rule('Poisson', (p, q) => {
  const [l1, l2] = [P(p, 'rate'), P(q, 'rate')]
  return add(sub(mul(l1, log(div(l1, l2))), l1), l2)
})

/**
 * Dirichlet: $\log\Gamma(\alpha_0) - \sum_i \log\Gamma(\alpha_i) - \log\Gamma(\beta_0) + \sum_i \log\Gamma(\beta_i)
 * + \sum_i (\alpha_i - \beta_i)(\psi(\alpha_i) - \psi(\alpha_0))$, with $\alpha_0 = \sum_i \alpha_i$ and
 * $\beta_0 = \sum_i \beta_i$.
 */
export const klDirichlet = rule('Dirichlet', (p, q) => {
  const [a, b] = [P(p, 'concentration'), P(q, 'concentration')]
  const a0 = sum(a, -1, true)
  const b0 = sumLast(b)
  return add(
    add(sub(logGamma(sumLast(a)), sumLast(logGamma(a))), sub(sumLast(logGamma(b)), logGamma(b0))),
    sumLast(mul(sub(a, b), sub(digamma(a), digamma(a0)))),
  )
})

/**
 * Log-probabilities of a categorical from its parameters (probabilities, normalised here, or logits).
 *
 * @param d The categorical distribution.
 * @returns $\log p_k$ along the last axis, with the shape of its parameters.
 */
function categoricalLogProbs(d: Distribution): Value {
  if ('logits' in d.params) return logSoftmax(d.params.logits)
  const probs = d.params.probs
  return log(div(probs, sum(probs, -1, true)))
}

/**
 * $x \log y$ given $\log y$, with $0 \cdot \log y = 0$ even where $\log y = -\infty$.
 *
 * @param x The factor $x$.
 * @param logY The logarithm $\log y$, possibly $-\infty$.
 */
function times(x: Value, logY: Value): Value {
  const ok = mask([x], (v) => v !== 0)
  return outside(ok, mul(x, guard(logY, ok, 0)), 0)
}

/** Categorical: $\sum_k p_k (\log p_k - \log q_k)$, with $0 \log 0 = 0$. */
export const klCategorical = rule('Categorical', (p, q) => {
  const lp = categoricalLogProbs(p)
  const w = exp(lp)
  return sumLast(sub(times(w, lp), times(w, categoricalLogProbs(q))))
})

/**
 * Bernoulli: $p \log(p/q) + (1 - p) \log\frac{1 - p}{1 - q}$, with $0 \log 0 = 0$; either distribution may be
 * given by probability or by logits.
 */
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
