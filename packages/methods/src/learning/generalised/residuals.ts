/**
 * Residuals of a generalised model at its fitted means (McCullagh and Nelder, 1989, "Generalized Linear Models", 2nd
 * ed., §2.4): response y − μ, Pearson (y − μ)√w/√V(μ), deviance sign(y − μ)√(w·d(y, μ)) and working (y − μ)/μ′(η).
 * Shared by `glm` and `gam`.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Family, Link } from 'aifn-compute/probability/likelihoods'

/** The kinds of residual. */
export type ResidualKind = 'response' | 'pearson' | 'deviance' | 'working'

/** What residuals are computed from: responses, fitted means and linear predictor, prior weights, family and link. */
export type ResidualInput = {
  y: Tensor
  mu: Tensor
  eta: Tensor
  /** Prior weights [n] (default 1). */
  weights?: Tensor
  family: Family
  link: Link
}

/** The residuals of a kind ([n]); `deviance` by default. */
export function residuals(
  { y, mu, eta, weights, family, link }: ResidualInput,
  kind: ResidualKind = 'deviance',
): Tensor {
  const ys = dense.data(y)
  const ms = dense.data(mu)
  const n = ys.length
  const w = weights ? dense.data(weights) : new Float64Array(n).fill(1)
  const out = new Float64Array(n)
  if (kind === 'response') for (let i = 0; i < n; i++) out[i] = ys[i] - ms[i]
  else if (kind === 'pearson') {
    const V = dense.data(family.variance(mu) as Tensor)
    for (let i = 0; i < n; i++) out[i] = ((ys[i] - ms[i]) * Math.sqrt(w[i])) / Math.sqrt(V[i])
  } else if (kind === 'working') {
    const dmu = dense.data(link.derivative(eta) as Tensor)
    for (let i = 0; i < n; i++) out[i] = (ys[i] - ms[i]) / dmu[i]
  } else {
    const unit = dense.data(family.unitDeviance(y, mu) as Tensor)
    for (let i = 0; i < n; i++) out[i] = Math.sign(ys[i] - ms[i]) * Math.sqrt(Math.max(w[i] * unit[i], 0))
  }
  return fromData(out, [n])
}
