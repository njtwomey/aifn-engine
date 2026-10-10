/**
 * Residuals of a generalised model at its fitted means (McCullagh and Nelder, 1989, "Generalized Linear Models", 2nd
 * ed., §2.4): response $y - \mu$, Pearson $(y - \mu)\sqrt{w}/\sqrt{V(\mu)}$, deviance
 * $\sgn(y - \mu)\sqrt{w\, d(y, \mu)}$ and working $(y - \mu)/\mu'(\eta)$, as R's `residuals.glm` and statsmodels'
 * `resid_response`, `resid_pearson`, `resid_deviance` and `resid_working`. Shared by `glm` and `gam`.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Family, Link } from 'aifn-compute/probability/likelihoods'

/** The kinds of residual: `response`, `pearson`, `deviance` or `working`. */
export type ResidualKind = 'response' | 'pearson' | 'deviance' | 'working'

/** What residuals are computed from: responses, fitted means and linear predictor, prior weights, family and link. */
export type ResidualInput = {
  /** Responses $y_i$, $n$. */
  y: Tensor
  /** Fitted means $\mu_i$, $n$. */
  mu: Tensor
  /** Linear predictor $\eta_i$, $n$ (read by `working` residuals only). */
  eta: Tensor
  /** Prior weights $w_i$, $n$ (default 1). */
  weights?: Tensor
  /** The family: its variance function $V$ and unit deviance $d$. */
  family: Family
  /** The link: its inverse's derivative $\mu'(\eta)$. */
  link: Link
}

/**
 * The residuals of a kind; `deviance` by default. Prior weights enter the Pearson and deviance residuals, not the
 * response and working ones. The deviance residuals' squares sum to the deviance.
 *
 * @param options The fit the residuals are of.
 * @param options.y The responses $y_i$, $n$.
 * @param options.mu The fitted means $\mu_i$, $n$.
 * @param options.eta The linear predictor $\eta_i$, $n$ (read by `working` residuals only).
 * @param options.weights Prior weights $w_i$, $n$ (default 1).
 * @param options.family The family: its variance function $V$ and unit deviance $d$.
 * @param options.link The link: its inverse's derivative $\mu'(\eta)$.
 * @param kind Which residuals.
 * @returns The residuals, $n$.
 *
 * @example Deviance residuals of a Poisson fit: their squares sum to the deviance
 * import { link, poissonFamily } from 'aifn-compute/probability/likelihoods'
 * const y = tensor([0, 1, 5])
 * const mu = tensor([1, 2, 3])
 * const fit = { y, mu, eta: log(mu), family: poissonFamily(), link: link('log') }
 * const r = residuals(fit)
 * print('deviance residuals =', r)
 * print('sum of squares =', sum(mul(r, r)), ' deviance =', deviance(poissonFamily(), y, mu))
 *
 * @example The four kinds side by side
 * import { link, poissonFamily } from 'aifn-compute/probability/likelihoods'
 * const mu = tensor([1, 2, 3])
 * const fit = { y: tensor([0, 1, 5]), mu, eta: log(mu), family: poissonFamily(), link: link('log') }
 * for (const kind of ['response', 'pearson', 'deviance', 'working']) print(kind + ':', residuals(fit, kind))
 */
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
