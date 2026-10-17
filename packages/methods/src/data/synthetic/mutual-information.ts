/**
 * Batches of Gaussian pairs with a known mutual information, scored by the optimal critic, for comparing the
 * variational bounds of `aifn-compute/probability/information` with the truth (Poole et al., 2019, §4).
 *
 * The pairs are $\xvec \sim \Gauss(\zeros, \Imat_d)$ and $\yvec = \rho\xvec + \sqrt{1 - \rho^2}\,\epsilonvec$, so
 * every coordinate pair has correlation $\rho$ and $I(\xvec; \yvec) = -\frac d2 \log(1 - \rho^2)$. The score of
 * $(\xvec_i, \yvec_j)$ is the exact log density ratio $\log p(\yvec_j \mid \xvec_i) / p(\yvec_j)$, the optimal critic
 * of InfoNCE and Donsker–Varadhan (the NWJ optimum is that plus 1).
 */

import { child, standardNormals, type Stream } from 'aifn-compute/foundation/random'
import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { definer } from 'aifn-compute/foundation/registry'

/** A scored batch of Gaussian pairs, as `gaussianPairLogRatios` returns it. */
export interface GaussianPairBatch {
  /** $\log p(\yvec_j \mid \xvec_i) / p(\yvec_j)$, row-major $n \times n$: positives on the diagonal. */
  scores: Float64Array
  /** The true mutual information $-\frac d2 \log(1 - \rho^2)$ in nats. */
  mutualInformation: number
}

/**
 * Draw $n$ Gaussian pairs (see the file comment) and score every $(\xvec_i, \yvec_j)$ with the exact log density
 * ratio $\sum_k \big[-\frac12 \log(1 - \rho^2) - \frac{(y_{jk} - \rho x_{ik})^2}{2(1 - \rho^2)} +
 * \frac{y_{jk}^2}{2}\big]$. Throws `DomainError` unless $\rho \in [0, 1)$ and $n$ is a positive integer.
 *
 * @param s The stream; $\xvec$ from its child `'x'`, the noise from `'noise'`.
 * @param options The batch.
 * @param options.n The batch size $n$.
 * @param options.rho The correlation $\rho$ of each coordinate pair, in $[0, 1)$.
 * @param options.dims The dimension $d$ (default 1).
 * @returns The score matrix and the true mutual information.
 *
 * @example InfoNCE and NWJ at the optimal critic, against the truth
 * import { correlationForMutualInformation, infoNceBound, nwjBound } from 'aifn-compute/probability/information'
 * const b = gaussianPairLogRatios(stream(1), { n: 64, rho: correlationForMutualInformation(1) })
 * print('true I:', b.mutualInformation)
 * print('InfoNCE:', infoNceBound(b.scores, 64), ' NWJ:', nwjBound(b.scores.map((v) => v + 1), 64))
 */
export function gaussianPairLogRatios(
  s: Stream,
  { n, rho, dims = 1 }: { n: number; rho: number; dims?: number },
): GaussianPairBatch {
  if (!(rho >= 0 && rho < 1))
    throw new DomainError('gaussianPairLogRatios', 'gaussianPairLogRatios: rho must lie in [0, 1)')
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('gaussianPairLogRatios', 'gaussianPairLogRatios: n must be a positive integer')
  const x = standardNormals(child(s, 'x'), n * dims)
  const e = standardNormals(child(s, 'noise'), n * dims)
  const v = 1 - rho * rho
  const c = Math.sqrt(v)
  const y = new Float64Array(n * dims)
  for (let k = 0; k < n * dims; k++) y[k] = rho * x[k] + c * e[k]
  const constant = -0.5 * dims * Math.log(v)
  const scores = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let t = constant
      for (let k = 0; k < dims; k++) {
        const yk = y[j * dims + k]
        const r = yk - rho * x[i * dims + k]
        t += (yk * yk) / 2 - (r * r) / (2 * v)
      }
      scores[i * n + j] = t
    }
  return { scores, mutualInformation: constant }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'gaussianPairLogRatios',
    name: 'Scored Gaussian pairs',
    summary: 'A batch of correlated Gaussian pairs scored by the exact log density ratio, with the true I(X; Y).',
    role: 'simulation',
    random: true,
    notes: ['mutual-information-maximisation'],
    cite: ['poole2019'],
  },
  gaussianPairLogRatios,
)
