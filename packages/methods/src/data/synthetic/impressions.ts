/**
 * Simulated ad impressions for click-through prediction: one active value per categorical field (plus a bias feature)
 * and clicks from a probit model with known weights, as in AdPredictor's setting (Graepel et al., 2010).
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'

/**
 * Simulated impressions: `fields` categorical fields with `values` values each, one active value per field drawn
 * uniformly, and clicks from a probit model. Feature 0 is a bias that every impression has, with weight `bias`; the
 * other weights are drawn as $w_i \sim \Gauss(0, \sigma^2)$, $\sigma$ = `scale`. An impression with active features
 * $A$ is clicked when $\sum_{i \in A} w_i + \varepsilon > 0$, $\varepsilon \sim \Gauss(0, 1)$, so with probability
 * $\Phi(\sum_{i \in A} w_i)$.
 *
 * @param s The stream the weights, the active values and the click noise are drawn from (children `'weights'`,
 *   `'values'` and `'noise'`).
 * @param options The size and the model. `n` (default 5000) is the number of impressions; `fields` (default 3) the
 *   number of categorical fields; `values` (default 10) the number of values of each field; `scale` (default 0.7) the
 *   standard deviation $\sigma$ of the true weights; `bias` (default $-1$) the weight of the bias feature, which sets
 *   the base click rate.
 * @returns `impressions`, one array per impression of its active feature indices (0, then one per field: field $f$'s
 *   value $v$ is feature $1 + f \cdot \text{values} + v$); `clicks`, whether each was clicked; `weights`, the true
 *   weights of all `features`; and `features`, their number $1 + \text{fields} \cdot \text{values}$.
 *
 * @example Sparse rows and the click rate the bias sets
 * const { impressions, clicks, weights, features } = simulatedImpressions(stream(1), { n: 2000 })
 * print('impressions:', impressions.length, ' features:', features)
 * print('first rows:', impressions.slice(0, 3))
 * print('bias weight:', weights[0])
 * print('click rate:', clicks.filter(Boolean).length / clicks.length)
 */
export function simulatedImpressions(
  s: Stream,
  options: { n?: number; fields?: number; values?: number; scale?: number; bias?: number } = {},
): { impressions: number[][]; clicks: boolean[]; weights: Float64Array; features: number } {
  const { n = 5000, fields = 3, values = 10, scale = 0.7, bias = -1 } = options
  const features = 1 + fields * values
  const weights = Float64Array.from(standardNormals(child(s, 'weights'), features), (z, i) =>
    i === 0 ? bias : scale * z,
  )
  const pick = units(child(s, 'values'), n * fields)
  const noise = standardNormals(child(s, 'noise'), n)
  const impressions: number[][] = []
  const clicks: boolean[] = []
  for (let r = 0; r < n; r++) {
    const active = [
      0,
      ...Array.from({ length: fields }, (_, f) => 1 + f * values + Math.floor(pick[r * fields + f] * values)),
    ]
    impressions.push(active)
    clicks.push(active.reduce((a, i) => a + weights[i], 0) + noise[r] > 0)
  }
  return { impressions, clicks, weights, features }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'simulatedImpressions',
    name: 'Simulated ad impressions',
    summary: 'Sparse one-hot impressions and probit clicks with known weights, for click-through prediction.',
    role: 'simulation',
    random: true,
    notes: ['adpredictor'],
    cite: ['graepel2010adpredictor'],
  },
  simulatedImpressions,
)
