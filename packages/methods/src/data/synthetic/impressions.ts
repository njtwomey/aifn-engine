/**
 * Simulated ad impressions for click-through prediction: one active value per categorical field (plus a bias feature)
 * and clicks from a probit model with known weights, as in AdPredictor's setting (Graepel et al., 2010).
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'

/**
 * Simulated impressions: `fields` categorical fields with `values` values each, one active value per field, and clicks
 * from a probit model with true weights N(0, scale²) (a bias field of one value comes first, with weight `bias`).
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
