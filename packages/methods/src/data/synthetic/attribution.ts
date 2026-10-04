/**
 * A tabular classification task with known feature roles, for explanation methods: x ~ N(0, I₆) and
 * y = 1 when 2x₁ − 1.5x₂ + 1.5x₃x₄ + noise > 0. Features 1 and 2 have main effects, 3 and 4 act only through their
 * product (an interaction), and 5 and 6 are irrelevant.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData } from 'aifn-compute/foundation/tensor'
import { checkCount, matrix, type Dataset } from '../types'

/** The noise-free score 2x₁ − 1.5x₂ + 1.5x₃x₄ of a row (zero-based x[0] … x[5]). */
export const attributionScore = (x: ArrayLike<number>): number => 2 * x[0] - 1.5 * x[1] + 1.5 * x[2] * x[3]

/** n rows of the attribution task with logistic-scale label noise of standard deviation `noise`. */
export function attributionTask(s: Stream, options: { n?: number; noise?: number } = {}): Dataset {
  const { n = 400, noise = 0.5 } = options
  checkCount(n, 'attributionTask')
  const d = 6
  const r = child(s, 'rows')
  const x = Float64Array.from({ length: n * d }, () => normal(r) as number)
  const e = child(s, 'noise')
  const y = Int32Array.from({ length: n }, (_, i) =>
    attributionScore(x.subarray(i * d, (i + 1) * d)) + noise * (normal(e) as number) > 0 ? 1 : 0,
  )
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: fromData(y, [n]),
    meta: {
      name: 'attribution task',
      description: `${n} rows of six standard normal features; y = 1 when 2x₁ − 1.5x₂ + 1.5x₃x₄ + noise (sd ${noise}) > 0. x₅ and x₆ are irrelevant.`,
      task: 'classification',
      featureNames: ['x₁ (main)', 'x₂ (main)', 'x₃ (interaction)', 'x₄ (interaction)', 'x₅ (none)', 'x₆ (none)'],
      labelNames: ['0', '1'],
      key: s.key,
    },
  }
}

definer<DatasetInfo>('dataset', 'data/synthetic')(
  {
    key: 'attributionTask',
    name: 'Attribution task',
    summary:
      'Six features with main effects, an interaction pair and two irrelevant ones: a test bed for explanations.',
    task: 'classification',
    output: 'dataset',
    knobs: space({ n: int(10, 10000, { default: 400 }), noise: real(0, 3, { default: 0.5 }) }),
    truth: false,
    random: true,
    notes: ['feature-attribution-methods'],
  },
  attributionTask,
)
