/**
 * A tabular classification task with known feature roles, for explanation methods: $\xvec \sim \Gauss(\zeros, \Imat_6)$
 * and $y = 1$ when $2x_1 - 1.5x_2 + 1.5x_3x_4 + \sigma\varepsilon > 0$, $\varepsilon \sim \Gauss(0, 1)$. Features 1
 * and 2 have main effects, 3 and 4 act only through their product (an interaction), and 5 and 6 are irrelevant.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData } from 'aifn-compute/foundation/tensor'
import { checkCount, matrix, type Dataset } from '../types'

/**
 * The noise-free score $2x_1 - 1.5x_2 + 1.5x_3x_4$ of a row of `attributionTask`; its sign is the label without noise.
 *
 * @param x A row of six features, zero-based: `x[0]` is $x_1$. Only the first four are read.
 * @returns The score.
 *
 * @example The interaction pair counts only together
 * print('x1 = 1:', attributionScore([1, 0, 0, 0, 0, 0]))
 * print('x3 = 1 alone:', attributionScore([0, 0, 1, 0, 0, 0]))
 * print('x3 = x4 = 1:', attributionScore([0, 0, 1, 1, 0, 0]))
 * print('x5, x6 ignored:', attributionScore([0, 0, 0, 0, 5, -5]))
 */
export const attributionScore = (x: ArrayLike<number>): number => 2 * x[0] - 1.5 * x[1] + 1.5 * x[2] * x[3]

/**
 * $n$ rows of the attribution task: six standard normal features, and $y = 1$ when the score $f(\xvec)$ of
 * `attributionScore` plus Gaussian noise $\sigma\varepsilon$, $\varepsilon \sim \Gauss(0, 1)$, is positive. Throws
 * `DomainError` when $n$ is not a non-negative integer.
 *
 * @param s The stream the rows (child `'rows'`) and the label noise (child `'noise'`) are drawn from.
 * @param options `n` (default 400), the number of rows, and `noise` (default 0.5), the standard deviation $\sigma$ of
 *   the Gaussian noise added to the score before thresholding.
 * @returns A classification dataset: `x` ($n \times 6$), `y` (labels 0 or 1), and feature names that give each one's
 *   role.
 *
 * @example Labels follow the score, up to noise
 * const d = attributionTask(stream(1), { n: 400 })
 * const x = toArray(d.x)
 * const y = toArray(d.y)
 * print('x:', d.x.shape, ' y:', d.y.shape)
 * print('first row:', x[0], ' label:', y[0])
 * print('share of labels that match the sign of the score:',
 *   x.filter((r, i) => (attributionScore(r) > 0 ? 1 : 0) === y[i]).length / x.length)
 */
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
