/**
 * Two classes on parallel sine waves in the plane, the data of a label-plus-style generative model (the M2 model of
 * Kingma et al., 2014): a shared style $t \sim \Unif(-2, 2)$ sets the position along the wave and the class
 * $c \in \{0, 1\}$ the vertical offset,
 *
 * $$\xvec = \big(t,\ (2c - 1)\,o + a \sin(f t)\big) + \epsilonvec,\quad \epsilonvec \sim \Gauss(\zeros,\sigma^2\Imat)$$
 *
 * The two waves never cross, but with an amplitude $a$ larger than the offset $o$ no horizontal line separates them,
 * so a classifier fitted to a few labels misses the wavy gap that the unlabelled points reveal. Rows alternate classes
 * (row $i$ has class $i \bmod 2$).
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { float64Data, type Tensor } from 'aifn-compute/foundation/tensor'
import { labels, matrix, type Dataset } from '../types'

/** Options of `parallelWaves`. */
export interface ParallelWavesOptions {
  /** Number of points (default 200). */
  n?: number
  /** Vertical offset $o$ of each wave from the axis (default 0.5). */
  offset?: number
  /** Amplitude $a$ (default 0.6). */
  amplitude?: number
  /** Angular frequency $f$ in radians per unit of $t$ (default 1.6). */
  frequency?: number
  /** Noise standard deviation $\sigma$ (default 0.08). */
  noise?: number
}

/** The two-wave data: a classification `Dataset` with each row's style. */
export interface ParallelWaves extends Dataset {
  /** The class of each row, 0 or 1. */
  readonly y: Tensor
  /** The style $t$ of each row, the position along the wave before noise. */
  readonly style: Float64Array
}

/**
 * Draw the two-wave data set; see the file comment. Throws `DomainError` unless $n$ is a non-negative integer and
 * $\sigma \ge 0$.
 *
 * @param s The stream: the style from its child `'style'`, the noise from its child `'noise'`.
 * @param options The size, offset, amplitude, frequency and noise.
 * @returns A classification dataset (`x` $n \times 2$, `y` int32 alternating 0 and 1) with `style`, each row's $t$.
 *
 * @example The noise-free waves sit at $\pm o$ plus the same sine
 * const d = parallelWaves(stream(1), { n: 6, noise: 0 })
 * print('x:', d.x)
 * print('y:', d.y, ' style:', d.style)
 */
export function parallelWaves(s: Stream, options: ParallelWavesOptions = {}): ParallelWaves {
  const { n = 200, offset: o = 0.5, amplitude: a = 0.6, frequency: f = 1.6, noise: sd = 0.08 } = options
  if (!(Number.isInteger(n) && n >= 0))
    throw new DomainError('parallelWaves', 'parallelWaves: n must be a non-negative integer')
  if (!(sd >= 0)) throw new DomainError('parallelWaves', 'parallelWaves: noise must be non-negative')
  const t = float64Data(uniform(child(s, 'style'), -2, 2, { shape: [n] }) as Tensor)
  const e = float64Data(normals(child(s, 'noise'), [n, 2], 0, sd))
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const c = i % 2
    y[i] = c
    x[2 * i] = t[i] + e[2 * i]
    x[2 * i + 1] = (2 * c - 1) * o + a * Math.sin(f * t[i]) + e[2 * i + 1]
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    style: Float64Array.from(t),
    meta: {
      name: 'parallel-waves',
      description: `${n} points on two parallel sine waves (offset ${o}, amplitude ${a}, frequency ${f}, noise ${sd}).`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['lower', 'upper'],
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'parallelWaves',
    name: 'Parallel waves',
    summary: 'Two classes on parallel sine waves: one shared style along the wave and the class as the offset.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 100000, { default: 200 }),
      offset: real(0, 2, { default: 0.5 }),
      amplitude: real(0, 2, { default: 0.6 }),
      frequency: real(0, 10, { default: 1.6 }),
      noise: real(0, 1, { default: 0.08 }),
    }),
    truth: false,
    random: true,
    notes: ['semi-supervised-variational-autoencoders'],
    cite: ['kingma2014semi'],
  },
  parallelWaves,
)
