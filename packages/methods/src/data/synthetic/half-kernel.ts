/**
 * The half-kernel: two nested half-ellipses, one per class, the second synthetic problem of the label-propagation
 * experiments for learning with label proportions (Poyiadzi, Santos-Rodriguez and Twomey 2018, Fig. 2). The paper
 * describes it as "a weighted combination of trigonometric functions, sin and cos, and added noise"; the recipe here is
 * the common two-dimensional benchmark of that name: for $\varphi \sim \Unif(0, \pi)$, a point of arc $r$ is
 * $(x_0 + r \sin\varphi,\ \rho r \cos\varphi) + \uvec$, with radius $r_1$ for the inner arc (class 1) and $r_2$ for
 * the outer one (class 0), and $\uvec$ uniform on $[-\nu/2, \nu/2]^2$. The defaults are $x_0 = -20$, $r_1 = 20$,
 * $r_2 = 35$, $\nu = 4$ and $\rho = 0.6$, and every coordinate is then multiplied by `scale` (default 0.05, so the
 * points lie within $[-1.1, 0.85] \times [-1.15, 1.15]$).
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import type { Dataset } from '../types'
import { checkCount, labels, matrix } from '../types'

/** Options of `halfKernel`. */
export interface HalfKernelOptions {
  /** Total points (default 300), split evenly between the arcs (the inner arc gets the odd one). */
  n?: number
  /** Radius $r_1$ of the inner arc, class 1 (default 20). */
  inner?: number
  /** Radius $r_2$ of the outer arc, class 0 (default 35). */
  outer?: number
  /** The arcs' left end $x_0$, where $\varphi = 0$ and $\pi$ put them (default $-20$). */
  left?: number
  /** Side $\nu$ of the square of uniform noise added to every point (default 4). */
  noise?: number
  /** The vertical squash $\rho$, which turns the half-circles into half-ellipses (default 0.6). */
  ratio?: number
  /** A factor applied to every coordinate (default 0.05). */
  scale?: number
}

/**
 * Two nested noisy half-ellipses (see the file comment), with $y = 1$ on the inner arc. The first $\lceil n/2 \rceil$
 * rows are the inner arc and the rest the outer one, in that order. Throws `DomainError` when $n$ is not a non-negative
 * integer.
 *
 * @param s The stream the angles (children `'phi'`) and the noise (children `'noise'`) are drawn from, one child per
 *   point, so a point does not change when $n$ grows.
 * @param options The size, the shape and the noise; every length is multiplied by `scale`.
 * @returns A classification dataset: `x` ($n \times 2$) and `y` (1 inner, 0 outer).
 *
 * @example Two arcs, the inner one first
 * const d = halfKernel(stream(1), { n: 300 })
 * const x = toArray(d.x)
 * const y = toArray(d.y)
 * print('x:', d.x.shape, ' first rows:', x.slice(0, 2), ' labels:', y.slice(0, 2))
 * // Distance from the arcs' centre (-1, 0) in the unsquashed frame: about 1 inside, 1.75 outside.
 * const radius = (r) => Math.hypot(r[0] + 1, r[1] / 0.6)
 * const mean = (rows) => rows.reduce((a, r) => a + radius(r), 0) / rows.length
 * print('mean radius, inner:', mean(x.filter((_, i) => y[i] === 1)), ' outer:', mean(x.filter((_, i) => y[i] === 0)))
 */
export function halfKernel(s: Stream, options: HalfKernelOptions = {}): Dataset {
  const { n = 300, inner = 20, outer = 35, left = -20, noise = 4, ratio = 0.6, scale = 0.05 } = options
  checkCount(n, 'halfKernel')
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  const ones = Math.ceil(n / 2)
  for (let i = 0; i < n; i++) {
    const c = i < ones ? 1 : 0
    const r = c === 1 ? inner : outer
    const phi = Math.PI * uniform(child(s, 'phi', i))
    const ux = noise * (uniform(child(s, 'noise', i, 0)) - 0.5)
    const uy = noise * (uniform(child(s, 'noise', i, 1)) - 0.5)
    x[2 * i] = scale * (left + r * Math.sin(phi) + ux)
    x[2 * i + 1] = scale * (ratio * r * Math.cos(phi) + uy)
    y[i] = c
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    meta: {
      name: 'half-kernel',
      description: `${n} points on two nested noisy half-ellipses (radii ${inner} and ${outer}, scaled by ${scale}); class 1 is the inner arc.`,
      task: 'classification',
      featureNames: ['x1', 'x2'],
      labelNames: ['outer', 'inner'],
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'halfKernel',
    name: 'Half-kernel',
    summary: 'Two nested noisy half-ellipses, one class each, as in the label-proportion propagation experiments.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 100000, { default: 300 }),
      noise: real(0, 20, { default: 4 }),
      scale: real(0.001, 10, { default: 0.05 }),
    }),
    truth: false,
    random: true,
    notes: ['label-propagation-for-label-proportions', 'learning-from-label-proportions'],
    cite: ['poyiadzi2018'],
  },
  halfKernel,
)
