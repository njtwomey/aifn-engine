/**
 * The half-kernel: two nested half-ellipses, one per class, the second synthetic problem of the label-propagation
 * experiments for learning with label proportions (Poyiadzi, Santos-Rodriguez and Twomey 2018, Fig. 2). The paper
 * describes it as "a weighted combination of trigonometric functions, sin and cos, and added noise"; the recipe here is
 * the common two-dimensional benchmark of that name: for φ ~ U[0, π],
 *
 *   inner (class 1): (x₀ + r₁ sin φ, ρ r₁ cos φ) + u,   outer (class 0): (x₀ + r₂ sin φ, ρ r₂ cos φ) + u,
 *
 * with u uniform on [−ν/2, ν/2]² and defaults x₀ = −20, r₁ = 20, r₂ = 35, ν = 4, ρ = 0.6, all multiplied by `scale`
 * (default 0.05, so the points lie within about ±1.75, the size of the Gaussian XOR).
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import type { Dataset } from '../types'
import { checkCount, labels, matrix } from '../types'

/** Options of {@link halfKernel}. */
export interface HalfKernelOptions {
  /** Total points (default 300), split evenly between the arcs (the inner arc gets the odd one). */
  n?: number
  /** Radii of the inner and outer arcs (default 20 and 35), their left end x₀ (default −20). */
  inner?: number
  outer?: number
  left?: number
  /** Side of the uniform noise square (default 4) and the vertical squash ρ (default 0.6). */
  noise?: number
  ratio?: number
  /** A factor applied to every coordinate (default 0.05). */
  scale?: number
}

/** Two nested noisy half-ellipses (module notes); y = 1 on the inner arc. */
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
