/**
 * Two views of one binary class with a knob for their conditional dependence: the setting of co-training (Blum and
 * Mitchell, 1998, §2: each view sufficient, the views conditionally independent given the class).
 *
 * Each point has a class $y$ (the first half 0, the second half 1, as the labelled generators order classes) and two
 * views in the square $[-1, 1]^2$. In view $\avec$ the class is a noisy function of the first coordinate,
 * $P(y = 1 \mid \avec) = \sigma(s a_1)$; in view $\bvec$ of the second, $P(y = 1 \mid \bvec) = \sigma(s b_2)$, for
 * the sharpness $s$. Each view's marginal density is uniform on the square, so the unlabelled points of one view carry
 * no information about its boundary: the cluster and low-density assumptions fail in each view alone. Given $y$, the
 * class coordinate is drawn from the density $\sigma(\pm s u)$ on $[-1, 1]$ by inverting its distribution function in
 * closed form (its mass is exactly 1, since $\log(1 + e^s) - \log(1 + e^{-s}) = s$). With probability
 * $1 - \rho$ ($\rho$ the dependence), $\bvec$ is drawn independently given $y$; with probability $\rho$ it copies
 * $\avec$ rotated by 90 degrees, $(b_1, b_2) = (-a_2, a_1)$, so $\avec$'s class coordinate becomes $\bvec$'s. Each
 * view alone has Bayes error $(\log 2 - \log(1 + e^{-s})) / s$. The construction is a toy of the co-training notes.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromRows, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { labels, type Dataset } from '../types'

/** Options of `twoViewDataset`. */
export interface TwoViewDatasetOptions {
  /** Number of points, split evenly over the two classes (an odd $n$ gives class 1 the extra point). Default 200. */
  n?: number
  /** Slope $s > 0$ of the class probability $\sigma(s u)$ along each view's class coordinate. Default 8. */
  sharpness?: number
  /**
   * Probability $\rho \in [0, 1]$ that view $\bvec$ copies view $\avec$ (rotated) instead of being drawn afresh.
   * Default 0.
   */
  dependence?: number
}

/** A two-view dataset: a classification `Dataset` whose `x` holds both views, with each view on its own. */
export interface TwoViewDataset extends Dataset {
  /** The class of each row, 0 or 1. */
  readonly y: Tensor
  /** View $\avec$, $n \times 2$; its class coordinate is the first. */
  readonly a: Tensor
  /** View $\bvec$, $n \times 2$; its class coordinate is the second. */
  readonly b: Tensor
  /** 1 for a point whose view $\bvec$ copies its view $\avec$. */
  readonly copied: Uint8Array
  /** The Bayes error of one view alone, $(\log 2 - \log(1 + e^{-s})) / s$. */
  readonly truth: { readonly viewBayesError: number }
}

/**
 * The class coordinate $u \in [-1, 1]$ given the class: the inverse at $w$ of the distribution function of the density
 * $\sigma(\pm s u)$ on $[-1, 1]$, $e^{su} = (1 + e^{-s}) e^{sw} - 1$, negated for class 0.
 *
 * @param w A uniform draw in $[0, 1]$.
 * @param cls The class, 0 or 1.
 * @param s The sharpness $s > 0$.
 * @returns The coordinate $u$, positive on average for class 1 and negative for class 0.
 *
 * @example The quartiles of the class-1 coordinate
 * print([0.25, 0.5, 0.75].map((w) => classCoordinate(w, 1, 8)))
 */
export function classCoordinate(w: number, cls: number, s: number): number {
  // The CDF from -1 is (log(1 + e^{s u}) - log(1 + e^{-s})) / s = w, so e^{s u} = expm1(s w) + e^{s (w - 1)}.
  const u = Math.log(Math.expm1(s * w) + Math.exp(s * (w - 1))) / s
  return cls === 1 ? u : -u
}

/**
 * Two views of a binary class with uniform marginals and a dependence knob; see the file comment. Throws
 * `DomainError` for a dependence outside $[0, 1]$, a sharpness that is not positive, or fewer than 2 points.
 *
 * @param s The random stream; its children `'a'`, `'b'` and `'copy'` give the draws of view $\avec$, view $\bvec$
 *   and the copy choice.
 * @param options The size, sharpness and dependence.
 * @returns A classification dataset (`x` the $n \times 4$ views side by side, $\avec$'s columns first; `y` int32), with
 *   the views `a` and `b`, which points copy their view (`copied`), and the Bayes error of one view (`truth`).
 *
 * @example One view's Bayes error, and the error of its Bayes rule on a sample
 * const d = twoViewDataset(stream(3), { n: 4000 })
 * const a = toArray(d.a)
 * const y = toArray(d.y)
 * print('Bayes error of a view:', d.truth.viewBayesError)
 * print('error of sign(a1):', a.filter((r, i) => (r[0] > 0 ? 1 : 0) !== y[i]).length / y.length)
 *
 * @example With dependence 1, view b is view a rotated
 * const d = twoViewDataset(stream(3), { n: 3, dependence: 1 })
 * print('a:', d.a)
 * print('b:', d.b)
 */
export function twoViewDataset(s: Stream, options: TwoViewDatasetOptions = {}): TwoViewDataset {
  const { n = 200, sharpness: sharp = 8, dependence: dep = 0 } = options
  if (!(dep >= 0 && dep <= 1)) throw new DomainError('twoViewDataset', 'twoViewDataset: dependence must be in [0, 1]')
  if (!(sharp > 0)) throw new DomainError('twoViewDataset', 'twoViewDataset: sharpness must be positive')
  if (!(Number.isInteger(n) && n >= 2))
    throw new DomainError('twoViewDataset', 'twoViewDataset: n must be an integer of at least 2')
  const ua = toFlat(uniform(child(s, 'a'), 0, 1, { shape: [n, 2] }))
  const ub = toFlat(uniform(child(s, 'b'), 0, 1, { shape: [n, 2] }))
  const uc = toFlat(uniform(child(s, 'copy'), 0, 1, { shape: [n] }))
  const n0 = Math.floor(n / 2)
  const a: number[][] = []
  const b: number[][] = []
  const y: number[] = []
  const copied = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const cls = i < n0 ? 0 : 1
    const a1 = classCoordinate(ua[2 * i], cls, sharp)
    const a2 = 2 * ua[2 * i + 1] - 1
    a.push([a1, a2])
    if (uc[i] < dep) {
      copied[i] = 1
      b.push([-a2, a1])
    } else b.push([2 * ub[2 * i] - 1, classCoordinate(ub[2 * i + 1], cls, sharp)])
    y.push(cls)
  }
  return {
    kind: 'dataset',
    x: fromRows(a.map((r, i) => [...r, ...b[i]])),
    y: labels(y),
    a: fromRows(a),
    b: fromRows(b),
    copied,
    truth: { viewBayesError: (Math.LN2 - Math.log1p(Math.exp(-sharp))) / sharp },
    meta: {
      name: 'two-view',
      description: `${n} points with two views of one binary class (sharpness ${sharp}, dependence ${dep}).`,
      task: 'classification',
      featureNames: ['a1', 'a2', 'b1', 'b2'],
      labelNames: ['0', '1'],
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'twoViewDataset',
    name: 'Two views of one class',
    summary: 'Two uniform views of a binary class, each sufficient, with a dial for how much one copies the other.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(2, 100000, { default: 200 }),
      sharpness: real(0.1, 100, { default: 8 }),
      dependence: real(0, 1, { default: 0 }),
    }),
    truth: false,
    random: true,
    notes: ['co-training'],
    cite: ['blum1998'],
  },
  twoViewDataset,
)
