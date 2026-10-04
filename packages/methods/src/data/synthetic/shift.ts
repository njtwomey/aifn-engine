/**
 * Generators for transfer, continual and meta-learning: a source and a shifted target domain of two moons (covariate
 * shift by rotation, translation or scaling), classes drawn with different priors in two domains (label shift), a
 * sequence of related classification tasks for continual learning. (The sinusoid tasks of MAML are part of
 * `aifn-methods/learning/transfer`'s meta-learning set-up.)
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

/** Two labelled point sets. */
export interface DomainPair {
  readonly source: { x: Tensor; y: Tensor }
  readonly target: { x: Tensor; y: Tensor }
}

/** n two-moons points (labels 0 and 1) with Gaussian noise. */
function moonPoints(s: Stream, n: number, noise: number): { x: Float64Array; y: Int32Array } {
  const u = units(child(s, 'angle'), n)
  const e = standardNormals(child(s, 'noise'), 2 * n)
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const c = i % 2
    const t = Math.PI * u[i]
    const px = c === 0 ? Math.cos(t) : 1 - Math.cos(t)
    const py = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t)
    // Centred on the origin.
    x[2 * i] = px - 0.5 + noise * e[2 * i]
    x[2 * i + 1] = py - 0.25 + noise * e[2 * i + 1]
    y[i] = c
  }
  return { x, y }
}

/** Options of `shiftedMoons`. */
export interface ShiftedMoonsOptions {
  /** Points per domain (default 400). */
  n?: number
  /** The shift: `rotation` by `amount` radians, `translation` by `amount` along (1, 1)/√2, or `scale` by 1 + amount. */
  shift?: 'rotation' | 'translation' | 'scale'
  amount?: number
  noise?: number
}

/** Two moons in a source domain and transformed in a target domain: covariate shift with the same labelling rule. */
export function shiftedMoons(s: Stream, options: ShiftedMoonsOptions = {}): DomainPair {
  const { n = 400, shift = 'rotation', amount = 0.6, noise = 0.1 } = options
  const src = moonPoints(child(s, 'source'), n, noise)
  const tgt = moonPoints(child(s, 'target'), n, noise)
  const map = (x: number, y: number): [number, number] => {
    if (shift === 'rotation')
      return [Math.cos(amount) * x - Math.sin(amount) * y, Math.sin(amount) * x + Math.cos(amount) * y]
    if (shift === 'translation') return [x + amount / Math.SQRT2, y + amount / Math.SQRT2]
    return [(1 + amount) * x, (1 + amount) * y]
  }
  for (let i = 0; i < n; i++) [tgt.x[2 * i], tgt.x[2 * i + 1]] = map(tgt.x[2 * i], tgt.x[2 * i + 1])
  return {
    source: { x: fromData(src.x, [n, 2]), y: fromData(src.y, [n]) },
    target: { x: fromData(tgt.x, [n, 2]), y: fromData(tgt.y, [n]) },
  }
}

/** Options of `labelShiftDomains`. */
export interface LabelShiftOptions {
  n?: number
  /** Class priors in the source and target domains (default [1/3, 1/3, 1/3] and [0.6, 0.3, 0.1]). */
  sourcePriors?: readonly number[]
  targetPriors?: readonly number[]
  /** Distance between the class means, which sit on a regular polygon (default 2.5). */
  separation?: number
}

/** Gaussian classes in 2-d with the same p(x | y) in both domains and different priors p(y): label shift. */
export function labelShiftDomains(s: Stream, options: LabelShiftOptions = {}): DomainPair & { classes: number } {
  const { n = 1000, sourcePriors = [1 / 3, 1 / 3, 1 / 3], targetPriors = [0.6, 0.3, 0.1], separation = 2.5 } = options
  const k = sourcePriors.length
  const r = separation / (2 * Math.sin(Math.PI / k))
  const draw = (stream: Stream, priors: readonly number[]) => {
    const u = units(child(stream, 'labels'), n)
    const e = standardNormals(child(stream, 'noise'), 2 * n)
    const x = new Float64Array(2 * n)
    const y = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      let c = 0
      let acc = priors[0]
      while (u[i] > acc && c < k - 1) acc += priors[++c]
      y[i] = c
      x[2 * i] = r * Math.cos((2 * Math.PI * c) / k) + e[2 * i]
      x[2 * i + 1] = r * Math.sin((2 * Math.PI * c) / k) + e[2 * i + 1]
    }
    return { x: fromData(x, [n, 2]), y: fromData(y, [n]) }
  }
  return { source: draw(child(s, 'source'), sourcePriors), target: draw(child(s, 'target'), targetPriors), classes: k }
}

/**
 * A sequence of binary tasks on 2-d inputs: two Gaussian blobs per task, in a region of its own, whose axis turns by
 * `turn` radians from task to task. A network can solve all of them together, but training on one at a time moves the
 * boundaries of the others.
 */
export function rotatingTasks(
  s: Stream,
  options: { tasks?: number; n?: number; turn?: number; separation?: number; spread?: number } = {},
): { x: Tensor; y: Tensor }[] {
  const { tasks = 3, n = 300, turn = Math.PI / 2, separation = 1.5, spread = 2.5 } = options
  return Array.from({ length: tasks }, (_, t) => {
    const a = t * turn
    const e = standardNormals(child(s, 'task', t), 2 * n)
    const x = new Float64Array(2 * n)
    const y = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      const c = i % 2
      const m = (c === 0 ? -1 : 1) * (separation / 2)
      // The two classes sit at ±m along the task's axis; a second, offset cluster per class makes the task non-linear.
      // Each task lives in its own region (centres on a circle of radius `spread`), so all tasks can be solved at once.
      const cx = tasks > 1 ? spread * Math.cos((2 * Math.PI * t) / tasks) : 0
      const cy = tasks > 1 ? spread * Math.sin((2 * Math.PI * t) / tasks) : 0
      x[2 * i] = cx + m * Math.cos(a) + 0.4 * e[2 * i]
      x[2 * i + 1] = cy + m * Math.sin(a) + 0.4 * e[2 * i + 1]
      y[i] = c
    }
    return { x: fromData(x, [n, 2]), y: fromData(y, [n]) }
  })
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'shiftedMoons',
    name: 'Shifted two moons',
    summary: 'Two moons in a source domain and rotated, translated or scaled in a target domain (covariate shift).',
    role: 'simulation',
    random: true,
    notes: ['domain-adversarial-training', 'domain-adaptation-theory'],
  },
  shiftedMoons,
)
fn(
  {
    key: 'labelShiftDomains',
    name: 'Label-shifted domains',
    summary: 'Gaussian classes with the same class-conditionals and different class priors in two domains.',
    role: 'simulation',
    random: true,
    notes: ['label-shift-and-target-shift'],
  },
  labelShiftDomains,
)
fn(
  {
    key: 'rotatingTasks',
    name: 'Rotating task sequence',
    summary: 'Binary tasks whose class axis turns from task to task, for continual learning.',
    role: 'simulation',
    random: true,
    notes: ['continual-learning', 'regularisation-based-continual-learning'],
  },
  rotatingTasks,
)
