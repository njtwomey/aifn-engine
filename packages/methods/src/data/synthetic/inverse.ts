/**
 * Seeded inverse problems (Bishop, 1994, "Mixture density networks", NCRG/94/004, §4 and §6): a target y is drawn
 * uniformly on a box and observed through a many-to-one map, x = f(y) + ε, and the task is to predict y from x. The
 * inverse of f is multi-valued, so y given x is multimodal, and the least-squares prediction E[y | x] can fall between
 * the solutions. Each generator carries an `InverseTruth` with every solution of f(y) = x.
 *
 * - `bishopInverse`: y = t ~ U(0, 1), x = t + a sin(2πt) + ε, an S on its side with three branches in the middle;
 * - `twoLinkArm`: joint angles of a planar two-link arm from the position of its hand, with the elbow-up and
 *   elbow-down solutions.
 */

import type { DatasetInfo, Size } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { logAddExp } from 'aifn-compute/numerics/special'
import { inverseTruth, type InverseModel, type InverseSolution, type Row } from '../truth'
import { checkCount, generatorRecipe, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

/** Bisection for the root of a continuous g on [a, b] with g(a), g(b) of opposite signs (or zero). */
function bisect(g: (t: number) => number, a: number, b: number): number {
  let lo = a
  let hi = b
  let glo = g(lo)
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi)
    const gm = g(mid)
    if (gm === 0) return mid
    if (gm < 0 === glo < 0) {
      lo = mid
      glo = gm
    } else hi = mid
  }
  return 0.5 * (lo + hi)
}

// ── Bishop's inverse problem ─────────────────────────────────────────────────────────────────────────────────────────

/** Options of `bishopInverse`. */
export interface BishopInverseOptions {
  n?: Size
  /** The sd σ of the noise on x (default 0.05; Bishop used uniform noise on (−0.1, 0.1), sd 0.058). */
  noise?: number
  /** The amplitude a of the sine (default 0.3); above 1/(2π) ≈ 0.16 the map folds and the inverse has three branches. */
  amplitude?: number
}

/**
 * Bishop's inverse problem (1994, §4): t ~ U(0, 1) and x = t + a sin(2πt) + ε, ε ~ N(0, σ²); the dataset's input is x
 * and its target t (the forward problem with the axes swapped). For x in the fold, t has three solutions; the
 * posterior p(t | x) ∝ N(x; t + a sin 2πt, σ²) on [0, 1] is exact up to a 600-point midpoint rule.
 */
export function bishopInverse(s: Stream, options: BishopInverseOptions = {}): Dataset {
  const { n = 400, noise = 0.05, amplitude = 0.3 } = options
  checkCount(n, 'bishopInverse')
  if (!(noise >= 0)) throw new DomainError('bishopInverse', 'bishopInverse: noise must be ≥ 0')
  const a = amplitude
  const f = (t: number) => t + a * Math.sin(2 * Math.PI * t)
  const slope = (t: number) => 1 + 2 * Math.PI * a * Math.cos(2 * Math.PI * t)
  // The turning points of f on [0, 1], where cos 2πt = −1/(2πa); they split [0, 1] into monotone pieces.
  const c = -1 / (2 * Math.PI * a)
  const turns = Math.abs(c) < 1 ? [Math.acos(c) / (2 * Math.PI), 1 - Math.acos(c) / (2 * Math.PI)] : []
  const pieces = [0, ...turns, 1]
  const solutions = (x: Row): InverseSolution[] => {
    const found: number[] = []
    for (let p = 0; p + 1 < pieces.length; p++) {
      const g = (t: number) => f(t) - x[0]
      const [lo, hi] = [pieces[p], pieces[p + 1]]
      if (g(lo) * g(hi) <= 0) {
        const t = bisect(g, lo, hi)
        if (!found.some((u) => Math.abs(u - t) < 1e-9)) found.push(t)
      }
    }
    const w = found.map((t) => 1 / Math.max(1e-12, Math.abs(slope(t))))
    const total = w.reduce((acc, v) => acc + v, 0)
    return found.map((t, i) => ({ value: [t], weight: w[i] / total })).sort((p, q) => q.weight - p.weight)
  }
  const Q = 600
  const nodes = Array.from({ length: Q }, (_, i) => (i + 0.5) / Q)
  const logKernel = (x: number, t: number) => -0.5 * ((x - f(t)) / noise) ** 2 - Math.log(noise) - LOG_SQRT_2PI
  const model: InverseModel = {
    name: "Bishop's inverse problem",
    outputs: 1,
    forward: (y) => [f(y[0])],
    solutions,
    atoms: (x) => {
      if (noise === 0) {
        const sol = solutions(x)
        return { values: sol.map((p) => p.value), weights: sol.map((p) => p.weight) }
      }
      const logs = nodes.map((t) => logKernel(x[0], t))
      const m = Math.max(...logs)
      return { values: nodes.map((t) => [t]), weights: logs.map((l) => Math.exp(l - m)) }
    },
    ...(noise > 0
      ? {
          logLikelihood: (x: Row, y: Row) => {
            if (!(y[0] >= 0 && y[0] <= 1)) return -Infinity
            // p(t | x) = N(x; f(t), σ²) / ∫₀¹ N(x; f(u), σ²) du, the prior being 1 on [0, 1].
            return (
              logKernel(x[0], y[0]) -
              (nodes.reduce((acc, t) => logAddExp(acc, logKernel(x[0], t)) as number, -Infinity) - Math.log(Q))
            )
          },
        }
      : {}),
    noise,
    lower: [0],
    upper: [1],
    formula: `x = t + ${a} sin(2πt) + ε`,
  }
  const ts = child(s, 't')
  const eps = child(s, 'noise')
  const t = Float64Array.from({ length: n }, () => uniform(ts) as number)
  const x = Float64Array.from(t, (v) => f(v) + noise * (normal(eps) as number))
  return {
    kind: 'dataset',
    x: matrix(x, n, 1),
    y: vector(t),
    meta: {
      name: 'bishop inverse',
      description: `${n} points of t ~ U(0, 1) observed through x = t + ${a} sin(2πt) + ε (σ ${noise}), to predict t from x: up to three values of t share one x.`,
      task: 'regression',
      featureNames: ['x'],
      targetName: 't',
      source: 'Bishop (1994), "Mixture density networks", §4',
      key: s.key,
      truth: inverseTruth(model),
      recipe: generatorRecipe('bishopInverse', s.key, { n, noise, amplitude }),
    },
  }
}

// ── Two-link arm ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The joint positions of a planar two-link arm with its shoulder at the origin: the elbow and the hand. */
export function twoLinkJoints(
  angles: Row,
  lengths: readonly [number, number],
): { elbow: [number, number]; hand: [number, number] } {
  const [t1, t2] = [angles[0], angles[1]]
  const elbow: [number, number] = [lengths[0] * Math.cos(t1), lengths[0] * Math.sin(t1)]
  return {
    elbow,
    hand: [elbow[0] + lengths[1] * Math.cos(t1 + t2), elbow[1] + lengths[1] * Math.sin(t1 + t2)],
  }
}

/**
 * Both solutions of a two-link arm's inverse kinematics at hand position p: θ₂ = ±arccos((|p|² − l₁² − l₂²)/(2l₁l₂))
 * (elbow one way, then the other) and θ₁ = atan2(p₂, p₁) − atan2(l₂ sin θ₂, l₁ + l₂ cos θ₂). Empty outside the
 * reachable annulus l₁ − l₂ ≤ |p| ≤ l₁ + l₂; one solution on its boundary.
 */
export function twoLinkInverse(position: Row, lengths: readonly [number, number]): [number, number][] {
  const [l1, l2] = lengths
  const r2 = position[0] ** 2 + position[1] ** 2
  const c = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
  if (!(Math.abs(c) <= 1)) return []
  const base = Math.atan2(position[1], position[0])
  const out: [number, number][] = []
  for (const sign of c === 1 || c === -1 ? [1] : [1, -1]) {
    const t2 = sign * Math.acos(c)
    out.push([base - Math.atan2(l2 * Math.sin(t2), l1 + l2 * Math.cos(t2)), t2])
  }
  return out
}

/** Options of `twoLinkArm`. */
export interface TwoLinkArmOptions {
  n?: Size
  /** The upper arm's length l₁ (default 0.8). */
  l1?: number
  /** The forearm's length l₂ (default 0.5). */
  l2?: number
  /** The sd of the noise on the hand position (default 0.01). */
  noise?: number
}

/** The prior box of the joint angles (radians): the shoulder θ₁ and the elbow θ₂ (both bends). */
const SHOULDER: readonly [number, number] = [0, 1.8]
const ELBOW: readonly [number, number] = [-2.4, 2.4]

const wrapInto = (v: number, [lo, hi]: readonly [number, number]) => {
  let w = v
  while (w < lo) w += 2 * Math.PI
  while (w > hi) w -= 2 * Math.PI
  return w >= lo && w <= hi ? w : NaN
}

/**
 * Inverse kinematics of a planar two-link arm (Bishop, 1994, §6, with both bends of the elbow): joint angles
 * θ₁ ~ U(0, 1.8) at the shoulder and θ₂ ~ U(−2.4, 2.4) at the elbow, the hand at
 * p = (l₁ cos θ₁ + l₂ cos(θ₁ + θ₂), l₁ sin θ₁ + l₂ sin(θ₁ + θ₂)) + ε, and the task is to predict (θ₁, θ₂) from p.
 * Most positions are reached with the elbow bent either way; the truth lists both solutions inside the prior's box,
 * which have equal weight in the small-noise limit (|det J| = l₁l₂|sin θ₂| is the same for both).
 */
export function twoLinkArm(s: Stream, options: TwoLinkArmOptions = {}): Dataset {
  const { n = 600, l1 = 0.8, l2 = 0.5, noise = 0.01 } = options
  checkCount(n, 'twoLinkArm')
  if (!(l1 > 0 && l2 > 0)) throw new DomainError('twoLinkArm', 'twoLinkArm: link lengths must be positive')
  const lengths: [number, number] = [l1, l2]
  const solutions = (x: Row): InverseSolution[] => {
    const kept = twoLinkInverse(x, lengths)
      .map(([t1, t2]) => [wrapInto(t1, SHOULDER), t2])
      .filter(([t1, t2]) => Number.isFinite(t1) && t2 >= ELBOW[0] && t2 <= ELBOW[1])
    return kept.map((value) => ({ value, weight: 1 / kept.length }))
  }
  const model: InverseModel = {
    name: 'two-link arm',
    outputs: 2,
    forward: (y) => twoLinkJoints(y, lengths).hand,
    solutions,
    atoms: (x) => {
      const sol = solutions(x)
      return { values: sol.map((p) => p.value), weights: sol.map((p) => p.weight) }
    },
    noise,
    lower: [SHOULDER[0], ELBOW[0]],
    upper: [SHOULDER[1], ELBOW[1]],
    formula: `p = (${l1} cos θ₁ + ${l2} cos(θ₁ + θ₂), ${l1} sin θ₁ + ${l2} sin(θ₁ + θ₂)) + ε`,
  }
  const as = child(s, 'angles')
  const eps = child(s, 'noise')
  const angles = new Float64Array(2 * n)
  const hands = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    angles[2 * i] = SHOULDER[0] + (SHOULDER[1] - SHOULDER[0]) * (uniform(as) as number)
    angles[2 * i + 1] = ELBOW[0] + (ELBOW[1] - ELBOW[0]) * (uniform(as) as number)
    const { hand } = twoLinkJoints(angles.subarray(2 * i, 2 * i + 2), lengths)
    hands[2 * i] = hand[0] + noise * (normal(eps) as number)
    hands[2 * i + 1] = hand[1] + noise * (normal(eps) as number)
  }
  return {
    kind: 'dataset',
    x: matrix(hands, n, 2),
    y: matrix(angles, n, 2),
    meta: {
      name: 'two-link arm',
      description: `${n} hand positions of a two-link arm (links ${l1} and ${l2}, noise sd ${noise}) with the joint angles that produced them, to predict the angles from the position: the elbow can bend either way.`,
      task: 'regression',
      featureNames: ['p1', 'p2'],
      targetName: 'θ',
      source: 'Bishop (1994), "Mixture density networks", §6',
      key: s.key,
      truth: inverseTruth(model),
      recipe: generatorRecipe('twoLinkArm', s.key, { n, l1, l2, noise }),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const datasetEntry = definer<DatasetInfo>('dataset', 'data/synthetic')
const NOTES = ['gaussian-mixture-model']

datasetEntry(
  {
    key: 'bishopInverse',
    name: "Bishop's inverse problem",
    summary: 't ~ U(0, 1) observed through x = t + 0.3 sin(2πt) + ε; predicting t from x has up to three answers.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 400 }),
      noise: real(0, 0.5, { default: 0.05 }),
      amplitude: real(0, 1, { default: 0.3 }),
    }),
    truth: true,
    random: true,
    notes: NOTES,
    cite: ['bishop2006'],
  },
  bishopInverse,
)
datasetEntry(
  {
    key: 'twoLinkArm',
    name: 'Two-link arm inverse kinematics',
    summary: 'Joint angles of a planar two-link arm from its hand position; the elbow can bend either way.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 600 }),
      l1: real(0.1, 2, { default: 0.8 }),
      l2: real(0.1, 2, { default: 0.5 }),
      noise: real(0, 0.2, { default: 0.01 }),
    }),
    truth: true,
    random: true,
    notes: NOTES,
    cite: ['bishop2006'],
  },
  twoLinkArm,
)
