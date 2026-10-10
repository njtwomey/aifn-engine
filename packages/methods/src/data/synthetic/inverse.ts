/**
 * Seeded inverse problems (Bishop, 1994, "Mixture density networks", NCRG/94/004, §4 and §6): a target $\yvec$ is
 * drawn uniformly on a box and observed through a many-to-one map, $\xvec = f(\yvec) + \epsilonvec$, and the task is
 * to predict $\yvec$ from $\xvec$. The inverse of $f$ is multi-valued, so $\yvec$ given $\xvec$ is multimodal, and the
 * least-squares prediction $\expect[\yvec \mid \xvec]$ can fall between the solutions. Each generator carries an
 * `InverseTruth` with every solution of $f(\yvec) = \xvec$.
 *
 * - `bishopInverse`: $y = t \sim \Unif(0, 1)$, $x = t + a \sin(2\pi t) + \varepsilon$, an S on its side with three
 *   branches in the middle;
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

/** $\log\sqrt{2\pi}$, the constant of the Gaussian log-density. */
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

/**
 * Bisection for the root of a continuous $g$ on $[a, b]$ with $g(a)$, $g(b)$ of opposite signs (or zero): 80 halvings,
 * so the bracket is below double precision.
 *
 * @param g The function whose root is sought.
 * @param a The left end of the bracket.
 * @param b The right end of the bracket.
 * @returns A root of $g$ in $[a, b]$: the midpoint of the final bracket, or a point where $g$ is exactly 0.
 */
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
  /** Points (default 400). */
  n?: Size
  /**
   * The standard deviation $\sigma$ of the noise on $x$, at least 0 (default 0.05; Bishop used uniform noise on
   * $(-0.1, 0.1)$, standard deviation 0.058).
   */
  noise?: number
  /**
   * The amplitude $a$ of the sine (default 0.3); above $1/(2\pi) \approx 0.16$ the map folds and the inverse has three
   * branches.
   */
  amplitude?: number
}

/**
 * Bishop's inverse problem (1994, §4): $t \sim \Unif(0, 1)$ and $x = t + a \sin(2\pi t) + \varepsilon$,
 * $\varepsilon \sim \Gauss(0, \sigma^2)$; the dataset's input is $x$ and its target $t$ (the forward problem with the
 * axes swapped). For $x$ in the fold, $t$ has three solutions, weighted by $1/\abs{f'(t)}$; the posterior
 * $p(t \mid x) \propto \Gauss(x; t + a \sin 2\pi t, \sigma^2)$ on $[0, 1]$ is exact up to a 600-point midpoint rule
 * (with $\sigma = 0$ it is the solutions themselves). Throws `DomainError` when $n$ is not a non-negative integer or
 * the noise is negative.
 *
 * @param s The stream the targets (child `'t'`) and the noise (child `'noise'`) are drawn from.
 * @param options The size, the noise and the amplitude of the sine.
 * @returns A regression dataset with `x` ($n \times 1$) and the targets `y` ($n$ values of $t$), and the truth
 *   (`meta.truth`, an `InverseTruth`).
 *
 * @example One input, three answers
 * const d = bishopInverse(stream(1), { n: 400 })
 * const [x, t] = [toArray(d.x).map((r) => r[0]), toArray(d.y)]
 * print('x:', d.x.shape, ' first (x, t):', [0, 1, 2].map((i) => [x[i], t[i]]))
 * const f = (v) => v + 0.3 * Math.sin(2 * Math.PI * v)
 * print('sd of x - f(t):', Math.sqrt(x.reduce((a, v, i) => a + (v - f(t[i])) ** 2, 0) / x.length))
 * print('solutions at x = 0.5:', d.meta.truth.solutions([0.5]).map((p) => p.value[0]))
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

/**
 * The joint positions of a planar two-link arm with its shoulder at the origin: the elbow and the hand. The shoulder
 * angle $\theta_1$ is measured from the $x_1$ axis and the elbow angle $\theta_2$ relative to the upper arm.
 *
 * @param angles The joint angles $(\theta_1, \theta_2)$ in radians; only the first two entries are read.
 * @param lengths The link lengths $(l_1, l_2)$: the upper arm, then the forearm.
 * @returns The elbow at $l_1(\cos\theta_1, \sin\theta_1)$ and the hand,
 *   $l_2(\cos(\theta_1 + \theta_2), \sin(\theta_1 + \theta_2))$ beyond it.
 *
 * @example A straight arm along the axis, then the elbow bent a right angle
 * const straight = twoLinkJoints([0, 0], [0.8, 0.5])
 * print('straight, elbow:', straight.elbow, ' hand:', straight.hand)
 * const bent = twoLinkJoints([0, Math.PI / 2], [0.8, 0.5])
 * print('bent, elbow:', bent.elbow, ' hand:', bent.hand)
 */
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
 * Both solutions of a two-link arm's inverse kinematics at hand position $\pvec$:
 * $\theta_2 = \pm\arccos((\norm{\pvec}^2 - l_1^2 - l_2^2)/(2l_1l_2))$ (elbow one way, then the other) and
 * $\theta_1 = \operatorname{atan2}(p_2, p_1) - \operatorname{atan2}(l_2 \sin\theta_2, l_1 + l_2 \cos\theta_2)$. Empty
 * outside the reachable annulus $\abs{l_1 - l_2} \le \norm{\pvec} \le l_1 + l_2$; one solution on its boundary. The
 * angles are not wrapped into any range.
 *
 * @param position The hand position $(p_1, p_2)$; only the first two entries are read.
 * @param lengths The link lengths $(l_1, l_2)$.
 * @returns The solutions as $(\theta_1, \theta_2)$ pairs: two (positive $\theta_2$ first), one, or none.
 *
 * @example Forward then back: both bends reach the same hand
 * const { hand } = twoLinkJoints([0.6, 1.2], [0.8, 0.5])
 * print('hand:', hand)
 * const both = twoLinkInverse(hand, [0.8, 0.5])
 * print('solutions:', both)
 * print('their hands:', both.map((a) => twoLinkJoints(a, [0.8, 0.5]).hand))
 * print('out of reach:', twoLinkInverse([2, 0], [0.8, 0.5]))
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
  /** Points (default 600). */
  n?: Size
  /** The upper arm's length $l_1$ (default 0.8). */
  l1?: number
  /** The forearm's length $l_2$ (default 0.5). */
  l2?: number
  /** The standard deviation of the noise on each coordinate of the hand position (default 0.01). */
  noise?: number
}

/** The prior range of the shoulder angle $\theta_1$ (radians). */
const SHOULDER: readonly [number, number] = [0, 1.8]
/** The prior range of the elbow angle $\theta_2$ (radians), covering both bends. */
const ELBOW: readonly [number, number] = [-2.4, 2.4]

/**
 * An angle moved by whole turns into a range, or NaN when no turn of it falls inside.
 *
 * @param v The angle, in radians.
 * @param range The range as its low and high ends, in radians.
 * @returns $v + 2\pi k$ for the integer $k$ that puts it in the range, or NaN.
 */
const wrapInto = (v: number, [lo, hi]: readonly [number, number]) => {
  let w = v
  while (w < lo) w += 2 * Math.PI
  while (w > hi) w -= 2 * Math.PI
  return w >= lo && w <= hi ? w : NaN
}

/**
 * Inverse kinematics of a planar two-link arm (Bishop, 1994, §6, with both bends of the elbow): joint angles
 * $\theta_1 \sim \Unif(0, 1.8)$ at the shoulder and $\theta_2 \sim \Unif(-2.4, 2.4)$ at the elbow, the hand at
 * $\pvec = (l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2),\ l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2))$ plus
 * noise $\epsilonvec$, and the task is to predict $(\theta_1, \theta_2)$ from $\pvec$. Most positions are reached with
 * the elbow bent either way; the truth lists both solutions inside the prior's box, which have equal weight in the
 * small-noise limit ($\abs{\det \Jmat} = l_1l_2\abs{\sin\theta_2}$ is the same for both). Throws `DomainError` when $n$
 * is not a non-negative integer or a link length is not positive.
 *
 * @param s The stream the angles (child `'angles'`) and the noise (child `'noise'`) are drawn from.
 * @param options The size, the link lengths and the noise.
 * @returns A regression dataset: `x` the hand positions ($n \times 2$), `y` the angles ($n \times 2$), and the truth
 *   (`meta.truth`, an `InverseTruth`).
 *
 * @example Many hand positions have two answers inside the prior's box
 * const d = twoLinkArm(stream(1), { n: 600 })
 * const [x, y] = [toArray(d.x), toArray(d.y)]
 * print('x:', d.x.shape, ' y:', d.y.shape)
 * print('first hand:', x[0], ' its angles:', y[0])
 * const counts = x.map((p) => d.meta.truth.solutions(p).length)
 * // The noise can move a hand to where no angles in the box reach it.
 * print('hands with 0, 1 and 2 solutions:', [0, 1, 2].map((k) => counts.filter((c) => c === k).length))
 * const i = counts.indexOf(2)
 * print('hand', i, x[i], ' drawn from', y[i], ' solutions:', d.meta.truth.solutions(x[i]).map((p) => p.value))
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
