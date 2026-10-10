/**
 * Seeded data for flows that split, cross and spread (Twomey, Kozłowski & Santos-Rodríguez, 2020, "Neural ODEs with
 * stochastic vector field mixtures", ECAI):
 *
 * - `odeFailureCase`: the 1-d problems of the paper's fig. 1 that a neural ODE cannot solve. *Crossing*: starts near
 *   $-1$ and $+1$ must swap places ($x \mapsto -x$), but a 1-d flow is increasing. *Splitting*: one start maps to two
 *   targets, $-1$ or $+1$. *Scaling*: one start maps to targets spread along the same direction, at log-normal
 *   distances. Each row is a start $x(0)$ with its target $x(1)$.
 * - `floorplanWalks`: a synthetic stand-in for the paper's behavioural data (section 4.1.2, fig. 12), which is not
 *   available:
 *   noisy human-like walks in a house from one origin (the sofa in the living room) to four targets (the front door,
 *   the kitchen, the landing at the foot of the stairs, the study), through the doorways, so paths share their first
 *   leg and branch where they diverge. Walking speed varies by walk, walkers accelerate and slow down, sometimes pause,
 *   and stay at the target once there. Each row is one walk sampled at regular times over a 10-second window, then the
 *   hour of the day; by day the four targets are equally likely, by night the landing (0.9) and the kitchen (0.1)
 *   (the paper's section 4.4 counterfactual).
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { checkCount, generatorRecipe, labels, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── Fig. 1 failure cases ─────────────────────────────────────────────────────────────────────────────────────────────

/** The three 1-d failure cases of fig. 1. */
export type OdeFailureKind = 'crossing' | 'splitting' | 'scaling'

/** Options of `odeFailureCase`. */
export interface OdeFailureOptions {
  /** Which failure case to draw. Default `splitting`. */
  kind?: OdeFailureKind
  /** Number of starts. Default 200. */
  n?: number
  /**
   * The standard deviation of the Gaussian jitter on the starts, and of an independent jitter on the targets of
   * splitting (a crossing target is exactly $-x(0)$). Default 0.05.
   */
  noise?: number
  /** Scaling: the standard deviation of the log distance travelled (median distance 2). Default 0.25. */
  spread?: number
}

/**
 * A 1-d failure case of a neural ODE (fig. 1): starts $x(0)$ (`x`, $n \times 1$) and their targets $x(1)$ (`y`,
 * length $n$). Crossing: $x(0) \approx \mp 1$, $x(1) = -x(0)$. Splitting: $x(0) \approx 0$, $x(1) \approx \pm 1$ with
 * equal probability. Scaling: $x(0) \approx -1$, $x(1) = x(0) + L$ with $\log L \sim \Gauss(\log 2, \sigma^2)$,
 * $\sigma$ = `spread`. The branches come from `child(s, 'branch')`, the jitter from `child(s, 'noise')`. Throws
 * `DomainError` when `n` is not a non-negative integer.
 *
 * @param s The stream the branches and the jitter are drawn from.
 * @param options The kind, the number of starts, the jitter and the spread of the scaling distances
 *   (`OdeFailureOptions`).
 * @returns The starts in `x` ($n \times 1$) and the targets in `y`.
 *
 * @example Crossing: every target is the start reflected
 * const d = odeFailureCase(stream(1), { kind: 'crossing', n: 6 })
 * const [x, y] = [toArray(d.x).map(([v]) => v), toArray(d.y)]
 * print('starts: ', x)
 * print('targets:', y)
 *
 * @example Splitting and scaling
 * const split = toArray(odeFailureCase(stream(1), { n: 400 }).y)
 * print('share of splitting targets above 0 (1/2):', split.filter((v) => v > 0).length / split.length)
 * const d = odeFailureCase(stream(2), { kind: 'scaling', n: 401 })
 * const x = toArray(d.x)
 * const gaps = toArray(d.y).map((v, i) => v - x[i][0]).sort((a, b) => a - b)
 * print('median scaling distance (2):', gaps[200])
 */
export function odeFailureCase(s: Stream, options: OdeFailureOptions = {}): Dataset {
  const { kind = 'splitting', n = 200, noise = 0.05, spread = 0.25 } = options
  checkCount(n, 'odeFailureCase')
  const u = units(child(s, 'branch'), n)
  const e = child(s, 'noise')
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const jitter = () => noise * (normal(e) as number)
    if (kind === 'crossing') {
      const side = u[i] < 0.5 ? -1 : 1
      x[i] = side + jitter()
      y[i] = -x[i]
    } else if (kind === 'splitting') {
      x[i] = jitter()
      y[i] = (u[i] < 0.5 ? -1 : 1) + jitter()
    } else {
      x[i] = -1 + jitter()
      y[i] = x[i] + 2 * Math.exp(spread * (normal(e) as number))
    }
  }
  const what = {
    crossing: 'starts near −1 and +1 swap places (x ↦ −x)',
    splitting: 'one start near 0 goes to −1 or +1',
    scaling: 'one start near −1 travels a log-normal distance to the right',
  }[kind]
  return {
    kind: 'dataset',
    x: matrix(x, n, 1),
    y: vector(y),
    meta: {
      name: `neural ODE ${kind}`,
      description: `${n} starts and targets of the ${kind} problem: ${what}.`,
      task: 'regression',
      featureNames: ['x(0)'],
      targetName: 'x(1)',
      source: 'Twomey, Kozłowski & Santos-Rodríguez (2020), fig. 1',
      key: s.key,
      recipe: generatorRecipe('odeFailureCase', s.key, { kind, n, noise, spread }),
    },
  }
}

// ── Floorplan walks ──────────────────────────────────────────────────────────────────────────────────────────────────

/** A point in metres. */
export type FloorPoint = readonly [number, number]

/** The house the walks happen in (metres; x to the right, y up). */
export type Floorplan = {
  /** The house's width in metres. */
  width: number
  /** The house's height in metres. */
  height: number
  /** Wall segments $[x_1, y_1, x_2, y_2]$, from one end to the other. */
  walls: readonly (readonly [number, number, number, number])[]
  /** Furniture and stair treads, as segments. */
  furniture: readonly (readonly [number, number, number, number])[]
  /** Room names at label positions. */
  rooms: readonly { name: string; at: FloorPoint }[]
  /** Where every walk starts (the sofa). */
  origin: FloorPoint
  /** The four targets, in label order. */
  targets: readonly { name: string; at: FloorPoint }[]
}

/**
 * The synthetic house: living room bottom left, hall and front door top left, dining room and study in the middle,
 * stairs top middle, kitchen on the right.
 */
export const FLOORPLAN: Floorplan = {
  width: 12,
  height: 7,
  walls: [
    [0, 0, 12, 0],
    [12, 0, 12, 7],
    [12, 7, 0, 7],
    [0, 7, 0, 6.6],
    [0, 5.4, 0, 0],
    // Living room | hall (doorway x 3.0–4.4).
    [0, 4.5, 3.0, 4.5],
    [4.4, 4.5, 5, 4.5],
    // Living room | study (doorway y 1.4–2.7).
    [5, 0, 5, 1.4],
    [5, 2.7, 5, 4.5],
    // Study | stairs.
    [5, 4.5, 8.5, 4.5],
    // Study | kitchen (doorway y 1.8–3.2) and stairs | kitchen.
    [8.5, 0, 8.5, 1.8],
    [8.5, 3.2, 8.5, 7],
  ],
  furniture: [
    // Sofa.
    [0.3, 0.8, 1.0, 0.8],
    [1.0, 0.8, 1.0, 2.6],
    [1.0, 2.6, 0.3, 2.6],
    [0.3, 2.6, 0.3, 0.8],
    // Desk in the study.
    [6.2, 3.4, 7.8, 3.4],
    [7.8, 3.4, 7.8, 4.2],
    [7.8, 4.2, 6.2, 4.2],
    [6.2, 4.2, 6.2, 3.4],
    // Kitchen counter.
    [11.3, 0.4, 11.3, 6.6],
    // Stair treads.
    ...Array.from({ length: 6 }, (_, i) => [7.4 + 0.18 * i, 4.9, 7.4 + 0.18 * i, 6.8] as const),
  ],
  rooms: [
    { name: 'living room', at: [2.5, 0.4] },
    { name: 'hall', at: [2.4, 6.6] },
    { name: 'study', at: [6.8, 0.4] },
    { name: 'stairs', at: [6.5, 6.6] },
    { name: 'kitchen', at: [10, 0.4] },
  ],
  origin: [1.6, 1.7],
  targets: [
    { name: 'front door', at: [0.45, 6.0] },
    { name: 'kitchen', at: [10.6, 4.4] },
    { name: 'landing', at: [6.9, 5.8] },
    { name: 'study', at: [6.8, 1.4] },
  ],
}

/** Waypoints of each route after the origin, with the sd of their jitter per axis (doorways jitter along the gap). */
const ROUTES: readonly (readonly { at: FloorPoint; sd: FloorPoint }[])[] = [
  [
    { at: [3.7, 4.5], sd: [0.25, 0.05] },
    { at: [2.2, 5.9], sd: [0.25, 0.2] },
  ],
  [
    { at: [5.0, 2.05], sd: [0.05, 0.25] },
    { at: [8.5, 2.5], sd: [0.05, 0.25] },
  ],
  [
    { at: [3.7, 4.5], sd: [0.25, 0.05] },
    { at: [5.0, 5.6], sd: [0.2, 0.25] },
  ],
  [{ at: [5.0, 2.05], sd: [0.05, 0.25] }],
]

/**
 * A uniform Catmull–Rom spline through points (the end points repeated as their own neighbours), sampled densely;
 * returns the samples and their cumulative lengths.
 *
 * @param points The points the curve passes through, in order; at least one.
 * @param perSegment Samples per segment between consecutive points (the last point is added once at the end).
 * @returns `out`, the samples, and `cum`, the arc length from the first sample to each, by straight segments.
 */
function smoothPath(points: readonly FloorPoint[], perSegment = 24) {
  const pts = [points[0], ...points, points[points.length - 1]]
  const out: [number, number][] = []
  for (let i = 1; i + 2 < pts.length; i++) {
    const [p0, p1, p2, p3] = [pts[i - 1], pts[i], pts[i + 1], pts[i + 2]]
    for (let k = 0; k < perSegment; k++) {
      const t = k / perSegment
      const t2 = t * t
      const t3 = t2 * t
      const at = (j: 0 | 1) =>
        0.5 *
        (2 * p1[j] +
          (-p0[j] + p2[j]) * t +
          (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
          (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3)
      out.push([at(0), at(1)])
    }
  }
  out.push([...points[points.length - 1]] as [number, number])
  const cum = [0]
  for (let i = 1; i < out.length; i++)
    cum.push(cum[i - 1] + Math.hypot(out[i][0] - out[i - 1][0], out[i][1] - out[i - 1][1]))
  return { out, cum }
}

/** Options of `floorplanWalks`. */
export interface FloorplanWalksOptions {
  /** Walks. Default 160. */
  n?: number
  /** Samples per walk over the window, an integer of at least 2. Default 51. */
  samples?: number
  /** The window in seconds. Default 10. */
  duration?: number
  /** The standard deviation of position noise per sample, in metres (SLAM error). Default 0.03. */
  noise?: number
  /** Probability that a walk pauses once on the way. Default 0.3. */
  pause?: number
  /**
   * When the walks happen: by day (targets equally likely), by night (landing 0.9, kitchen 0.1) or both, half each (the
   * odd-numbered walks by night). Default `day`.
   */
  timeOfDay?: 'day' | 'night' | 'mixed'
}

const ROUTE_NAMES = FLOORPLAN.targets.map((t) => t.name)
const NIGHT = [0, 0.1, 0.9, 0]

/**
 * Walks in `FLOORPLAN` from the origin to one of four targets (labels 0 front door, 1 kitchen, 2 landing, 3 study).
 * Features: $x_0, y_0, x_1, y_1, \dots$ (metres, at `samples` regular times from 0 to `duration` seconds), then the
 * hour of the day. Day walks cycle through the targets in turn (walk $i$ goes to target $i \bmod 4$, so each gets
 * $n/4$) at an hour uniform from 8 to 20; night walks draw their target (landing 0.9, kitchen 0.1) at an hour from 22
 * to 6. Each walk follows a spline through jittered waypoints (the doorways), with a mean speed of 0.9 to 1.4 m/s
 * (raised when needed to arrive within 92% of the window), a delay of up to 0.6 s standing up, with probability
 * `pause` one pause of 0.5 to 1.5 s, and a smoothstep in time so it speeds up and slows down. Every walk draws from
 * its own `child(s, 'walk', i)`. Throws `DomainError` when `samples` is not an integer of at least 2 or `n` is not a
 * non-negative integer.
 *
 * @param s The stream the walks are drawn from.
 * @param options The number of walks, the sampling, the noise, the pause probability and the time of day
 *   (`FloorplanWalksOptions`).
 * @returns The walks in `x` ($n \times (2M + 1)$ for $M$ samples: the positions, then the hour) and the targets in `y`
 *   (int32).
 *
 * @example Walks start at the sofa and end at their target
 * const d = floorplanWalks(stream(1), { n: 8, samples: 11 })
 * const x = toArray(d.x)
 * print('x:', d.x.shape, ' targets:', toArray(d.y))
 * print('walk 0, first positions:', x[0].slice(0, 6), ' hour:', x[0][22])
 * print('walk 1 ends at', x[1].slice(20, 22), ' kitchen:', FLOORPLAN.targets[1].at)
 *
 * @example By night most walks go to the landing
 * const y = toArray(floorplanWalks(stream(1), { n: 200, samples: 5, timeOfDay: 'night' }).y)
 * print('share to the landing (0.9):', y.filter((v) => v === 2).length / y.length)
 */
export function floorplanWalks(s: Stream, options: FloorplanWalksOptions = {}): Dataset {
  const { n = 160, samples = 51, duration = 10, noise = 0.03, pause = 0.3, timeOfDay = 'day' } = options
  checkCount(n, 'floorplanWalks')
  if (!(Number.isInteger(samples) && samples >= 2))
    throw new DomainError('floorplanWalks', 'floorplanWalks: samples must be ≥ 2')
  const M = samples
  const x = new Float64Array(n * (2 * M + 1))
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const w = child(s, 'walk', i)
    const night = timeOfDay === 'night' || (timeOfDay === 'mixed' && i % 2 === 1)
    const hour = night
      ? (22 + 8 * (uniform(child(w, 'hour')) as number)) % 24
      : 8 + 12 * (uniform(child(w, 'hour')) as number)
    // The target: uniform by day; the night's weights by night. Day walks cycle the targets so each gets n/4.
    let target = i % 4
    if (night) {
      const u = uniform(child(w, 'target')) as number
      let acc = 0
      target = 3
      for (let k = 0; k < 4; k++) {
        acc += NIGHT[k]
        if (u < acc) {
          target = k
          break
        }
      }
    }
    y[i] = target
    const jit = child(w, 'jitter')
    const g = () => normal(jit) as number
    const origin: FloorPoint = [FLOORPLAN.origin[0] + 0.1 * g(), FLOORPLAN.origin[1] + 0.1 * g()]
    const goal = FLOORPLAN.targets[target].at
    const end: FloorPoint = [goal[0] + 0.12 * g(), goal[1] + 0.12 * g()]
    const via = ROUTES[target].map((p) => [p.at[0] + p.sd[0] * g(), p.at[1] + p.sd[1] * g()] as FloorPoint)
    const { out, cum } = smoothPath([origin, ...via, end])
    const L = cum[cum.length - 1]
    const r = units(child(w, 'timing'), 4)
    // Mean speed 0.9–1.4 m/s, a delay standing up, maybe one pause; the walk ends inside the window.
    let speed = 0.9 + 0.5 * r[0]
    const delay = 0.6 * r[1]
    const pauseLength = r[2] < pause ? 0.5 + 1.0 * r[3] : 0
    const pauseAt = 0.25 + 0.5 * (uniform(child(w, 'pause')) as number)
    speed = Math.max(speed, L / (0.92 * duration - delay - pauseLength))
    const moving = L / speed
    const noiseStream = child(w, 'noise')
    for (let j = 0; j < M; j++) {
      const t = (j / (M - 1)) * duration - delay
      const tp = pauseAt * moving
      const tau = t <= 0 ? 0 : t < tp ? t : t < tp + pauseLength ? tp : t - pauseLength
      const q = Math.min(1, tau / moving)
      // Smoothstep: the walker speeds up from rest and slows down at the target.
      const sArc = L * q * q * (3 - 2 * q)
      let k = 1
      while (k < cum.length - 1 && cum[k] < sArc) k++
      const span = cum[k] - cum[k - 1] || 1
      const a = Math.min(1, Math.max(0, (sArc - cum[k - 1]) / span))
      const px = out[k - 1][0] + a * (out[k][0] - out[k - 1][0])
      const py = out[k - 1][1] + a * (out[k][1] - out[k - 1][1])
      x[i * (2 * M + 1) + 2 * j] = px + noise * (normal(noiseStream) as number)
      x[i * (2 * M + 1) + 2 * j + 1] = py + noise * (normal(noiseStream) as number)
    }
    x[i * (2 * M + 1) + 2 * M] = hour
  }
  const featureNames = [...Array.from({ length: M }, (_, j) => [`x${j}`, `y${j}`]).flat(), 'hour']
  return {
    kind: 'dataset',
    x: matrix(x, n, 2 * M + 1),
    y: labels(y),
    meta: {
      name: 'floorplan walks (synthetic)',
      description: `${n} synthetic walks in a house from the sofa to the ${ROUTE_NAMES.join(', ')}, ${M} samples over ${duration} s each (then the hour); ${timeOfDay === 'day' ? 'by day, targets equally likely' : timeOfDay === 'night' ? 'by night: landing 0.9, kitchen 0.1' : 'half by day, half by night'}.`,
      task: 'classification',
      featureNames,
      targetName: 'target',
      labelNames: ROUTE_NAMES,
      source: 'synthetic analogue of Twomey, Kozłowski & Santos-Rodríguez (2020), §4.1.2 and fig. 12',
      key: s.key,
      recipe: generatorRecipe('floorplanWalks', s.key, { n, samples, duration, noise, pause, timeOfDay }),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const datasetEntry = definer<DatasetInfo>('dataset', 'data/synthetic')
const NOTES = ['neural-ordinary-differential-equations']

datasetEntry(
  {
    key: 'odeFailureCase',
    name: 'Neural ODE failure cases (crossing, splitting, scaling)',
    summary:
      '1-d starts and targets a neural ODE cannot map: swapping places, one start to two targets, spread distances.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      kind: oneOf(['crossing', 'splitting', 'scaling'], { default: 'splitting' }),
      n: int(2, 5000, { default: 200 }),
      noise: real(0, 0.5, { default: 0.05 }),
      spread: real(0, 1, { default: 0.25 }),
    }),
    truth: false,
    random: true,
    notes: NOTES,
    cite: ['dupont2019'],
  },
  odeFailureCase,
)
datasetEntry(
  {
    key: 'floorplanWalks',
    name: 'Floorplan walks (synthetic behaviour)',
    summary:
      'Synthetic walks in a house from the sofa to the front door, kitchen, landing or study, branching at doorways.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(4, 2000, { default: 160 }),
      samples: int(2, 400, { default: 51 }),
      duration: real(1, 60, { default: 10 }),
      noise: real(0, 0.5, { default: 0.03 }),
      pause: real(0, 1, { default: 0.3 }),
      timeOfDay: oneOf(['day', 'night', 'mixed'], { default: 'day' }),
    }),
    truth: false,
    random: true,
    notes: NOTES,
    cite: [],
  },
  floorplanWalks,
)
