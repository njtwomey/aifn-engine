/**
 * Test beds for explanation methods, each with a known answer:
 *
 * - `feasibilityTask`: two dense classes joined by a curved corridor of data, with empty space between them, so the
 *   shortest change that crosses the decision boundary lands where no data lie (counterfactuals: Wachter against FACE).
 * - `correlatedEffects`: two strongly correlated features and an additive target $x_1 + x_2^2$, so the true effect of
 *   each feature is known while partial dependence must average over combinations that never occur (ALE against PD).
 * - `interactionTask`: three independent features with three main effects and one pairwise interaction of chosen
 *   strength, so the functional ANOVA decomposition is known (main effects and interactions against the truth).
 * - `plantedPatterns`: noise images ($8 \times 8$) or sequences (length 32) in which class 1 carries a motif at a random
 *   place; `t` records where, and `plantedMask` gives the motif's cells (attribution against ground truth).
 * - `conceptImages` and `conceptExamples`: $8 \times 8$ images built from three visual concepts (horizontal stripes, a
 *   corner dot, a vertical bar), each present at random; the label follows a rule over the concepts, and images that
 *   always hold one concept serve TCAV.
 *
 * Every generator draws from `child(s, 'rows')` (and the regression tasks their noise from `child(s, 'noise')`), so the
 * same stream gives the same data, and throws `DomainError` when `n` is not a non-negative integer.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { fromData } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { checkCount, labels, matrix, type Dataset } from '../types'

// ── Feasibility ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $n$ points in two dimensions: a class-0 blob at $(-1.6, 0.6)$ (35%), a class-1 blob at $(1.6, 0.9)$ (25%) and a
 * corridor (40%) along the lower arc $(-1.6 \cos \pi t, 0.6 - 2 \sin \pi t)$, $t \sim \Unif(0, 1)$, labelled 1 where
 * the jittered $x_1 > 0$; Gaussian jitter of standard deviation `noise` on the corridor and 0.3 on the blobs. Each row
 * picks its part with one uniform draw. The space between the blobs near $(0, 0.7)$ is empty, so the shortest move
 * across the boundary from a class-0 point lands where there are no data.
 *
 * @param s The stream the rows are drawn from.
 * @param options `n`, the number of points (default 300), and `noise`, the jitter's standard deviation on the
 *   corridor (default 0.15; the blobs keep 0.3).
 * @returns The points in `x` ($n \times 2$) and the class labels in `y` (int32, 0 or 1).
 *
 * @example The blobs, the corridor and the empty middle
 * const d = feasibilityTask(stream(1), { n: 200 })
 * const x = toArray(d.x)
 * print('x:', d.x.shape, ' y:', d.y.shape)
 * print('first rows:', x.slice(0, 3), ' labels:', toArray(d.y).slice(0, 3))
 * print('points within 0.4 of (0, 0.7):', x.filter(([a, b]) => Math.hypot(a, b - 0.7) < 0.4).length)
 * print('points below x2 = -0.5 (the corridor):', x.filter(([, b]) => b < -0.5).length)
 */
export function feasibilityTask(s: Stream, options: { n?: number; noise?: number } = {}): Dataset {
  const { n = 300, noise = 0.15 } = options
  checkCount(n, 'feasibilityTask')
  const x = new Float64Array(n * 2)
  const y = new Int32Array(n)
  const r = child(s, 'rows')
  for (let i = 0; i < n; i++) {
    const u = uniform(r) as number
    const e1 = normal(r) as number
    const e2 = normal(r) as number
    if (u < 0.35) {
      x[2 * i] = -1.6 + 0.3 * e1
      x[2 * i + 1] = 0.6 + 0.3 * e2
      y[i] = 0
    } else if (u < 0.6) {
      x[2 * i] = 1.6 + 0.3 * e1
      x[2 * i + 1] = 0.9 + 0.3 * e2
      y[i] = 1
    } else {
      const t = uniform(r) as number
      x[2 * i] = -1.6 * Math.cos(Math.PI * t) + noise * e1
      x[2 * i + 1] = 0.6 - 2 * Math.sin(Math.PI * t) + noise * e2
      y[i] = x[2 * i] > 0 ? 1 : 0
    }
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: labels(y),
    meta: {
      name: 'feasibility task',
      description: `${n} points: a class-0 blob on the left, a class-1 blob on the right, and a corridor of data along a lower arc between them (labelled by the side it is on); the space between the blobs is empty.`,
      task: 'classification',
      featureNames: ['x₁', 'x₂'],
      labelNames: ['0', '1'],
      key: s.key,
    },
  }
}

// ── Correlated effects ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The true effect of a feature of `correlatedEffects` at a value $v$, before centring: $v$ for $x_1$, $v^2$ for $x_2$.
 *
 * @param feature The feature's index: 0 for $x_1$; any other value is taken as $x_2$.
 * @param v The feature's value.
 * @returns The feature's term of the target, $v$ or $v^2$.
 *
 * @example The two terms at the same value
 * print('x1 term at 0.5:', correlatedEffectsTerm(0, 0.5))
 * print('x2 term at 0.5:', correlatedEffectsTerm(1, 0.5))
 */
export const correlatedEffectsTerm = (feature: number, v: number): number => (feature === 0 ? v : v * v)

/**
 * The target of `correlatedEffects` without noise: $x_1 + x_2^2$.
 *
 * @param x A row of two features, $x_1$ then $x_2$; only the first two entries are read.
 * @returns $x_1 + x_2^2$.
 *
 * @example The noise-free target matches `f`
 * const d = correlatedEffects(stream(1), { n: 3 })
 * const x = toArray(d.x)
 * print('truth:', x.map(correlatedEffectsTruth))
 * print('f:', d.f)
 */
export const correlatedEffectsTruth = (x: ArrayLike<number>): number => x[0] + x[1] ** 2

/**
 * $n$ rows of $x_1 \sim \Unif(-1, 1)$ and $x_2 = \rho x_1 + \sqrt{1 - \rho^2}\, z$ with $z \sim \Unif(-1, 1)$, so
 * $\corr(x_1, x_2) = \rho$, and $y = x_1 + x_2^2 + \varepsilon$, $\varepsilon \sim \Gauss(0, \sigma^2)$. Throws
 * `DomainError` when the correlation is outside $[0, 1]$.
 *
 * @param s The stream the rows (`child(s, 'rows')`) and the noise (`child(s, 'noise')`) are drawn from.
 * @param options `n`, the number of rows (default 400); `correlation`, $\rho$ (default 0.9); and `noise`, the noise's
 *   standard deviation $\sigma$ (default 0.1).
 * @returns The features in `x` ($n \times 2$), the noisy target in `y` and the noise-free target in `f`.
 *
 * @example The features are correlated as asked
 * const d = correlatedEffects(stream(1), { n: 500, correlation: 0.9 })
 * const x = toArray(d.x)
 * print('x:', d.x.shape, ' first rows:', x.slice(0, 2))
 * const m = (v) => v.reduce((a, b) => a + b, 0) / v.length
 * const [a, b] = [x.map((r) => r[0]), x.map((r) => r[1])]
 * const c = m(a.map((v, i) => v * b[i])) - m(a) * m(b)
 * const sa = Math.sqrt(m(a.map((v) => v * v)) - m(a) ** 2)
 * const sb = Math.sqrt(m(b.map((v) => v * v)) - m(b) ** 2)
 * print('sample correlation:', c / (sa * sb))
 */
export function correlatedEffects(
  s: Stream,
  options: { n?: number; correlation?: number; noise?: number } = {},
): Dataset {
  const { n = 400, correlation = 0.9, noise = 0.1 } = options
  checkCount(n, 'correlatedEffects')
  if (!(correlation >= 0 && correlation <= 1))
    throw new DomainError('correlatedEffects', 'correlatedEffects: correlation must lie in [0, 1]')
  const r = child(s, 'rows')
  const e = child(s, 'noise')
  const x = new Float64Array(n * 2)
  const y = new Float64Array(n)
  const f = new Float64Array(n)
  const c = Math.sqrt(1 - correlation ** 2)
  for (let i = 0; i < n; i++) {
    const a = 2 * (uniform(r) as number) - 1
    const z = 2 * (uniform(r) as number) - 1
    x[2 * i] = a
    x[2 * i + 1] = correlation * a + c * z
    f[i] = correlatedEffectsTruth(x.subarray(2 * i, 2 * i + 2))
    y[i] = f[i] + noise * (normal(e) as number)
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 2),
    y: fromData(y, [n]),
    f: fromData(f, [n]),
    meta: {
      name: 'correlated effects',
      description: `${n} rows with x₁ ~ U(−1, 1) and x₂ = ${correlation}x₁ + √(1 − ${correlation}²) z, z ~ U(−1, 1); y = x₁ + x₂² + noise (sd ${noise}).`,
      task: 'regression',
      featureNames: ['x₁', 'x₂'],
      targetName: 'y',
      key: s.key,
    },
  }
}

// ── An additive model with an interaction ───────────────────────────────────────────────────────────────────────────

/**
 * The noise-free target of `interactionTask`: $\sin(\pi x_1) + x_2^2 + \tfrac{1}{2} x_3 + \beta x_1 x_2$.
 *
 * @param x A row of three features, $x_1$, $x_2$ and $x_3$; only the first three entries are read.
 * @param interaction The interaction's coefficient $\beta$.
 * @returns The target at `x`.
 *
 * @example The interaction is all that differs from the additive part
 * print('beta = 0:', interactionTaskTruth([0.5, 0.5, 0], 0))
 * print('beta = 2:', interactionTaskTruth([0.5, 0.5, 0], 2))
 */
export const interactionTaskTruth = (x: ArrayLike<number>, interaction: number): number =>
  Math.sin(Math.PI * x[0]) + x[1] ** 2 + 0.5 * x[2] + interaction * x[0] * x[1]

/**
 * $n$ rows of three independent $\Unif(-1, 1)$ features with
 * $y = \sin(\pi x_1) + x_2^2 + \tfrac{1}{2} x_3 + \beta x_1 x_2 + \varepsilon$, $\varepsilon \sim \Gauss(0, \sigma^2)$:
 * three main effects and one pairwise interaction, whose functional ANOVA is known ($x_1 x_2$ has no main effect under
 * independent symmetric features).
 *
 * @param s The stream the rows (`child(s, 'rows')`) and the noise (`child(s, 'noise')`) are drawn from.
 * @param options `n`, the number of rows (default 500); `interaction`, $\beta$ (default 1); and `noise`, the noise's
 *   standard deviation $\sigma$ (default 0.1).
 * @returns The features in `x` ($n \times 3$), the noisy target in `y` and the noise-free target in `f`.
 *
 * @example The noise is what separates y from f
 * const d = interactionTask(stream(1), { n: 400, noise: 0.1 })
 * print('x:', d.x.shape, ' first row:', toArray(d.x)[0])
 * const y = toArray(d.y)
 * const f = toArray(d.f)
 * print('RMS of y - f:', Math.sqrt(y.reduce((a, v, i) => a + (v - f[i]) ** 2, 0) / y.length))
 */
export function interactionTask(
  s: Stream,
  options: { n?: number; interaction?: number; noise?: number } = {},
): Dataset {
  const { n = 500, interaction = 1, noise = 0.1 } = options
  checkCount(n, 'interactionTask')
  const r = child(s, 'rows')
  const e = child(s, 'noise')
  const x = new Float64Array(n * 3)
  const y = new Float64Array(n)
  const f = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) x[3 * i + j] = 2 * (uniform(r) as number) - 1
    f[i] = interactionTaskTruth(x.subarray(3 * i, 3 * i + 3), interaction)
    y[i] = f[i] + noise * (normal(e) as number)
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 3),
    y: fromData(y, [n]),
    f: fromData(f, [n]),
    meta: {
      name: 'interaction task',
      description: `${n} rows of three independent U(−1, 1) features; y = sin(πx₁) + x₂² + ½x₃ + ${interaction}x₁x₂ + noise (sd ${noise}).`,
      task: 'regression',
      featureNames: ['x₁', 'x₂', 'x₃'],
      targetName: 'y',
      key: s.key,
    },
  }
}

// ── Planted patterns ─────────────────────────────────────────────────────────────────────────────────────────────────

/** The motif's cells: a $3 \times 3$ plus in an $8 \times 8$ image, or a bump of five in a sequence of 32. */
const PLUS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, 2],
  [2, 1],
]
const BUMP = [0.5, 1, 1.5, 1, 0.5]

/**
 * The input shape of a planted-pattern kind: $8 \times 8$ for an image, $1 \times 32$ for a sequence.
 *
 * @param kind `'image'` or `'sequence'`.
 * @returns The height and width.
 *
 * @example The two shapes
 * print('image:', plantedShape('image'), ' sequence:', plantedShape('sequence'))
 */
export const plantedShape = (kind: 'image' | 'sequence'): [number, number] => (kind === 'image' ? [8, 8] : [1, 32])

/**
 * The motif's weight at each of the $d$ cells of a row whose motif starts at `t`, or zeros when $t < 0$: 1 on the five
 * cells of a plus for an image, the bump $0.5, 1, 1.5, 1, 0.5$ for a sequence. Positions are not checked: an image
 * motif whose column is above 5 wraps onto the next row.
 *
 * @param kind `'image'` or `'sequence'`.
 * @param t Where the motif starts, as `plantedPatterns` records it: for an image the row-major index of the top-left
 *   cell of the plus's $3 \times 3$ box, for a sequence its first step; $-1$ (no motif) gives zeros.
 * @returns The $d$ weights, row-major ($d = 64$ for an image, 32 for a sequence).
 *
 * @example The plus of a motif placed at row 1, column 2
 * const m = plantedMask('image', 1 * 8 + 2)
 * print('cells:', [...m.keys()].filter((p) => m[p] > 0).map((p) => [Math.floor(p / 8), p % 8]))
 * print('bump from step 3:', plantedMask('sequence', 3).slice(0, 10))
 */
export function plantedMask(kind: 'image' | 'sequence', t: number): Float64Array {
  const [h, w] = plantedShape(kind)
  const out = new Float64Array(h * w)
  if (t < 0) return out
  if (kind === 'image') {
    const r0 = Math.floor(t / w)
    const c0 = t % w
    for (const [dr, dc] of PLUS) out[(r0 + dr) * w + c0 + dc] = 1
  } else BUMP.forEach((v, k) => (out[t + k] = v))
  return out
}

/**
 * $n$ rows of Gaussian noise of an $8 \times 8$ image or a length-32 sequence; class 1 (each row with probability
 * $\tfrac{1}{2}$) adds `strength` times the motif of `plantedMask` at a uniform position: a $3 \times 3$ plus in an
 * image (its box's top-left cell in rows and columns 0 to 5), a five-step bump in a sequence (starting at step 0 to
 * 27), so the motif always fits.
 *
 * @param s The stream the rows are drawn from.
 * @param options `n`, the number of rows (default 800); `kind`, `'image'` (default) or `'sequence'`; `noise`, the
 *   noise's standard deviation (default 0.35); and `strength`, the scale of the motif (default 1.2).
 * @returns The flattened inputs in `x` ($n \times 64$ or $n \times 32$, row-major), the labels in `y` (int32, 1 where
 *   a motif was planted) and the motif's start in `t` ($-1$ for class 0), for `plantedMask`.
 *
 * @example The motif sits where t says
 * const d = plantedPatterns(stream(1), { n: 6 })
 * print('x:', d.x.shape, ' y:', toArray(d.y), ' t:', toArray(d.t))
 * const i = toArray(d.y).indexOf(1)
 * const row = toArray(d.x)[i]
 * const mask = plantedMask('image', toArray(d.t)[i])
 * print('mean on the motif:', row.filter((_, p) => mask[p] > 0).reduce((a, v) => a + v, 0) / 5)
 * print('mean elsewhere:', row.filter((_, p) => mask[p] === 0).reduce((a, v) => a + v, 0) / 59)
 */
export function plantedPatterns(
  s: Stream,
  options: { n?: number; kind?: 'image' | 'sequence'; noise?: number; strength?: number } = {},
): Dataset {
  const { n = 800, kind = 'image', noise = 0.35, strength = 1.2 } = options
  checkCount(n, 'plantedPatterns')
  const [h, w] = plantedShape(kind)
  const d = h * w
  const x = new Float64Array(n * d)
  const y = new Int32Array(n)
  const t = new Float64Array(n).fill(-1)
  const r = child(s, 'rows')
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < d; p++) x[i * d + p] = noise * (normal(r) as number)
    if ((uniform(r) as number) < 0.5) continue
    y[i] = 1
    const start =
      kind === 'image'
        ? Math.floor((uniform(r) as number) * 6) * w + Math.floor((uniform(r) as number) * 6)
        : Math.floor((uniform(r) as number) * (w - BUMP.length + 1))
    t[i] = start
    const mask = plantedMask(kind, start)
    for (let p = 0; p < d; p++) x[i * d + p] += strength * mask[p]
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: labels(y),
    t: fromData(t, [n]),
    meta: {
      name: kind === 'image' ? 'planted plus' : 'planted bump',
      description:
        kind === 'image'
          ? `${n} 8 × 8 noise images (sd ${noise}); class 1 carries a 3 × 3 plus of height ${strength} at a random place (t: its top-left pixel).`
          : `${n} length-32 noise sequences (sd ${noise}); class 1 carries a five-step bump of height ${strength} at a random place (t: its first step).`,
      task: 'classification',
      featureNames: Array.from({ length: d }, (_, p) =>
        kind === 'image' ? `pixel ${Math.floor(p / w)},${p % w}` : `step ${p}`,
      ),
      labelNames: ['no motif', 'motif'],
      key: s.key,
    },
  }
}

// ── Concepts ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The visual concepts of `conceptImages`. */
export const CONCEPTS = ['stripes', 'dot', 'bar'] as const
/** One of the visual concepts: horizontal stripes, a corner dot or a vertical bar. */
export type Concept = (typeof CONCEPTS)[number]

/**
 * Paint a concept onto an $8 \times 8$ image: stripes add 0.9 on rows 1, 4 and 7; the dot adds 1.4 on the
 * $2 \times 2$ block of rows 0 and 1, columns 6 and 7; the bar adds 0.9 on column 3.
 *
 * @param img The image, 64 values row-major; modified in place.
 * @param concept The concept to add.
 */
function paint(img: Float64Array, concept: Concept): void {
  if (concept === 'stripes') for (const r of [1, 4, 7]) for (let c = 0; c < 8; c++) img[r * 8 + c] += 0.9
  else if (concept === 'dot')
    for (const [r, c] of [
      [0, 6],
      [0, 7],
      [1, 6],
      [1, 7],
    ])
      img[r * 8 + c] += 1.4
  else for (let r = 0; r < 8; r++) img[r * 8 + 3] += 0.9
}

/** The label rule of `conceptImages`: which concepts make class 1. */
export type ConceptRule = 'stripes' | 'dot' | 'stripes-or-dot' | 'stripes-and-dot'

/**
 * Whether a label rule holds for the concepts an image holds.
 *
 * @param rule The rule: one concept, or the stripes and the dot combined by or or by and.
 * @param has Which concepts the image holds.
 * @returns True when the image is class 1 under the rule.
 */
const ruleOf = (rule: ConceptRule, has: Record<Concept, boolean>) =>
  rule === 'stripes'
    ? has.stripes
    : rule === 'dot'
      ? has.dot
      : rule === 'stripes-or-dot'
        ? has.stripes || has.dot
        : has.stripes && has.dot

/**
 * $n$ noisy $8 \times 8$ images, each holding each concept (horizontal stripes on rows 1, 4 and 7; a $2 \times 2$ dot
 * in the top-right corner; a vertical bar on column 3) independently with probability $\tfrac{1}{2}$; $y = 1$ when the
 * rule holds. The bar never matters.
 *
 * @param s The stream the images are drawn from.
 * @param options `n`, the number of images (default 800); `rule`, the label rule (default `'stripes'`); and `noise`,
 *   the standard deviation of the Gaussian noise under the concepts (default 0.3).
 * @returns The images in `x` ($n \times 64$, row-major) and the labels in `y` (int32).
 *
 * @example Under the default rule the label is whether row 4 is lit
 * const d = conceptImages(stream(1), { n: 8 })
 * const x = toArray(d.x)
 * print('x:', d.x.shape, ' y:', toArray(d.y))
 * print('mean of row 4:', x.map((r) => r.slice(32, 40).reduce((a, v) => a + v, 0) / 8))
 */
export function conceptImages(s: Stream, options: { n?: number; rule?: ConceptRule; noise?: number } = {}): Dataset {
  const { n = 800, rule = 'stripes', noise = 0.3 } = options
  checkCount(n, 'conceptImages')
  const x = new Float64Array(n * 64)
  const y = new Int32Array(n)
  const r = child(s, 'rows')
  for (let i = 0; i < n; i++) {
    const img = new Float64Array(64)
    for (let p = 0; p < 64; p++) img[p] = noise * (normal(r) as number)
    const has = { stripes: false, dot: false, bar: false } as Record<Concept, boolean>
    for (const c of CONCEPTS) {
      has[c] = (uniform(r) as number) < 0.5
      if (has[c]) paint(img, c)
    }
    y[i] = ruleOf(rule, has) ? 1 : 0
    x.set(img, i * 64)
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 64),
    y: labels(y),
    meta: {
      name: 'concept images',
      description: `${n} 8 × 8 images (noise sd ${noise}) with stripes, a corner dot and a vertical bar each present at random; y = 1 when ${rule.replaceAll('-', ' ')}.`,
      task: 'classification',
      featureNames: Array.from({ length: 64 }, (_, p) => `pixel ${Math.floor(p / 8)},${p % 8}`),
      labelNames: ['0', '1'],
      key: s.key,
    },
  }
}

/**
 * $n$ examples of a concept: images drawn as `conceptImages` draws them (each other concept present with probability
 * $\tfrac{1}{2}$) but always holding this one, so that against `'random'` images (drawn exactly as `conceptImages`) the
 * only systematic difference is the concept: the contrast a CAV needs. The examples have no labels.
 *
 * @param s The stream the images are drawn from.
 * @param concept The concept every image holds, or `'random'` for the images of `conceptImages` with their labels
 *   removed.
 * @param options `n`, the number of images (default 50), and `noise`, the noise's standard deviation (default 0.3).
 * @returns The images in `x` ($n \times 64$, row-major).
 *
 * @example Every dot example lights the top-right corner
 * const dots = toArray(conceptExamples(stream(1), 'dot', { n: 20 }).x)
 * const random = toArray(conceptExamples(stream(2), 'random', { n: 20 }).x)
 * const corner = (rows) => rows.reduce((a, r) => a + r[7], 0) / rows.length
 * print('x:', [dots.length, dots[0].length])
 * print('mean of pixel (0, 7), dot:', corner(dots), ' random:', corner(random))
 */
export function conceptExamples(
  s: Stream,
  concept: Concept | 'random',
  options: { n?: number; noise?: number } = {},
): Dataset {
  const { n = 50, noise = 0.3 } = options
  checkCount(n, 'conceptExamples')
  if (concept === 'random') return { ...conceptImages(s, { n, noise }), y: undefined }
  const x = new Float64Array(n * 64)
  const r = child(s, 'rows')
  for (let i = 0; i < n; i++) {
    const img = new Float64Array(64)
    for (let p = 0; p < 64; p++) img[p] = noise * (normal(r) as number)
    for (const c of CONCEPTS) if (c === concept || (uniform(r) as number) < 0.5) paint(img, c)
    x.set(img, i * 64)
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, 64),
    meta: {
      name: `${concept} examples`,
      description: `${n} noisy 8 × 8 images (noise sd ${noise}) that all hold the ${concept} concept, the others at random.`,
      task: 'images',
      featureNames: Array.from({ length: 64 }, (_, p) => `pixel ${Math.floor(p / 8)},${p % 8}`),
      key: s.key,
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'feasibilityTask',
    name: 'Feasibility task',
    summary: 'Two classes joined by a curved corridor of data with empty space between them, for counterfactuals.',
    task: 'classification',
    output: 'dataset',
    knobs: space({ n: int(20, 5000, { default: 300 }), noise: real(0, 1, { default: 0.15 }) }),
    truth: false,
    random: true,
    notes: ['counterfactual-explanations'],
  },
  feasibilityTask,
)
dataset(
  {
    key: 'correlatedEffects',
    name: 'Correlated effects',
    summary:
      'Two correlated features with an additive target x₁ + x₂²: known effects where partial dependence extrapolates.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(20, 10000, { default: 400 }),
      correlation: real(0, 1, { default: 0.9 }),
      noise: real(0, 1, { default: 0.1 }),
    }),
    truth: false,
    random: true,
    notes: ['interpretability'],
  },
  correlatedEffects,
)
dataset(
  {
    key: 'interactionTask',
    name: 'Interaction task',
    summary: 'Three independent features with main effects and one pairwise interaction of chosen strength.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(20, 10000, { default: 500 }),
      interaction: real(-3, 3, { default: 1 }),
      noise: real(0, 1, { default: 0.1 }),
    }),
    truth: false,
    random: true,
    notes: ['interpreting-generalised-additive-models', 'interpretability'],
  },
  interactionTask,
)
dataset(
  {
    key: 'plantedPatterns',
    name: 'Planted patterns',
    summary: 'Noise images or sequences in which class 1 carries a motif at a random place, recorded in t.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(20, 10000, { default: 800 }),
      kind: oneOf(['image', 'sequence'], { default: 'image' }),
      noise: real(0, 2, { default: 0.35 }),
      strength: real(0, 3, { default: 1.2 }),
    }),
    truth: false,
    random: true,
    notes: ['feature-attribution-methods', 'evaluating-explanations'],
  },
  plantedPatterns,
)
dataset(
  {
    key: 'conceptImages',
    name: 'Concept images',
    summary: 'Images built from stripes, a corner dot and a bar, labelled by a rule over them, for TCAV.',
    task: 'classification',
    output: 'dataset',
    knobs: space({
      n: int(20, 10000, { default: 800 }),
      rule: oneOf(['stripes', 'dot', 'stripes-or-dot', 'stripes-and-dot'], { default: 'stripes' }),
      noise: real(0, 2, { default: 0.3 }),
    }),
    truth: false,
    random: true,
    notes: ['concept-based-explanations'],
  },
  conceptImages,
)
dataset(
  {
    key: 'conceptExamples',
    name: 'Concept examples',
    summary: 'Images that all hold one concept (the others at random), or random images from the same distribution.',
    task: 'images',
    output: 'dataset',
    knobs: space({ n: int(2, 5000, { default: 50 }), noise: real(0, 2, { default: 0.3 }) }),
    truth: false,
    random: true,
    notes: ['concept-based-explanations'],
  },
  conceptExamples,
)
