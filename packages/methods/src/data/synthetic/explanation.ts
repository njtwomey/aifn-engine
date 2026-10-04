/**
 * Test beds for explanation methods, each with a known answer:
 *
 * - `feasibilityTask`: two dense classes joined by a curved corridor of data, with empty space between them, so the
 *   shortest change that crosses the decision boundary lands where no data lie (counterfactuals: Wachter against FACE).
 * - `correlatedEffects`: two strongly correlated features and an additive target x₁ + x₂², so the true effect of each
 *   feature is known while partial dependence must average over combinations that never occur (ALE against PD).
 * - `plantedPatterns`: noise images (8 × 8) or sequences (length 32) in which class 1 carries a motif at a random place;
 *   `t` records where, and `plantedMask` gives the motif's cells (attribution against ground truth).
 * - `conceptImages` and `conceptExamples`: 8 × 8 images built from three visual concepts (horizontal stripes, a corner
 *   dot, a vertical bar), each present at random; the label follows a rule over the concepts, and images that always
 *   hold one concept serve TCAV.
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
 * n points in two dimensions: a class-0 blob at (−1.6, 0.6) (35%), a class-1 blob at (1.6, 0.9) (25%) and a corridor
 * (40%) along the lower arc (−1.6 cos πt, 0.6 − 2 sin πt), labelled 1 where x₁ > 0; Gaussian jitter of sd `noise`
 * (default 0.15) on the corridor and 0.3 on the blobs. The space between the blobs near (0, 0.7) is empty.
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

/** The true effect of feature j (0 or 1) of `correlatedEffects` at v, before centring: v, or v². */
export const correlatedEffectsTerm = (feature: number, v: number): number => (feature === 0 ? v : v * v)

/** The target of `correlatedEffects` without noise: x₁ + x₂². */
export const correlatedEffectsTruth = (x: ArrayLike<number>): number => x[0] + x[1] ** 2

/**
 * n rows of x₁ ~ U(−1, 1) and x₂ = ρx₁ + √(1 − ρ²) z, z ~ U(−1, 1) (correlation ρ, default 0.9), with
 * y = x₁ + x₂² + noise (sd `noise`, default 0.1); `f` holds the noise-free target.
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

/** The noise-free target of `interactionTask`: sin(πx₁) + x₂² + ½x₃ + βx₁x₂. */
export const interactionTaskTruth = (x: ArrayLike<number>, interaction: number): number =>
  Math.sin(Math.PI * x[0]) + x[1] ** 2 + 0.5 * x[2] + interaction * x[0] * x[1]

/**
 * n rows of three independent U(−1, 1) features with y = sin(πx₁) + x₂² + ½x₃ + βx₁x₂ + noise (β = `interaction`,
 * default 1; noise sd default 0.1): three main effects and one pairwise interaction, whose functional ANOVA is known
 * (x₁x₂ has no main effect under independent symmetric features).
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

/** The motif's cells: a 3 × 3 plus in an 8 × 8 image, or a bump of five in a sequence of 32. */
const PLUS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, 2],
  [2, 1],
]
const BUMP = [0.5, 1, 1.5, 1, 0.5]

/** The input shape of a planted-pattern kind. */
export const plantedShape = (kind: 'image' | 'sequence'): [number, number] => (kind === 'image' ? [8, 8] : [1, 32])

/**
 * The motif's weight at each cell [d] of a row whose motif starts at `t` (the top-left cell's index, row-major), or
 * zeros when t < 0.
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
 * n noise rows (sd `noise`, default 0.35) of an 8 × 8 image or a length-32 sequence; class 1 (each row with
 * probability ½) adds a motif of height `strength` (default 1.2) at a uniform position: a 3 × 3 plus in an image, a
 * five-step bump in a sequence. `t` is the motif's first cell (−1 for class 0).
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
export type Concept = (typeof CONCEPTS)[number]

/** The pixels each concept lights in an 8 × 8 image, and its height. */
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

const ruleOf = (rule: ConceptRule, has: Record<Concept, boolean>) =>
  rule === 'stripes'
    ? has.stripes
    : rule === 'dot'
      ? has.dot
      : rule === 'stripes-or-dot'
        ? has.stripes || has.dot
        : has.stripes && has.dot

/**
 * n noisy 8 × 8 images (sd `noise`, default 0.3), each holding each concept (horizontal stripes on rows 1, 4 and 7;
 * a 2 × 2 dot in the top-right corner; a vertical bar on column 3) independently with probability ½; y = 1 when the
 * rule holds (default: stripes). The bar never matters.
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
 * n examples of a concept (noise sd `noise`, default 0.3) as rows [n, 64]: images drawn as `conceptImages` draws them
 * (each other concept present with probability ½) but always holding this one, so that against `random` images (drawn
 * exactly as `conceptImages`) the only systematic difference is the concept: the contrast a CAV needs.
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
