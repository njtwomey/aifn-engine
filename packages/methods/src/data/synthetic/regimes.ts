/**
 * Seeded data with regimes: a gate over $x$ chooses which of $K$ simple functions generated $y$, the generative model
 * of a mixture of experts (Jacobs, Jordan, Nowlan and Hinton, 1991). Every generator returns the regime of each row
 * (`regime`, int32) and a `RegimeTruth` with the true gate, so a figure can score how well a fitted gate recovers the
 * regimes (the adjusted Rand index of its assignments against `regime`).
 *
 * - `piecewiseLinear`: 1-D, $K$ lines on consecutive intervals with jumps at known breakpoints;
 * - `quadrantPlanes`: 2-D, a plane per quadrant (regression) or a linear boundary per quadrant (classification);
 * - `interleavedFunctions`: 1-D, two lines that take turns over alternating bands of $x$;
 * - `regressionMixture`: 1-D, $K$ lines with a soft gate, so the regimes overlap and $y$ given $x$ is multimodal.
 *
 * The inputs are uniform on a box (from `child(s, 'x')`, sorted when 1-D), the regimes are drawn from the gate
 * (`child(s, 'regime')`) and the noise from `child(s, 'noise')`; `f` holds the regime's noise-free function at each
 * row. Every generator throws `DomainError` when `n` is not a non-negative integer.
 */

import type { DatasetInfo, Size } from 'aifn-compute/foundation/contracts'
import { categorical, child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { regimeTruth, type RegimeModel, type Row } from '../truth'
import { checkCount, generatorRecipe, labels, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A dataset with the regime that generated each row. */
export type RegimeDataset = Dataset & {
  /** The regime of each row (int32 [n]), drawn from the true gate. */
  readonly regime: Tensor
}

/**
 * A coefficient for a caption: two decimals, or `'0'` when it rounds to zero.
 *
 * @param v The coefficient.
 * @returns The text.
 */
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2))
/**
 * A line $a + bx$ as caption text, with the sign of $b$ written as an operator.
 *
 * @param a The intercept.
 * @param b The slope.
 * @param x The name of the variable.
 * @returns The text, e.g. `'0.60 + 0.80x'`.
 */
const line = (a: number, b: number, x = 'x') => `${fmt(a)} ${b < 0 ? '−' : '+'} ${fmt(Math.abs(b))}${x}`

/**
 * Draw $n$ rows from a regime model: $\xvec$ uniform on the box, the regime from the gate (taken directly when the gate
 * is one-hot), and $y$ from the regime: its function plus Gaussian noise (regression), or a Bernoulli draw of
 * $\sigma(f)$ (classification).
 *
 * @param s The stream the inputs (`child(s, 'x')`), the regimes (`child(s, 'regime')`) and the noise
 *   (`child(s, 'noise')`) are drawn from.
 * @param model The regime model: its box, gate, functions and noise.
 * @param n The number of rows.
 * @param sorted Whether to sort the rows by $x$ (applied only when the input is 1-D).
 * @returns The inputs `x` ($n \times d$), the targets or labels `y`, the regimes' noise-free values `f` (the log-odds
 *   for classification) and the regime of each row.
 */
function draw(s: Stream, model: RegimeModel, n: Size, sorted: boolean) {
  const d = model.lower.length
  const xs = child(s, 'x')
  const rows: number[][] = []
  for (let i = 0; i < n; i++)
    rows.push(Array.from({ length: d }, (_, c) => model.lower[c] + (model.upper[c] - model.lower[c]) * uniform(xs)))
  if (sorted && d === 1) rows.sort((a, b) => a[0] - b[0])
  const pick = child(s, 'regime')
  const eps = child(s, 'noise')
  const regime = new Int32Array(n)
  const y = new Float64Array(n)
  const f = new Float64Array(n)
  rows.forEach((r, i) => {
    const g = model.gate(r)
    const k = g.filter((w) => w > 0).length === 1 ? g.indexOf(1) : (categorical(pick, g) as number)
    regime[i] = k
    f[i] = model.fn(r, k)
    y[i] =
      model.task === 'regression'
        ? f[i] + model.noiseSd * normal(eps)
        : uniform(eps) < 1 / (1 + Math.exp(-f[i]))
          ? 1
          : 0
  })
  return { x: matrix(Float64Array.from(rows.flat()), n, d), y, f, regime }
}

/**
 * The dataset of a regime model: $n$ rows drawn by `draw` (sorted when 1-D), with the truth and the recipe in `meta`.
 *
 * @param s The generator's stream, whose key is recorded.
 * @param model The regime model.
 * @param n The number of rows.
 * @param base The generator's name, for the recipe.
 * @param knobs The generator's options, for the recipe.
 * @param description The dataset's one-line description.
 * @returns The dataset with its regimes.
 */
function dataset(
  s: Stream,
  model: RegimeModel,
  n: Size,
  base: string,
  knobs: Record<string, unknown>,
  description: string,
): RegimeDataset {
  const { x, y, f, regime } = draw(s, model, n, model.lower.length === 1)
  const d = model.lower.length
  const classification = model.task === 'classification'
  return {
    kind: 'dataset',
    x,
    y: classification ? labels(y) : vector(y),
    f: vector(f),
    regime: fromData(regime, [n]),
    meta: {
      name: model.name,
      description,
      task: model.task,
      featureNames: d === 1 ? ['x'] : ['x1', 'x2'],
      ...(classification ? { labelNames: ['class 0', 'class 1'] } : { targetName: 'y' }),
      key: s.key,
      truth: regimeTruth(model),
      recipe: generatorRecipe(base, s.key, knobs),
    },
  }
}

/**
 * The one-hot gate of regime $k$ among $K$.
 *
 * @param K The number of regimes.
 * @param k The chosen regime.
 * @returns $K$ weights, 1 at $k$ and 0 elsewhere.
 */
const oneHot = (K: number, k: number) => Array.from({ length: K }, (_, j) => (j === k ? 1 : 0))

// ── Piecewise linear ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `piecewiseLinear`. */
export interface PiecewiseLinearOptions {
  /** Number of points (default 200). */
  n?: Size
  /** Number of pieces $K$, at least 1 (default 3), on $K$ equal intervals of $[-3, 3]$. */
  pieces?: Size
  /** Noise standard deviation (default 0.15). */
  noise?: number
  /** The size of the jump at each breakpoint, alternating in sign (default 1; 0 joins the pieces). */
  jump?: number
}

/**
 * $K$ lines on consecutive equal intervals of $[-3, 3]$ (breakpoints at $-3 + 6j/K$): slopes of alternating sign and
 * size in $[0.5, 1.5)$ drawn from `child(s, 'lines')`, the first line starting at a $\Gauss(0, 0.5^2)$ height and each
 * later one where the last ended plus a jump of alternating sign ($+$`jump` at the first breakpoint). The gate is hard:
 * the regime of a row is its interval. Throws `DomainError` when there are no pieces.
 *
 * @param s The stream the lines and the points are drawn from.
 * @param options The number of points and pieces, the noise and the jump (`PiecewiseLinearOptions`).
 * @returns The points (`x` $n \times 1$, sorted; `y`; `f`), the regime of each and the truth.
 *
 * @example The truth jumps by `jump` at the first breakpoint
 * const d = piecewiseLinear(stream(1), { n: 300, pieces: 3, jump: 1 })
 * const r = toArray(d.regime)
 * print('x:', d.x.shape, ' rows per regime:', [0, 1, 2].map((k) => r.filter((v) => v === k).length))
 * print('lines:', d.meta.truth.model.formulas)
 * const f = toArray(d.meta.truth.mean(tensor([[-1.0001], [-0.9999]])))
 * print('jump at x = -1:', f[1] - f[0])
 */
export function piecewiseLinear(s: Stream, options: PiecewiseLinearOptions = {}): RegimeDataset {
  const { n = 200, pieces = 3, noise = 0.15, jump = 1 } = options
  checkCount(n, 'piecewiseLinear')
  if (!(pieces >= 1)) throw new DomainError('piecewiseLinear', 'piecewiseLinear: needs at least one piece')
  const K = pieces
  const breaks = Array.from({ length: K + 1 }, (_, j) => -3 + (6 * j) / K)
  const ls = child(s, 'lines')
  const slopes: number[] = []
  const intercepts: number[] = []
  let start = 0.5 * normal(ls)
  for (let k = 0; k < K; k++) {
    const b = (k % 2 === 0 ? 1 : -1) * (0.5 + uniform(ls))
    if (k > 0) start = slopes[k - 1] * breaks[k] + intercepts[k - 1] + (k % 2 === 1 ? jump : -jump)
    slopes.push(b)
    intercepts.push(start - b * breaks[k])
  }
  const piece = (x: number) => Math.min(K - 1, Math.max(0, Math.floor(((x + 3) / 6) * K)))
  const model: RegimeModel = {
    name: 'piecewise linear',
    task: 'regression',
    regimes: K,
    gate: (r: Row) => oneHot(K, piece(r[0])),
    fn: (r: Row, k: number) => intercepts[k] + slopes[k] * r[0],
    noiseSd: noise,
    lower: [-3],
    upper: [3],
    formulas: slopes.map((b, k) => `y = ${line(intercepts[k], b)}`),
  }
  return dataset(
    s,
    model,
    n,
    'piecewiseLinear',
    { n, pieces, noise, jump },
    `${n} noisy points (sd ${noise}) from ${K} lines on [−3, 3] with breakpoints at ${breaks
      .slice(1, -1)
      .map(fmt)
      .join(', ')}.`,
  )
}

// ── Quadrants ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `quadrantPlanes`. */
export interface QuadrantPlanesOptions {
  /** Number of points (default 300). */
  n?: Size
  /** `regression`: a plane per quadrant; `classification`: a linear boundary per quadrant (default regression). */
  task?: 'regression' | 'classification'
  /** Noise standard deviation (regression; default 0.15). */
  noise?: number
  /** Classification: the slope of the logistic link across each boundary (default 6; large is nearly noise-free). */
  sharpness?: number
}

/**
 * The quadrant of a point, counter-clockwise from the positive one: 0 $(+, +)$, 1 $(-, +)$, 2 $(-, -)$, 3 $(+, -)$. A
 * coordinate of 0 counts as positive.
 *
 * @param x1 The first coordinate.
 * @param x2 The second coordinate.
 * @returns The quadrant, 0 to 3.
 *
 * @example The four quadrants, and the axes
 * print('quadrants:', [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => quadrantOf(a, b)))
 * print('origin:', quadrantOf(0, 0), ' negative x1-axis:', quadrantOf(-1, 0))
 */
export const quadrantOf = (x1: number, x2: number): number => (x2 >= 0 ? (x1 >= 0 ? 0 : 1) : x1 < 0 ? 2 : 3)

/**
 * Inputs uniform on $[-2, 2]^2$, four regimes by quadrant (`quadrantOf`). Regression:
 * $y = \wvec_k^\top \xvec + b_k + \varepsilon$ with $\wvec_k \sim \Gauss(\zeros, \Imat)$ and $b_k \sim \Gauss(0, 1)$
 * per quadrant. Classification: $P(y = 1 \mid \xvec) = \sigma(\kappa \wvec_k^\top (\xvec - \cvec_k))$ with a unit
 * direction $\wvec_k$ at a uniform angle per quadrant and $\cvec_k$ the quadrant's centre $(\pm 1, \pm 1)$, so each
 * quadrant holds a differently oriented boundary through its middle. The planes come from `child(s, 'planes')`.
 *
 * @param s The stream the planes and the points are drawn from.
 * @param options The number of points, the task, the regression noise and the classification sharpness $\kappa$
 *   (`QuadrantPlanesOptions`).
 * @returns The points (`x` $n \times 2$; `y`, targets or int32 labels; `f`, the plane's value or the log-odds), the
 *   regime of each and the truth.
 *
 * @example The regime is the quadrant, and the noise is the rest
 * const d = quadrantPlanes(stream(1), { n: 400, noise: 0.15 })
 * const [x, y, f, r] = [toArray(d.x), toArray(d.y), toArray(d.f), toArray(d.regime)]
 * print('x:', d.x.shape, ' planes:', d.meta.truth.model.formulas)
 * print('regime is the quadrant:', x.every(([a, b], i) => quadrantOf(a, b) === r[i]))
 * print('RMS of y - f (0.15):', Math.sqrt(y.reduce((a, v, i) => a + (v - f[i]) ** 2, 0) / y.length))
 */
export function quadrantPlanes(s: Stream, options: QuadrantPlanesOptions = {}): RegimeDataset {
  const { n = 300, task = 'regression', noise = 0.15, sharpness = 6 } = options
  checkCount(n, 'quadrantPlanes')
  const ps = child(s, 'planes')
  const centres = [
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
  ]
  const planes = centres.map((c) => {
    if (task === 'regression') return { w: [normal(ps), normal(ps)], b: normal(ps) }
    const angle = 2 * Math.PI * uniform(ps)
    const w = [Math.cos(angle), Math.sin(angle)]
    return { w, b: -(w[0] * c[0] + w[1] * c[1]) }
  })
  const scale = task === 'regression' ? 1 : sharpness
  const model: RegimeModel = {
    name: task === 'regression' ? 'quadrant planes' : 'quadrant boundaries',
    task,
    regimes: 4,
    gate: (r: Row) => oneHot(4, quadrantOf(r[0], r[1])),
    fn: (r: Row, k: number) => scale * (planes[k].w[0] * r[0] + planes[k].w[1] * r[1] + planes[k].b),
    noiseSd: task === 'regression' ? noise : 0,
    lower: [-2, -2],
    upper: [2, 2],
    formulas: planes.map((p) =>
      task === 'regression'
        ? `y = ${fmt(p.b)} ${p.w[0] < 0 ? '−' : '+'} ${fmt(Math.abs(p.w[0]))}x₁ ${p.w[1] < 0 ? '−' : '+'} ${fmt(Math.abs(p.w[1]))}x₂`
        : `logit P(y = 1) = ${fmt(sharpness)}(${fmt(p.w[0])}x₁ + ${fmt(p.w[1])}x₂ + ${fmt(p.b)})`,
    ),
  }
  return dataset(
    s,
    model,
    n,
    'quadrantPlanes',
    { n, task, noise, sharpness },
    task === 'regression'
      ? `${n} points on [−2, 2]² with a different plane in each quadrant and noise sd ${noise}.`
      : `${n} points on [−2, 2]² labelled by a differently oriented linear boundary in each quadrant (logistic slope ${sharpness}).`,
  )
}

// ── Interleaved functions ────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `interleavedFunctions`. */
export interface InterleavedOptions {
  /** Number of points (default 240). */
  n?: Size
  /** Number of bands of $[-3, 3]$, at least 1 (default 4); the two functions take turns band by band. */
  bands?: Size
  /** Noise standard deviation (default 0.1). */
  noise?: number
}

/**
 * Two lines, $y = 0.8x + 0.6$ and $y = -0.5x - 0.6$, taking turns over `bands` equal bands of $[-3, 3]$: regime $k$
 * on the bands $j$ with $j \bmod 2 = k$. A gate linear in $x$ splits the line once, so fitting the bands needs as many
 * experts as bands, or a gate that can bend (an MLP router or a hierarchy). The number of bands is not checked.
 *
 * @param s The stream the points are drawn from.
 * @param options The number of points and bands, and the noise (`InterleavedOptions`).
 * @returns The points (`x` $n \times 1$, sorted; `y`; `f`), the regime of each and the truth.
 *
 * @example The regimes alternate band by band
 * const d = interleavedFunctions(stream(1), { n: 200, bands: 4 })
 * print('x:', d.x.shape, ' lines:', d.meta.truth.model.formulas)
 * print('regime at x = -2.5, -1, 0.5, 2:', d.meta.truth.regime(tensor([[-2.5], [-1], [0.5], [2]])))
 */
export function interleavedFunctions(s: Stream, options: InterleavedOptions = {}): RegimeDataset {
  const { n = 240, bands = 4, noise = 0.1 } = options
  checkCount(n, 'interleavedFunctions')
  const lines = [
    { a: 0.6, b: 0.8 },
    { a: -0.6, b: -0.5 },
  ]
  const band = (x: number) => Math.min(bands - 1, Math.max(0, Math.floor(((x + 3) / 6) * bands)))
  const model: RegimeModel = {
    name: 'interleaved functions',
    task: 'regression',
    regimes: 2,
    gate: (r: Row) => oneHot(2, band(r[0]) % 2),
    fn: (r: Row, k: number) => lines[k].a + lines[k].b * r[0],
    noiseSd: noise,
    lower: [-3],
    upper: [3],
    formulas: lines.map((l) => `y = ${line(l.a, l.b)}`),
  }
  return dataset(
    s,
    model,
    n,
    'interleavedFunctions',
    { n, bands, noise },
    `${n} points from two lines that take turns over ${bands} bands of [−3, 3], noise sd ${noise}.`,
  )
}

// ── Mixture of regressions ───────────────────────────────────────────────────────────────────────────────────────────

/** Options of `regressionMixture`. */
export interface RegressionMixtureOptions {
  /** Number of points (default 300). */
  n?: Size
  /** Number of regimes $K$ (default 2). */
  regimes?: Size
  /** Noise standard deviation (default 0.2). */
  noise?: number
  /**
   * How far the regimes overlap, $\tau > 0$ (default 1): the gate is $\operatorname{softmax}_k(c_k x / \tau)$ with
   * $c_k$ evenly spaced on $[-3, 3]$; near 0 it is a hard partition of $x$, large values mix the regimes everywhere.
   */
  overlap?: number
}

/**
 * $K$ lines (intercepts $\Gauss(0, 1)$ and slopes of alternating sign and size in $[0.5, 1.5)$, drawn from
 * `child(s, 'lines')`) chosen by a soft gate $\operatorname{softmax}_k(c_k x / \tau)$, $c_k$ evenly spaced on
 * $[-3, 3]$ ($c_0 = 0$ when $K = 1$) and $\tau$ = `overlap`: where the gate is mixed, $y$ given $x$ has a mode on each
 * line. Throws `DomainError` when `overlap` is not positive.
 *
 * @param s The stream the lines and the points are drawn from.
 * @param options The number of points and regimes, the noise and the overlap (`RegressionMixtureOptions`).
 * @returns The points (`x` $n \times 1$, sorted; `y`; `f`), the regime each was drawn from, and the truth.
 *
 * @example The gate is mixed in the middle and decided at the ends
 * const d = regressionMixture(stream(1), { n: 300, regimes: 2, overlap: 1 })
 * const r = toArray(d.regime)
 * print('x:', d.x.shape, ' lines:', d.meta.truth.model.formulas)
 * print('gate at x = -2, 0, 2:', d.meta.truth.gate(tensor([[-2], [0], [2]])))
 * print('rows per regime:', [0, 1].map((k) => r.filter((v) => v === k).length))
 */
export function regressionMixture(s: Stream, options: RegressionMixtureOptions = {}): RegimeDataset {
  const { n = 300, regimes = 2, noise = 0.2, overlap = 1 } = options
  checkCount(n, 'regressionMixture')
  if (!(overlap > 0)) throw new DomainError('regressionMixture', 'regressionMixture: overlap must be positive')
  const K = regimes
  const ls = child(s, 'lines')
  const lines = Array.from({ length: K }, (_, k) => ({
    a: normal(ls),
    b: (k % 2 === 0 ? 1 : -1) * (0.5 + uniform(ls)),
  }))
  const c = Array.from({ length: K }, (_, k) => (K === 1 ? 0 : -3 + (6 * k) / (K - 1)))
  const model: RegimeModel = {
    name: 'mixture of regressions',
    task: 'regression',
    regimes: K,
    gate: (r: Row) => {
      const z = c.map((ck) => (ck * r[0]) / overlap)
      const m = Math.max(...z)
      const e = z.map((v) => Math.exp(v - m))
      const total = e.reduce((a, b) => a + b, 0)
      return e.map((v) => v / total)
    },
    fn: (r: Row, k: number) => lines[k].a + lines[k].b * r[0],
    noiseSd: noise,
    lower: [-3],
    upper: [3],
    formulas: lines.map((l) => `y = ${line(l.a, l.b)}`),
  }
  return dataset(
    s,
    model,
    n,
    'regressionMixture',
    { n, regimes, noise, overlap },
    `${n} points from ${K} lines chosen by a soft gate over x (overlap ${overlap}), noise sd ${noise}.`,
  )
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const datasetEntry = definer<DatasetInfo>('dataset', 'data/synthetic')
const NOTES = ['mixture-of-experts']

datasetEntry(
  {
    key: 'piecewiseLinear',
    name: 'Piecewise linear',
    summary: 'K lines on consecutive intervals with jumps at known breakpoints; the regime of each row is known.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 200 }),
      pieces: int(1, 8, { default: 3 }),
      noise: real(0, 2, { default: 0.15 }),
      jump: real(-3, 3, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: NOTES,
  },
  piecewiseLinear,
)
datasetEntry(
  {
    key: 'quadrantPlanes',
    name: 'Quadrant planes',
    summary: 'A different plane (regression) or linear boundary (classification) in each quadrant of the plane.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 300 }),
      task: oneOf(['regression', 'classification']),
      noise: real(0, 2, { default: 0.15 }),
      sharpness: real(0.1, 50, { default: 6 }),
    }),
    truth: true,
    random: true,
    notes: NOTES,
  },
  quadrantPlanes,
)
datasetEntry(
  {
    key: 'interleavedFunctions',
    name: 'Interleaved functions',
    summary: 'Two lines that take turns over alternating bands of x.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 240 }),
      bands: int(1, 12, { default: 4 }),
      noise: real(0, 2, { default: 0.1 }),
    }),
    truth: true,
    random: true,
    notes: NOTES,
  },
  interleavedFunctions,
)
datasetEntry(
  {
    key: 'regressionMixture',
    name: 'Mixture of regressions',
    summary: 'K lines chosen by a soft gate over x, so the regimes overlap and y given x is multimodal.',
    task: 'regression',
    output: 'dataset',
    knobs: space({
      n: int(2, 5000, { default: 300 }),
      regimes: int(1, 6, { default: 2 }),
      noise: real(0, 2, { default: 0.2 }),
      overlap: real(0.01, 10, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: NOTES,
  },
  regressionMixture,
)
