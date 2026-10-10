/**
 * Data for matrix factorisations and linear latent-variable models:
 *
 * - `cocktailParty`: three independent non-Gaussian sources (a sine, a square wave and a sawtooth) mixed linearly into
 *   three microphones, the classic ICA demonstration (Hyvärinen and Oja, 2000, §1).
 * - `strokeGlyphs`: binary glyphs on a small grid, each the union of a few strokes (bars across, down and the two
 *   diagonals); the strokes are the parts that NMF can recover (Lee and Seung, 1999).
 * - `latentFactorModel`, `latentFactors`: $\xvec = \Wmat\zvec + \muvec + \epsilonvec$ with
 *   $\zvec \sim \Gauss(\zeros, \Imat_q)$ and noise variances that differ by feature, the setting where factor analysis
 *   and probabilistic PCA differ.
 */

import type { DatasetInfo, FunctionInfo } from 'aifn-compute/foundation/contracts'
import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkCount, matrix, type Dataset } from '../types'

/** The sources, their mixture and the mixing matrix of `cocktailParty`. */
export type CocktailParty = {
  /** Sample times, $n$ values evenly spaced from 0 (step $8/n$). */
  t: Tensor
  /**
   * The three sources as columns, $n \times 3$: $\sin 2t$, the square wave $\sgn \sin 3t$ and the sawtooth
   * $(1.7t \bmod 2) - 1$, each with values in $[-1, 1]$ and mean near zero.
   */
  sources: Tensor
  /** The microphones as columns, $n \times 3$: $\Smat\Amat^\top$ plus noise, $\Smat$ the sources. */
  mixed: Tensor
  /** The mixing matrix $\Amat$, $3 \times 3$ (row $i$: microphone $i$'s weights on the sources). */
  mixing: Tensor
}

/**
 * Three sources over $t \in [0, 8)$ mixed by $\Amat$ into three microphones ($\Amat$'s entries uniform on
 * $[0.2, 1.2]$, drawn from the stream; `noise` adds Gaussian sensor noise of that standard deviation). The sources are
 * deterministic; only the mixing and the noise are random. Throws `DomainError` when $n$ is not a non-negative integer.
 *
 * @param s The stream the mixing matrix (child `'mixing'`) and the sensor noise (child `'noise'`) are drawn from.
 * @param options `n` (default 400), the number of samples; `noise` (default 0), the standard deviation of the sensor
 *   noise.
 * @returns The times, the sources, the microphone signals and the mixing matrix.
 *
 * @example Without noise the microphones are exactly the mixed sources
 * const c = cocktailParty(stream(1), { n: 400 })
 * print('t:', c.t.shape, ' sources:', c.sources.shape, ' mixed:', c.mixed.shape)
 * print('mixing A =', c.mixing)
 * print('first sources:', toArray(c.sources).slice(0, 2))
 * print('first mixed:', toArray(c.mixed).slice(0, 2))
 * print('S A^T, first rows:', toArray(matmul(c.sources, transpose(c.mixing))).slice(0, 2))
 */
export function cocktailParty(s: Stream, options: { n?: number; noise?: number } = {}): CocktailParty {
  const { n = 400, noise = 0 } = options
  checkCount(n, 'cocktailParty')
  const S = new Float64Array(n * 3)
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const ti = (8 * i) / n
    t[i] = ti
    S[i * 3] = Math.sin(2 * ti)
    S[i * 3 + 1] = Math.sign(Math.sin(3 * ti)) || 1
    S[i * 3 + 2] = ((1.7 * ti) % 2) - 1
  }
  const m = child(s, 'mixing')
  const A = Float64Array.from({ length: 9 }, () => 0.2 + (uniform(m) as number))
  const r = child(s, 'noise')
  const X = new Float64Array(n * 3)
  for (let i = 0; i < n; i++)
    for (let m = 0; m < 3; m++) {
      let v = 0
      for (let k = 0; k < 3; k++) v += A[m * 3 + k] * S[i * 3 + k]
      X[i * 3 + m] = v + (noise > 0 ? noise * (normal(r) as number) : 0)
    }
  return { t: fromData(t, [n]), sources: matrix(S, n, 3), mixed: matrix(X, n, 3), mixing: matrix(A, 3, 3) }
}

/**
 * The strokes of `strokeGlyphs` on a grid of side $m$, as the rows of an $8 \times m^2$ matrix of 0s and 1s (one
 * row-major grid per stroke): the middle row, the middle column, the top and bottom rows, the left and right columns,
 * and the two diagonals. Deterministic.
 *
 * @param options The grid.
 * @param options.size The side $m$ of the grid in pixels; for an even side the middle row and column are the ones
 *   just below and right of centre.
 * @returns The strokes, $8 \times m^2$.
 *
 * @example The eight strokes of a 5 by 5 grid
 * const strokes = toArray(glyphStrokes({ size: 5 }))
 * print('strokes:', strokes.length, ' pixels each:', strokes[0].length)
 * print('pixels on per stroke:', strokes.map((r) => r.reduce((a, v) => a + v, 0)))
 * // The last stroke, the anti-diagonal, drawn as a grid.
 * print([0, 1, 2, 3, 4].map((r) => strokes[7].slice(5 * r, 5 * r + 5).map((v) => (v ? '#' : '.')).join('')).join('\n'))
 */
export function glyphStrokes({ size = 7 }: { size?: number } = {}): Tensor {
  const mid = Math.floor(size / 2)
  const on = (f: (r: number, c: number) => boolean) =>
    Array.from({ length: size * size }, (_, k) => (f(Math.floor(k / size), k % size) ? 1 : 0))
  const strokes = [
    on((r) => r === mid),
    on((_r, c) => c === mid),
    on((r) => r === 0),
    on((r) => r === size - 1),
    on((_r, c) => c === 0),
    on((_r, c) => c === size - 1),
    on((r, c) => r === c),
    on((r, c) => r + c === size - 1),
  ]
  return matrix(Float64Array.from(strokes.flat()), strokes.length, size * size)
}

/**
 * Glyphs made of strokes: each of $n$ glyphs switches each stroke of `glyphStrokes` on with probability `p` (when none
 * is, one is chosen uniformly), takes the union (pixels capped at 1) and adds Gaussian noise of standard deviation
 * `noise`, clipped at 0 so the data stay non-negative. `x` is $n \times m^2$ for a grid of side $m$; `y` counts the
 * strokes in each glyph. Throws `DomainError` when $n$ is not a non-negative integer.
 *
 * @param s The stream the glyphs are drawn from (glyph $i$ from `child(s, 'glyph', i)`).
 * @param options `n` (default 200), the number of glyphs; `size` (default 7), the side $m$ of the grid; `p` (default
 *   0.3), the probability of each stroke; `noise` (default 0.05), the standard deviation of the pixel noise.
 * @returns An image dataset: `x` the glyphs, one row of $m^2$ pixels each, and `y` the number of strokes in each.
 *
 * @example About 8p strokes per glyph
 * const d = strokeGlyphs(stream(1), { n: 1000, p: 0.3 })
 * const y = toArray(d.y)
 * print('x:', d.x.shape, ' first stroke counts:', y.slice(0, 10))
 * print('first pixels of glyph 0:', toArray(d.x)[0].slice(0, 7))
 * // 8 x 0.3 = 2.4, plus the glyphs that would have had none.
 * print('mean strokes:', y.reduce((a, v) => a + v, 0) / y.length)
 */
export function strokeGlyphs(
  s: Stream,
  options: { n?: number; size?: number; p?: number; noise?: number } = {},
): Dataset {
  const { n = 200, size = 7, p = 0.3, noise = 0.05 } = options
  checkCount(n, 'strokeGlyphs')
  const parts = glyphStrokes({ size }).data as Float64Array
  const k = parts.length / (size * size)
  const d = size * size
  const x = new Float64Array(n * d)
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const r = child(s, 'glyph', i)
    const chosen: number[] = []
    for (let j = 0; j < k; j++) if ((uniform(r) as number) < p) chosen.push(j)
    if (chosen.length === 0) chosen.push(Math.floor((uniform(r) as number) * k))
    y[i] = chosen.length
    for (let q = 0; q < d; q++) {
      let v = 0
      for (const j of chosen) v = Math.max(v, parts[j * d + q])
      x[i * d + q] = Math.max(0, v + (noise > 0 ? noise * (normal(r) as number) : 0))
    }
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: fromData(y, [n]),
    meta: {
      name: 'stroke glyphs',
      description: `${n} ${size} × ${size} glyphs, each the union of strokes switched on with probability ${p} (noise sd ${noise}, clipped at 0); y counts the strokes.`,
      task: 'images',
      featureNames: Array.from({ length: d }, (_, q) => `pixel ${Math.floor(q / size)},${q % size}`),
      key: s.key,
    },
  }
}

/** The true parameters of `latentFactors`. */
export type LatentFactorModel = {
  /**
   * Loadings $\Wmat$, $d \times q$: the features are split into $q$ consecutive blocks of $b = \lceil d/q \rceil$; the
   * $i$th feature of block $j$ (counting from 0) loads $1 + 0.25i$ on factor $j$ and 0.3 on factor $j + 1$
   * (cyclically).
   */
  loadings: Tensor
  /** Noise variances $\psi$, $d$ values from 0.05 to `spread`, rising geometrically with the feature index. */
  noise: Tensor
  /** The mean $\muvec$, $d$ values (zero). */
  mean: Tensor
}

/**
 * The deterministic model of `latentFactors` for $d$ features and $q$ factors (see `LatentFactorModel`).
 *
 * @param options The sizes and the noise.
 * @param options.d The number of features $d$.
 * @param options.latent The number of factors $q$.
 * @param options.spread The largest noise variance, that of the last feature; the first's is 0.05.
 * @returns The loadings, the noise variances and the (zero) mean.
 *
 * @example Block loadings and noise rising across the features
 * const m = latentFactorModel({ d: 4, latent: 2, spread: 2 })
 * print('W =', m.loadings)
 * print('psi =', m.noise)
 */
export function latentFactorModel({
  d = 8,
  latent = 2,
  spread = 2,
}: { d?: number; latent?: number; spread?: number } = {}): LatentFactorModel {
  const W = new Float64Array(d * latent)
  const block = Math.ceil(d / latent)
  for (let i = 0; i < d; i++) {
    const j = Math.min(Math.floor(i / block), latent - 1)
    W[i * latent + j] = 1 + 0.25 * (i % block)
    if (latent > 1) W[i * latent + ((j + 1) % latent)] = 0.3
  }
  const psi = Float64Array.from({ length: d }, (_, i) => 0.05 * (spread / 0.05) ** (d > 1 ? i / (d - 1) : 0))
  return { loadings: matrix(W, d, latent), noise: fromData(psi, [d]), mean: fromData(new Float64Array(d), [d]) }
}

/**
 * $n$ draws of $\xvec = \Wmat\zvec + \epsilonvec$, $\zvec \sim \Gauss(\zeros, \Imat_q)$ and
 * $\epsilonvec \sim \Gauss(\zeros, \diag(\psi))$, from `latentFactorModel` (`x` is $n \times d$; the latent $\zvec$
 * is not returned). Throws `DomainError` when $n$ is not a non-negative integer.
 *
 * @param s The stream the draws are taken from (child `'draws'`).
 * @param options `n` (default 300), the number of draws; `d` (default 8), `latent` (default 2) and `spread` (default
 *   2), as for `latentFactorModel`.
 * @returns A dataset of the draws `x`, with no targets.
 *
 * @example The sample variances match the model's
 * const d = latentFactors(stream(1), { n: 4000, d: 4, latent: 2 })
 * const x = toArray(d.x)
 * print('x:', d.x.shape, ' first row:', x[0])
 * const m = latentFactorModel({ d: 4, latent: 2 })
 * const [W, psi] = [toArray(m.loadings), toArray(m.noise)]
 * print('model variances:', W.map((r, f) => r.reduce((a, w) => a + w * w, 0) + psi[f]))
 * print('sample variances:', [0, 1, 2, 3].map((f) => x.reduce((a, r) => a + r[f] * r[f], 0) / x.length))
 */
export function latentFactors(
  s: Stream,
  options: { n?: number; d?: number; latent?: number; spread?: number } = {},
): Dataset {
  const { n = 300, d = 8, latent = 2, spread = 2 } = options
  checkCount(n, 'latentFactors')
  const { loadings, noise } = latentFactorModel({ d, latent, spread })
  const W = loadings.data as Float64Array
  const psi = noise.data as Float64Array
  const x = new Float64Array(n * d)
  const r = child(s, 'draws')
  for (let i = 0; i < n; i++) {
    const z = Array.from({ length: latent }, () => normal(r) as number)
    for (let f = 0; f < d; f++) {
      let v = 0
      for (let j = 0; j < latent; j++) v += W[f * latent + j] * z[j]
      x[i * d + f] = v + Math.sqrt(psi[f]) * (normal(r) as number)
    }
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    meta: {
      name: 'latent factors',
      description: `${n} draws of x = Wz + ε with ${latent} factors in ${d} features; noise variances from 0.05 to ${spread} across the features.`,
      task: 'manifold',
      featureNames: Array.from({ length: d }, (_, f) => `x${f + 1}`),
      key: s.key,
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'cocktailParty',
    name: 'Cocktail party',
    summary: 'A sine, a square wave and a sawtooth mixed linearly into three microphones.',
    task: 'sequence',
    output: 'series',
    knobs: space({ n: int(10, 5000, { default: 400 }), noise: real(0, 1, { default: 0 }) }),
    truth: false,
    random: true,
    notes: ['independent-component-analysis'],
  },
  cocktailParty,
)
dataset(
  {
    key: 'glyphStrokes',
    name: 'Glyph strokes',
    summary: 'The eight strokes (bars and diagonals) that stroke glyphs are made of.',
    task: 'images',
    output: 'patterns',
    knobs: space({ size: int(3, 32, { default: 7 }) }),
    truth: false,
    random: false,
    notes: ['non-negative-matrix-factorisation'],
  },
  glyphStrokes,
)
dataset(
  {
    key: 'strokeGlyphs',
    name: 'Stroke glyphs',
    summary: 'Glyphs that are unions of random strokes, with non-negative noise: parts for NMF to find.',
    task: 'images',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 200 }),
      size: int(3, 32, { default: 7 }),
      p: real(0.05, 1, { default: 0.3 }),
      noise: real(0, 1, { default: 0.05 }),
    }),
    truth: false,
    random: true,
    notes: ['non-negative-matrix-factorisation'],
  },
  strokeGlyphs,
)
dataset(
  {
    key: 'latentFactors',
    name: 'Latent factors',
    summary: 'Draws of x = Wz + ε with block loadings and noise variances that differ by feature.',
    task: 'manifold',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      d: int(2, 50, { default: 8 }),
      latent: int(1, 10, { default: 2 }),
      spread: real(0.05, 10, { default: 2 }),
    }),
    truth: false,
    random: true,
    notes: ['factor-analysis', 'probabilistic-principal-component-analysis'],
  },
  latentFactors,
)
definer<FunctionInfo>('function', 'data/synthetic')(
  {
    key: 'latentFactorModel',
    name: 'Latent factor model',
    summary: 'The loadings and per-feature noise variances that latentFactors draws from.',
    role: 'construction',
    notes: ['factor-analysis'],
  },
  latentFactorModel,
)
