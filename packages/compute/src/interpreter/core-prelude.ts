/**
 * The core prelude, in namespaces that mirror aifn's module tree, and the root random stream of a run.
 *
 * `math` is generated from aifn's primitive registry (every elementwise primitive of the tensor, special-function and
 * neural-network modules), so a new primitive appears without a new line; `array`, `random`, `stats`, `linalg` and
 * `signal` wrap their modules by hand. Everything takes numbers, plain arrays or tensors and returns numbers, plain
 * (nested) arrays or plain objects of them. Each random entry draws from `ctx.draw()`, a fresh child of the run's root
 * stream, so the draws depend on the seed and the call order only.
 */
import { convolve } from 'aifn-compute/foundation/convolution'
import { rfft, rfftfreq } from 'aifn-compute/foundation/fourier'
import * as draws from 'aifn-compute/foundation/random'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  arange,
  argmax,
  argmin,
  clip,
  complexAbs,
  concat,
  cumsum,
  dot,
  eye,
  full,
  linspace,
  map,
  matmul,
  max,
  mean,
  min,
  norm,
  outer,
  registry,
  reshape,
  shapeOf,
  stack,
  std,
  sum,
  toFlat,
  transpose,
  variance,
  zeros,
  type Primitive,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { cholesky, det, eigh, inverse, lstsq, qr, solve, svd } from 'aifn-compute/numerics/linalg'
// Imported for their primitives too: loading a module registers them, and `math` lists the registry.
import * as special from 'aifn-compute/numerics/special'
import * as functional from 'aifn-compute/nn/functional'
import { histogram, median, quantile } from 'aifn-compute/probability/stats'
import {
  makePrelude,
  params,
  withPrelude,
  type Context,
  type Namespace,
  type Prelude,
  type PreludeEntry,
} from './prelude'
import { describe, toData, toShape, toTensor, toValue } from './values'

/** The modules whose primitives are loaded for the generated maths. */
export const PRIMITIVE_SOURCES = { special, functional }

/** Modules whose elementwise primitives become `math` functions. */
const MATH_MODULES = new Set(['foundation/tensor', 'numerics/special', 'nn/functional'])

/** Elementwise primitives left out: complex-valued ones and internal pieces of other algorithms. */
const MATH_EXCLUDED = new Set([
  'complex',
  'expj',
  'conj',
  'realPart',
  'imagPart',
  'complexAbs',
  'angle',
  'truncatedNormalV',
  'truncatedNormalW',
  'truncatedNormalVDraw',
  'truncatedNormalWDraw',
])

/** Docs for primitives registered without a summary. */
const MATH_DOCS: Record<string, string> = {
  erf: 'The error function.',
  erfc: 'The complementary error function 1 − erf(x).',
  erfcx: 'The scaled complementary error function exp(x²)·erfc(x).',
  logErfc: 'log erfc(x), accurate in the tails.',
  erfinv: 'The inverse error function.',
  erfcinv: 'The inverse complementary error function.',
  normalPdf: 'The standard normal density.',
  normalLogPdf: 'The standard normal log density.',
  normalCdf: 'The standard normal cdf Φ(x).',
  normalLogCdf: 'log Φ(x), accurate in the lower tail.',
  normalQuantile: 'The standard normal quantile Φ⁻¹(p).',
  logGamma: 'log Γ(x).',
  gamma: 'The gamma function Γ(x).',
  digamma: 'The digamma function ψ(x) = d log Γ(x) / dx.',
  trigamma: 'The trigamma function ψ′(x).',
  polygamma: 'The polygamma function ψ⁽ⁿ⁾(x), as polygamma(n, x).',
  logBeta: 'log B(a, b).',
  logFactorial: 'log n!.',
  logChoose: 'log of the binomial coefficient (n choose k).',
  softplus: 'log(1 + eˣ), without overflow.',
  sigmoid: 'The logistic sigmoid 1 / (1 + e⁻ˣ).',
  logSigmoid: 'log σ(x), without overflow.',
  logit: 'The logit log(p / (1 − p)), the inverse of the sigmoid.',
  log1mexp: 'log(1 − eˣ) for x < 0, accurate near 0.',
  logExpm1: 'log(eˣ − 1), without overflow.',
  log1pmx: 'log(1 + x) − x, accurate for small x.',
  logAddExp: 'log(eᵃ + eᵇ), without overflow.',
  logDiffExp: 'log(eᵃ − eᵇ) for a > b, without overflow.',
  xlogy: 'x·log(y), 0 when x = 0.',
  xlog1py: 'x·log(1 + y), 0 when x = 0.',
  besselI0: 'The modified Bessel function I₀(x).',
  besselI1: 'The modified Bessel function I₁(x).',
  besselRatio: 'The ratio I₁(x) / I₀(x).',
  logBesselI0: 'log I₀(x), without overflow.',
  binaryEntropy: 'The binary entropy −p log p − (1 − p) log(1 − p), in nats.',
  regularisedGammaP: 'The regularised lower incomplete gamma P(a, x).',
  regularisedGammaQ: 'The regularised upper incomplete gamma Q(a, x).',
  regularisedBeta: 'The regularised incomplete beta Iₓ(a, b), as regularisedBeta(x, a, b).',
  studentTCdf: 'The Student t cdf, as studentTCdf(t, ν).',
  chiSquareCdf: 'The chi-square cdf, as chiSquareCdf(x, k).',
  chiSquareSf: 'The chi-square survival function 1 − cdf, as chiSquareSf(x, k).',
}

const PARAM_NAMES = ['x', 'x, y', 'a, b, c']
const SELF = 'aifn-compute/interpreter'
const TENSOR = 'aifn-compute/foundation/tensor'
const RANDOM = 'aifn-compute/foundation/random'
const STATS = 'aifn-compute/probability/stats'
const LINALG = 'aifn-compute/numerics/linalg'

/** The core namespaces. */
export const CORE_NAMESPACES: readonly Namespace[] = [
  {
    name: 'math',
    doc: 'Elementwise functions over numbers and arrays, from the primitive registry.',
    source: 'aifn-compute/foundation/tensor, aifn-compute/numerics/special, aifn-compute/nn/functional',
  },
  { name: 'array', doc: 'Make, join, reshape and reduce arrays.', source: TENSOR },
  { name: 'random', doc: 'Seeded random draws: the same program and seed give the same data.', source: RANDOM },
  { name: 'stats', doc: 'Summaries of data: means, spreads, quantiles and histograms.', source: STATS },
  { name: 'linalg', doc: 'Matrices: products, solves, inverses and factorisations.', source: LINALG },
  {
    name: 'signal',
    doc: 'Spectra and convolution of sampled signals.',
    source: 'aifn-compute/foundation/fourier, aifn-compute/foundation/convolution',
  },
]

/**
 * An entry's implementation from an aifn function: the program's arguments as aifn values, through `f`, and back to
 * program data.
 *
 * @param name The entry's qualified name, used in error messages (`'math.sin: argument 1'`).
 * @param f The aifn function, taking the arguments as numbers or tensors.
 * @returns The implementation, which ignores the context.
 */
const lift =
  (name: string, f: (...v: Value[]) => Value) =>
  (_ctx: Context, ...args: unknown[]) =>
    toData(f(...args.map((a, i) => toValue(a, `${name}: argument ${i + 1}`))))

/**
 * One `math` entry per registered elementwise primitive of the maths modules (less `MATH_EXCLUDED`), its parameters
 * named by arity and its doc the primitive's summary, else `MATH_DOCS`, else a generic line.
 *
 * @returns The entries, in the registry's order.
 */
function mathFromRegistry(): PreludeEntry[] {
  return registry
    .list()
    .filter((p) => p.kind === 'elementwise' && MATH_MODULES.has(p.module) && !MATH_EXCLUDED.has(p.name))
    .map((p: Primitive) => ({
      namespace: 'math',
      name: p.name,
      params: params(p.name === 'where' ? 'condition, a, b' : PARAM_NAMES[(p.arity as number) - 1]),
      doc: p.doc.summary ?? MATH_DOCS[p.name] ?? `${p.name}, elementwise.`,
      source: `aifn-compute/${p.module}`,
      returns: 'number or number[]',
      impl: lift(`math.${p.name}`, (...v) => p.apply(v, undefined)),
    }))
}

/**
 * A `math` entry from a plain scalar function mapped elementwise (what has no primitive: `tan`, `floor`, …).
 *
 * @param name The entry's name in `math`.
 * @param doc Its one-sentence doc.
 * @param f The function of one number.
 * @returns The entry, of one parameter `x`.
 */
function scalar(name: string, doc: string, f: (x: number) => number): PreludeEntry {
  return {
    namespace: 'math',
    name,
    params: params('x'),
    doc,
    source: TENSOR,
    returns: 'number or number[]',
    impl: lift(`math.${name}`, (x) => map(x, f)),
  }
}

/**
 * A maker of entries of one namespace: the returned function takes the entry's name, its parameters as a `params`
 * spec, its doc, what it returns in words, and its implementation.
 *
 * @param namespace The namespace, or null for top-level entries.
 * @param source The aifn module the entries wrap.
 * @returns The maker.
 */
const def =
  (namespace: string | null, source: string) =>
  (
    name: string,
    spec: string,
    doc: string,
    returns: string,
    impl: (ctx: Context, ...args: never[]) => unknown,
  ): PreludeEntry => ({
    namespace,
    name,
    params: params(spec),
    doc,
    source,
    returns,
    impl: impl as PreludeEntry['impl'],
  })

/**
 * A program's axis argument: undefined or null means all elements.
 *
 * @param axis The argument as the program gave it.
 * @returns The axis, or null for a reduction over every element.
 */
const axisOf = (axis: unknown) => (axis === undefined || axis === null ? null : (axis as number))

/**
 * An entry's implementation from a reduction: over all elements (a number) or along an axis (an array).
 *
 * @param f The reduction, given the input as a tensor and the axis (null for all elements).
 * @returns The implementation, of the arguments `x` and an optional `axis`.
 */
const reduction =
  (f: (x: Value, axis: number | null) => Value) =>
  (_ctx: Context, x: unknown, axis?: unknown): unknown =>
    toData(f(toTensor(x), axisOf(axis)))

const MATH_EXTRA: PreludeEntry[] = [
  scalar('tan', 'The tangent (radians).', Math.tan),
  scalar('asin', 'The arcsine, in radians.', Math.asin),
  scalar('acos', 'The arccosine, in radians.', Math.acos),
  scalar('atan', 'The arctangent, in radians.', Math.atan),
  scalar('sinh', 'The hyperbolic sine.', Math.sinh),
  scalar('cosh', 'The hyperbolic cosine.', Math.cosh),
  scalar('log2', 'The base-2 logarithm.', Math.log2),
  scalar('log10', 'The base-10 logarithm.', Math.log10),
  scalar('floor', 'The largest integer not above x.', Math.floor),
  scalar('ceil', 'The smallest integer not below x.', Math.ceil),
  scalar('round', 'x rounded to the nearest integer.', Math.round),
  // A composite, not a primitive (minimum of maximum), so the registry does not list it.
  {
    namespace: 'math',
    name: 'clip',
    params: params('x, lo, hi'),
    doc: 'x limited to [lo, hi] elementwise; lo and hi may be numbers or arrays.',
    source: TENSOR,
    returns: 'number or number[]',
    impl: lift('math.clip', (x, lo, hi) => clip(x, lo, hi)),
  },
]

const arr = def('array', TENSOR)
const ARRAY: PreludeEntry[] = [
  arr(
    'linspace',
    'start, stop, n = 50',
    'n evenly spaced values from start to stop inclusive.',
    'number[]',
    (_c, a: number, b: number, n = 50) => toData(linspace(a, b, n)),
  ),
  arr(
    'arange',
    'start, stop, step = 1',
    'Values from start up to (not including) stop in steps; arange(n) is 0 … n − 1.',
    'number[]',
    (_c, a: number, b?: number, step = 1) => toData(arange(a, b, step)),
  ),
  arr('range', 'n', 'The integers 0, 1, …, n − 1.', 'number[]', (_c, n: number) =>
    Array.from({ length: n }, (_, i) => i),
  ),
  arr('zeros', 'shape', 'An array of zeros: zeros(n) or zeros([rows, cols]).', 'number[]', (_c, s: unknown) =>
    toData(zeros(toShape(s, 'array.zeros'))),
  ),
  arr('ones', 'shape', 'An array of ones: ones(n) or ones([rows, cols]).', 'number[]', (_c, s: unknown) =>
    toData(full(toShape(s, 'array.ones'), 1)),
  ),
  arr('full', 'shape, value', 'An array filled with one value.', 'number[]', (_c, s: unknown, v: number) =>
    toData(full(toShape(s, 'array.full'), v)),
  ),
  arr('eye', 'n', 'The n × n identity matrix.', 'number[][]', (_c, n: number) => toData(eye(n))),
  arr(
    'stack',
    'arrays, axis = 0',
    'Join same-shaped arrays along a new axis: stack([x, y], 1) gives rows [xᵢ, yᵢ].',
    'number[][]',
    (_c, xs: unknown[], axis = 0) =>
      toData(
        stack(
          xs.map((x) => toTensor(x, 'array.stack')),
          axis,
        ),
      ),
  ),
  arr(
    'concat',
    'arrays, axis = 0',
    'Join arrays end to end along an existing axis.',
    'number[]',
    (_c, xs: unknown[], axis = 0) =>
      toData(
        concat(
          xs.map((x) => toTensor(x, 'array.concat')),
          axis,
        ),
      ),
  ),
  arr(
    'reshape',
    'x, shape',
    'The same elements in a new shape (−1 for one inferred length).',
    'number[]',
    (_c, x: unknown, s: number[]) => toData(reshape(toTensor(x), s)),
  ),
  arr(
    'shape',
    'x',
    'The shape of an array: [n] for a vector, [rows, cols] for a matrix.',
    'number[]',
    (_c, x: unknown) => (typeof x === 'number' ? [] : [...shapeOf(toTensor(x))]),
  ),
  arr('len', 'x', 'The length of an array (its first axis).', 'number', (_c, x: unknown) => {
    if (Array.isArray(x) || ArrayBuffer.isView(x)) return (x as ArrayLike<unknown>).length
    throw new TypeError(`array.len: expected an array, got ${describe(x)}`)
  }),
  arr('column', 'X, j', 'Column j of a matrix, as an array.', 'number[]', (_c, X: unknown, j: number) => {
    const t = toTensor(X, 'array.column')
    const [n, d] = shapeOf(t)
    if (!(j >= 0 && j < d)) throw new RangeError(`array.column: index ${j} outside 0 … ${d - 1}`)
    const flat = toFlat(t)
    return Array.from({ length: n }, (_, i) => flat[i * d + j])
  }),
  arr(
    'take',
    'x, indices',
    'The elements (or rows) of x at the given indices.',
    'array',
    (_c, x: unknown[], idx: number[]) => idx.map((i) => x[i]),
  ),
  arr(
    'sum',
    'x, axis',
    'The sum of the elements, or along an axis.',
    'number',
    reduction((x, a) => sum(x, a)),
  ),
  arr(
    'min',
    'x, axis',
    'The smallest element, or along an axis.',
    'number',
    reduction((x, a) => min(x, a)),
  ),
  arr(
    'max',
    'x, axis',
    'The largest element, or along an axis.',
    'number',
    reduction((x, a) => max(x, a)),
  ),
  arr('argmin', 'x', 'The index of the smallest element.', 'number', (_c, x: unknown) => toData(argmin(toTensor(x)))),
  arr('argmax', 'x', 'The index of the largest element.', 'number', (_c, x: unknown) => toData(argmax(toTensor(x)))),
  arr('cumsum', 'x', 'Running sums.', 'number[]', (_c, x: unknown) => toData(cumsum(toTensor(x)))),
]

/**
 * The draws' size argument: omitted for one number, a length or a shape for an array.
 *
 * @param n The program's size argument: undefined or null, a length, or a shape.
 * @param what The entry's name, used in error messages.
 * @returns The shape, or undefined for a single draw.
 */
const sizeOf = (n: unknown, what: string) => (n === undefined || n === null ? undefined : toShape(n, what))

/**
 * A draw of `shape` (a number when no shape) from a sampler on a stream, as program data.
 *
 * @param shape The shape of the draw, or undefined for one number.
 * @param draw The sampler, given `{ shape }` or undefined, as the sampling functions of
 *   `aifn-compute/foundation/random` take their options.
 * @returns A number, or a nested array of the given shape.
 */
function sampled(shape: number[] | undefined, draw: (options: { shape: number[] } | undefined) => unknown): unknown {
  return toData(draw(shape === undefined ? undefined : { shape }) as Value)
}

const rnd = def('random', RANDOM)
const RANDOM_ENTRIES: PreludeEntry[] = [
  rnd(
    'normal',
    'n, mean = 0, sd = 1',
    'Normal draws: one number, or an array of length (or shape) n.',
    'number or number[]',
    (ctx, n: unknown, m = 0, sd = 1) =>
      sampled(sizeOf(n, 'random.normal'), (o) => draws.normal(ctx.draw(), toValue(m), toValue(sd), o)),
  ),
  rnd(
    'uniform',
    'n, low = 0, high = 1',
    'Uniform draws on [low, high): one number, or an array of length (or shape) n.',
    'number or number[]',
    (ctx, n: unknown, lo = 0, hi = 1) =>
      sampled(sizeOf(n, 'random.uniform'), (o) => draws.uniform(ctx.draw(), toValue(lo), toValue(hi), o)),
  ),
  rnd(
    'bernoulli',
    'p, n',
    'Coin flips (1 with probability p, else 0), elementwise over an array p, or n flips of one p.',
    'number or number[]',
    (ctx, p: unknown, n?: unknown) =>
      sampled(sizeOf(n, 'random.bernoulli') ?? (typeof p === 'number' ? undefined : [...shapeOf(toTensor(p))]), (o) =>
        draws.bernoulli(ctx.draw(), toValue(p), o),
      ),
  ),
  rnd(
    'exponential',
    'n, rate = 1',
    'Exponential draws with the given rate (mean 1 / rate).',
    'number or number[]',
    (ctx, n: unknown, rate = 1) =>
      sampled(sizeOf(n, 'random.exponential'), (o) => draws.exponential(ctx.draw(), toValue(rate), o)),
  ),
  rnd(
    'integers',
    'high, n',
    'Uniform integers in 0 … high − 1: one, or an array of length (or shape) n.',
    'number or number[]',
    (ctx, high: number, n?: unknown) =>
      sampled(sizeOf(n, 'random.integers'), (o) => draws.integers(ctx.draw(), high, o)),
  ),
  rnd(
    'choice',
    'items, n, replace = true',
    'n items drawn from an array (or indices from 0 … items − 1 when items is a number).',
    'array',
    (ctx, items: unknown, n: number, replace = true) => {
      const pool = typeof items === 'number' ? null : (items as unknown[])
      const count = pool === null ? (items as number) : pool.length
      const idx = toFlat(draws.choice(ctx.draw(), count, n, { replace }))
      return Array.from(idx, (k) => (pool === null ? k : pool[k]))
    },
  ),
  rnd('shuffle', 'items', 'A shuffled copy of an array.', 'array', (ctx, items: unknown[]) =>
    draws.shuffle(ctx.draw(), [...items]),
  ),
  rnd('permutation', 'n', 'A random ordering of 0 … n − 1.', 'number[]', (ctx, n: number) =>
    Array.from(toFlat(draws.permutation(ctx.draw(), n))),
  ),
]

const sts = def('stats', STATS)
const STATS_ENTRIES: PreludeEntry[] = [
  def('stats', TENSOR)(
    'mean',
    'x, axis',
    'The mean of the elements, or along an axis.',
    'number',
    reduction((x, a) => mean(x, a)),
  ),
  def('stats', TENSOR)(
    'std',
    'x, axis',
    'The standard deviation (divisor n), or along an axis.',
    'number',
    reduction((x, a) => std(x, a)),
  ),
  def('stats', TENSOR)(
    'var',
    'x, axis',
    'The variance (divisor n), or along an axis.',
    'number',
    reduction((x, a) => variance(x, a)),
  ),
  sts('median', 'x', 'The median of the elements.', 'number', (_c, x: unknown) => median(toTensor(x, 'stats.median'))),
  sts(
    'quantile',
    'x, q',
    'The q-quantile (linear interpolation); q may be an array of probabilities.',
    'number or number[]',
    (_c, x: unknown, q: unknown) =>
      typeof q === 'number'
        ? quantile(toTensor(x, 'stats.quantile'), q)
        : toData(quantile(toTensor(x, 'stats.quantile'), toTensor(q))),
  ),
  sts(
    'histogram',
    'x, bins = 10',
    'Counts in equal-width bins: { edges, counts, density }.',
    '{ edges, counts, density }',
    (_c, x: unknown, bins = 10) => {
      const h = histogram(toTensor(x, 'stats.histogram'), { bins })
      return { edges: toData(h.edges), counts: toData(h.counts), density: toData(h.density) }
    },
  ),
]

const lin = def('linalg', LINALG)
const lt = def('linalg', TENSOR)
/**
 * A program's matrix (or vector) argument as a tensor.
 *
 * @param x The argument.
 * @param what The entry's name, used in error messages.
 * @returns The tensor.
 */
const mat = (x: unknown, what: string): Tensor => toTensor(x, what)
const LINALG_ENTRIES: PreludeEntry[] = [
  lt(
    'matmul',
    'A, B',
    'The matrix product A·B (a vector on the right is a column).',
    'number[][]',
    (_c, a: unknown, b: unknown) => toData(matmul(mat(a, 'linalg.matmul'), mat(b, 'linalg.matmul'))),
  ),
  lt('dot', 'a, b', 'The inner product of two vectors.', 'number', (_c, a: unknown, b: unknown) =>
    toData(dot(mat(a, 'linalg.dot'), mat(b, 'linalg.dot'))),
  ),
  lt('outer', 'a, b', 'The outer product abᵀ of two vectors.', 'number[][]', (_c, a: unknown, b: unknown) =>
    toData(outer(mat(a, 'linalg.outer'), mat(b, 'linalg.outer'))),
  ),
  lt('transpose', 'A', 'The transpose: rows become columns.', 'number[][]', (_c, a: unknown) =>
    toData(transpose(mat(a, 'linalg.transpose'))),
  ),
  lt('norm', 'x', 'The Euclidean (Frobenius) norm.', 'number', (_c, x: unknown) => toData(norm(mat(x, 'linalg.norm')))),
  lin(
    'solve',
    'A, b',
    'The solution x of A·x = b (b a vector or a matrix of columns), by LU.',
    'number[]',
    (_c, a: unknown, b: unknown) => toData(solve(mat(a, 'linalg.solve'), mat(b, 'linalg.solve'))),
  ),
  lin('inv', 'A', 'The inverse of a square matrix.', 'number[][]', (_c, a: unknown) =>
    toData(inverse(mat(a, 'linalg.inv'))),
  ),
  lin('det', 'A', 'The determinant of a square matrix.', 'number', (_c, a: unknown) =>
    toData(det(mat(a, 'linalg.det'))),
  ),
  lin(
    'cholesky',
    'A',
    'The lower-triangular L with L·Lᵀ = A, for symmetric positive-definite A.',
    'number[][]',
    (_c, a: unknown) => toData(cholesky(mat(a, 'linalg.cholesky')).L),
  ),
  lin(
    'eigh',
    'A',
    'Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix.',
    '{ values, vectors }',
    (_c, a: unknown) => {
      const e = eigh(mat(a, 'linalg.eigh'))
      return { values: toData(e.values), vectors: toData(e.vectors) }
    },
  ),
  lin('svd', 'A', 'The thin SVD A = U·diag(S)·Vᵀ, singular values descending.', '{ U, S, V }', (_c, a: unknown) => {
    const r = svd(mat(a, 'linalg.svd'))
    return { U: toData(r.U), S: toData(r.S), V: toData(r.V) }
  }),
  lin('qr', 'A', 'The reduced QR factorisation A = Q·R.', '{ Q, R }', (_c, a: unknown) => {
    const r = qr(mat(a, 'linalg.qr'))
    return { Q: toData(r.Q), R: toData(r.R) }
  }),
  lin(
    'lstsq',
    'A, b',
    'The least-squares solution x minimising ‖A·x − b‖ (minimum norm when rank deficient).',
    'number[]',
    (_c, a: unknown, b: unknown) => toData(lstsq(mat(a, 'linalg.lstsq'), mat(b, 'linalg.lstsq')).x),
  ),
]

const SIGNAL_ENTRIES: PreludeEntry[] = [
  def('signal', 'aifn-compute/foundation/fourier')(
    'spectrum',
    'x',
    'The magnitudes |X_k| of the DFT of a real signal at frequencies k = 0 … ⌊n/2⌋.',
    'number[]',
    (_c, x: unknown) => toData(complexAbs(rfft(mat(x, 'signal.spectrum')) as Tensor)),
  ),
  def('signal', 'aifn-compute/foundation/fourier')(
    'frequencies',
    'n, d = 1',
    'The frequencies of signal.spectrum for n samples spaced d apart.',
    'number[]',
    (_c, n: number, d = 1) => toData(rfftfreq(n, d)),
  ),
  def('signal', 'aifn-compute/foundation/convolution')(
    'convolve',
    'x, h, mode = "full"',
    'The convolution of x with h: "full", "same" or "valid".',
    'number[]',
    (_c, x: unknown, h: unknown, mode: 'full' | 'same' | 'valid' = 'full') =>
      toData(convolve(mat(x, 'signal.convolve'), mat(h, 'signal.convolve'), { mode })),
  ),
]

/**
 * A value as `print` writes it: a string as itself, numbers to six significant digits, anything else as JSON (with
 * numbers rounded alike) cut to 400 characters, or its string form when it has no JSON.
 *
 * @param x The value printed.
 * @returns Its text.
 */
const show = (x: unknown): string => {
  if (typeof x === 'string') return x
  if (typeof x === 'number') return String(Number(x.toPrecision(6)))
  try {
    const text = JSON.stringify(x, (_k, v: unknown) => (typeof v === 'number' ? Number(v.toPrecision(6)) : v))
    return text === undefined ? String(x) : text.length > 400 ? `${text.slice(0, 400)}…` : text
  } catch {
    return String(x)
  }
}

const top = def(null, SELF)
const TOP: PreludeEntry[] = [
  top(
    'seed',
    's',
    'Seed the random draws that follow; the same program and seed give the same data.',
    'nothing',
    (ctx, s: number | string) => {
      ctx.seed(s)
    },
  ),
  top('print', '...values', 'Write values to the output panel.', 'nothing', (ctx, ...values: unknown[]) => {
    ctx.log(values.map(show).join(' '))
  }),
]

/** The core prelude: `seed` and `print`, then the namespaces. */
export const corePrelude: Prelude = withPrelude(
  makePrelude(CORE_NAMESPACES, []),
  makePrelude(
    [],
    [
      ...TOP,
      ...mathFromRegistry(),
      ...MATH_EXTRA,
      ...ARRAY,
      ...RANDOM_ENTRIES,
      ...STATS_ENTRIES,
      ...LINALG_ENTRIES,
      ...SIGNAL_ENTRIES,
    ],
  ),
)

/**
 * The root stream of a run: the run's seed, keyed further by the program's own `seed(s)` when it calls one, so a seed
 * control outside the code still varies the data.
 *
 * @param runSeed The run's seed (`runProgram`'s `seed` option).
 * @param codeSeed The seed the program passed to `seed(s)`, if it called it.
 * @returns The stream every random call of the run draws a child of.
 *
 * @example The program's seed keys the run's
 * const draw = (root) => uniform(root, 0, 1)
 * print('run seed 1:       ', draw(rootStream(1)))
 * print('run 1, seed(7):   ', draw(rootStream(1, 7)))
 * print('run 2, seed(7):   ', draw(rootStream(2, 7)))
 * print('run 1 again:      ', draw(rootStream(1)))
 */
export function rootStream(runSeed: number | string, codeSeed?: number | string) {
  const root = stream(runSeed)
  return codeSeed === undefined ? root : child(root, 'seed', codeSeed)
}

/** Constants a program sees at the top level. */
export const CONSTANTS = { PI: Math.PI, E: Math.E } as const
