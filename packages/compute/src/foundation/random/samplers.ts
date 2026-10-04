/**
 * Draws built on `randomBits` (design K §6). Every sampler takes the stream first and draws a whole block of words at
 * once, so a draw depends only on the stream's key and position, and a batch of n draws costs one pass.
 *
 * Word use (fixed, so later draws never shift with the values drawn):
 * - a uniform uses 2 words (53 random bits); normals use 4 words per pair (Box–Muller on two uniforms, both outputs
 *   kept), so `normal(s, 0, 1, { shape: [n] })` uses 4⌈n/2⌉ words and a scalar normal uses 4 (the sine is discarded);
 * - a bounded integer uses 1 word (2 above 2³²); the rare rejected word is redrawn from that element's own child key;
 * - samplers with rejection loops (`drawEach`) give element i the child key `child(s, '~', position + i)` and advance
 *   the stream by one word per element, so a variable number of trials never moves another element's draws.
 *
 * Batched draws are filled in row-major order. A batch is not the same as successive scalar calls: a scalar normal
 * discards the second output of its pair.
 */

import {
  arange,
  broadcastShapes,
  broadcastTo,
  fromData,
  isTensor,
  reshape,
  showShape,
  toFlat,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Raw as Param, SampleOptions, Size, Stream } from 'aifn-compute/foundation/contracts'
import { child, randomBits } from './stream'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Raw as Param, SampleOptions } from 'aifn-compute/foundation/contracts'

// ── Shapes and broadcasting ──────────────────────────────────────────────────────────────────────────────────────────

type AnyTensor<P extends readonly unknown[]> = true extends {
  [K in keyof P]: P[K] extends Tensor ? true : false
}[number]
  ? true
  : false

/**
 * The result type of a sampler with parameters `P` and options `O`: a tensor when `O` sets a shape or any parameter
 * is a tensor, a number when every parameter is a number and no shape is set, and `number | Tensor` when that is only
 * known at run time.
 */
export type Drawn<P extends readonly unknown[], O> = O extends { shape: readonly number[] }
  ? Tensor
  : AnyTensor<P> extends true
    ? Tensor
    : [P[number]] extends [number | undefined]
      ? O extends { shape?: undefined }
        ? number
        : number | Tensor
      : number | Tensor

/** Check that a parameter shape broadcasts to the requested shape (the requested shape wins, as NumPy's `size`). */
export function checkBroadcast(name: string, from: readonly number[], to: readonly number[]): void {
  let joint: number[] | null = null
  try {
    joint = broadcastShapes(from, to)
  } catch {
    joint = null
  }
  if (joint === null || joint.length !== to.length || joint.some((d, k) => d !== to[k]))
    throw new ShapeError(
      name,
      `${name}: parameters of shape ${showShape(from)} do not broadcast to shape ${showShape(to)}`,
      [from, to],
    )
}

/** The output shape of an elementwise sampler: `options.shape`, or the parameters' broadcast shape. */
function outputShape(name: string, params: readonly Param[], options: SampleOptions | undefined): readonly number[] {
  const joint = broadcastShapes([], ...params.filter(isTensor).map((t) => t.shape))
  if (options?.shape === undefined) return joint
  checkBroadcast(name, joint, options.shape)
  return options.shape
}

/** A vector-valued parameter as its batch shape, event length and row-major values. */
export function eventRows(
  x: Tensor | ArrayLike<number>,
  name: string,
): { batch: number[]; k: number; values: Float64Array } {
  if (!isTensor(x)) return { batch: [], k: x.length, values: Float64Array.from(x) }
  if (x.shape.length === 0) throw new ShapeError(name, `${name}: needs a vector (or a batch of vectors), not a scalar`)
  return { batch: x.shape.slice(0, -1), k: x.shape[x.shape.length - 1], values: Float64Array.from(toFlat(x)) }
}

/** For each element of `shape` (row-major), the row-major index of the batch element it broadcasts from. */
export function batchIndex(batch: readonly number[], shape: readonly number[]): number[] {
  const n = batch.reduce((a, b) => a * b, 1)
  return toFlat(broadcastTo(reshape(arange(n), batch), shape))
}

/** A copy of an array as a plain `ArrayLike` of numbers (tensors flattened; they must be rank 1). */
function vectorValues(x: Tensor | ArrayLike<number>, name: string): ArrayLike<number> {
  if (!isTensor(x)) return x
  if (x.shape.length !== 1)
    throw new ShapeError(name, `${name}: needs a rank-1 tensor, got shape ${showShape(x.shape)}`)
  return toFlat(x)
}

// ── Blocks of base variates ──────────────────────────────────────────────────────────────────────────────────────────

const TWO_POW_26 = 67108864
const TWO_POW_32 = 4294967296
const TWO_POW_53 = 9007199254740992

/** n uniforms in [0, 1) with 53 random bits each (27 + 26 bits of two words, as MT19937's genrand_res53). */
export function units(s: Stream, n: Size): Float64Array {
  const w = randomBits(s, 2 * n)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = ((w[2 * i] >>> 5) * TWO_POW_26 + (w[2 * i + 1] >>> 6)) / TWO_POW_53
  return out
}

/**
 * n standard normals by Box–Muller (Box and Muller 1958), using both outputs of each pair, with no clamp on the tail:
 * the radius uses 1 − u in (0, 1], so |z| can reach about 8.57 (the limit set by 53-bit uniforms, probability 1e-17).
 */
export function standardNormals(s: Stream, n: Size): Float64Array {
  const pairs = Math.ceil(n / 2)
  const u = units(s, 2 * pairs)
  const out = new Float64Array(n)
  for (let k = 0; k < pairs; k++) {
    const r = Math.sqrt(-2 * Math.log(1 - u[2 * k]))
    const theta = 2 * Math.PI * u[2 * k + 1]
    out[2 * k] = r * Math.cos(theta)
    if (2 * k + 1 < n) out[2 * k + 1] = r * Math.sin(theta)
  }
  return out
}

/** One uniform integer below m (1 ≤ m ≤ 2⁵³) from `words` at `at`, or null when the words fall in the rejected tail. */
function boundedFrom(words: Uint32Array, at: number, m: number): number | null {
  if (m <= TWO_POW_32) {
    // Reject the top partial copy of {0, …, m − 1} so every residue is equally likely.
    const u = words[at]
    return u < TWO_POW_32 - (TWO_POW_32 % m) ? u % m : null
  }
  const u = (words[at] >>> 5) * TWO_POW_26 + (words[at + 1] >>> 6)
  return u < TWO_POW_53 - (TWO_POW_53 % m) ? u % m : null
}

const wordsFor = (m: number) => (m <= TWO_POW_32 ? 1 : 2)

function checkBound(m: number, name: string): void {
  if (!(Number.isInteger(m) && m >= 1 && m <= TWO_POW_53))
    throw new DomainError(name, `${name}: needs an integer bound 1 ≤ n ≤ 2^53, got ${m}`)
}

/**
 * Uniform integers, element i below `bounds[i]`, without modulo bias. Element i uses its own words (1, or 2 above
 * 2³²); a rejected word is redrawn from the element's child key, so the words each element uses are fixed in advance.
 */
export function boundedIntegers(s: Stream, bounds: ArrayLike<number>, name = 'integers'): Float64Array {
  const n = bounds.length
  let total = 0
  for (let i = 0; i < n; i++) {
    checkBound(bounds[i], name)
    total += wordsFor(bounds[i])
  }
  const start = s.position
  const words = randomBits(s, total)
  const out = new Float64Array(n)
  for (let i = 0, at = 0; i < n; at += wordsFor(bounds[i]), i++) {
    const m = bounds[i]
    let x = boundedFrom(words, at, m)
    if (x === null) {
      const retry = child(s, '~', start + at)
      while (x === null) x = boundedFrom(randomBits(retry, wordsFor(m)), 0, m)
    }
    out[i] = x
  }
  return out
}

// ── Elementwise draws over broadcast parameters ──────────────────────────────────────────────────────────────────────

/** Parameters broadcast to an output: the shape (null for a scalar draw), the size, and each parameter's values. */
function broadcastParams(
  name: string,
  params: readonly Param[],
  options: SampleOptions | undefined,
): { shape: readonly number[] | null; n: number; columns: (ArrayLike<number> | number)[] } {
  if (options?.shape === undefined && params.every((p) => typeof p === 'number'))
    return { shape: null, n: 1, columns: params as number[] }
  const shape = outputShape(name, params, options)
  const columns = params.map((p) => (typeof p === 'number' ? p : toFlat(broadcastTo(p, shape))))
  return { shape, n: shape.reduce((a, b) => a * b, 1), columns }
}

const at = (c: ArrayLike<number> | number, k: number) => (typeof c === 'number' ? c : c[k])

/**
 * Draw elementwise from a block: `base(s, n)` draws n base variates at once (uniforms, normals), and `f(z, ...values)`
 * maps element k's variate and its broadcast parameters to the draw. Returns a number when every parameter is a number
 * and no shape is given.
 */
export function drawBlock(
  name: string,
  params: readonly Param[],
  options: SampleOptions | undefined,
  base: (s: Stream, n: Size) => Float64Array,
  s: Stream,
  f: (z: number, ...values: number[]) => number,
): number | Tensor {
  const { shape, n, columns } = broadcastParams(name, params, options)
  const z = base(s, n)
  const values = new Array<number>(columns.length)
  const out = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < columns.length; i++) values[i] = at(columns[i], k)
    out[k] = f(z[k], ...values)
  }
  return shape === null ? out[0] : fromData(out, shape)
}

/**
 * Draw elementwise with a variable number of words per element (rejection samplers): element k draws from its own
 * child stream `child(s, '~', position + k)`, and `s` advances by one word per element. Returns a number when every
 * parameter is a number and no shape is given.
 */
export function drawEach(
  name: string,
  params: readonly Param[],
  options: SampleOptions | undefined,
  s: Stream,
  draw: (element: Stream, ...values: number[]) => number,
): number | Tensor {
  const { shape, n, columns } = broadcastParams(name, params, options)
  const start = s.position
  s.position += n
  const values = new Array<number>(columns.length)
  const out = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < columns.length; i++) values[i] = at(columns[i], k)
    out[k] = draw(child(s, '~', start + k), ...values)
  }
  return shape === null ? out[0] : fromData(out, shape)
}

// ── Continuous samplers ──────────────────────────────────────────────────────────────────────────────────────────────

/** Uniform draws on [a, b) (default [0, 1)), elementwise over broadcast a and b. */
export function uniform<A extends Param = number, B extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  a?: A,
  b?: B,
  options?: O,
): Drawn<[A, B], O> {
  return drawBlock('uniform', [a ?? 0, b ?? 1], options, units, s, (u, lo, hi) => lo + (hi - lo) * u) as Drawn<
    [A, B],
    O
  >
}

/**
 * Normal draws with the given mean and standard deviation (default N(0, 1)), elementwise over broadcast parameters,
 * by Box–Muller (see `standardNormals`).
 */
export function normal<M extends Param = number, D extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  mean?: M,
  sd?: D,
  options?: O,
): Drawn<[M, D], O> {
  return drawBlock('normal', [mean ?? 0, sd ?? 1], options, standardNormals, s, (z, m, d) => m + d * z) as Drawn<
    [M, D],
    O
  >
}

/**
 * Normal draws as a tensor of shape `[n]` (or the given shape), with mean and standard deviation broadcast to it:
 * `normal(s, mean, sd, { shape })` with the shape spelled as a length.
 */
export function normals(s: Stream, n: Size | readonly Size[], mean: Param = 0, sd: Param = 1): Tensor {
  return normal(s, mean, sd, { shape: typeof n === 'number' ? [n] : n })
}

/** Exponential draws with rate λ > 0 (mean 1/λ), by inversion −log(1 − u)/λ, elementwise over broadcast λ. */
export function exponential<R extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  rate?: R,
  options?: O,
): Drawn<[R], O> {
  return drawBlock('exponential', [rate ?? 1], options, units, s, (u, r) => -Math.log1p(-u) / r) as Drawn<[R], O>
}

// ── Discrete samplers ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Bernoulli(p) draws: 1 with probability p, else 0 (a number, or float64 tensor), elementwise over broadcast p.
 */
export function bernoulli<P extends Param, O extends SampleOptions = object>(
  s: Stream,
  p: P,
  options?: O,
): Drawn<[P], O> {
  return drawBlock('bernoulli', [p], options, units, s, (u, q) => (u < q ? 1 : 0)) as Drawn<[P], O>
}

/**
 * Uniform integers in {0, …, n − 1} for an integer 1 ≤ n ≤ 2⁵³, without modulo bias: a number without `shape`, else
 * an int32 tensor of that shape (float64 when n > 2³¹).
 */
export function integers(s: Stream, n: Size): number
export function integers(s: Stream, n: Size, options: SampleOptions & { shape: readonly number[] }): Tensor
export function integers(s: Stream, n: Size, options?: SampleOptions): number | Tensor
export function integers(s: Stream, n: Size, options?: SampleOptions): number | Tensor {
  if (options?.shape === undefined) return boundedIntegers(s, [n])[0]
  const count = options.shape.reduce((a, b) => a * b, 1)
  const values = boundedIntegers(s, new Float64Array(count).fill(n))
  return fromData(n <= 2 ** 31 ? Int32Array.from(values) : values, options.shape)
}

/** The categorical index of u ∈ [0, 1) under non-negative weights w[offset … offset + k − 1] (a linear scan). */
function categoricalIndex(u: number, w: ArrayLike<number>, offset: number, k: number): number {
  let total = 0
  for (let j = 0; j < k; j++) {
    // A negative weight would count in the total but never be picked, biasing every other index.
    if (w[offset + j] < 0) throw new DomainError('categorical', 'categorical needs non-negative weights')
    total += w[offset + j]
  }
  if (!(total > 0) || !Number.isFinite(total))
    throw new DomainError('categorical', 'categorical needs finite weights with a positive sum')
  const target = u * total
  let acc = 0
  let last = -1
  for (let j = 0; j < k; j++) {
    const wj = w[offset + j]
    if (wj <= 0) continue
    acc += wj
    last = j
    if (target < acc) return j
  }
  // Rounding in the running sum can leave the target just above it; the draw then belongs to the last positive weight.
  return last
}

/**
 * Categorical draws: index k with probability wₖ / Σ w, for non-negative weights (unnormalised is fine) of length K,
 * or a batch of shape [..., K]. Returns a number for one weight vector and no shape, else an int32 tensor of indices
 * of the batch (or requested) shape. One uniform and a linear scan per draw; for many draws from the same weights use
 * {@link aliasTable} and {@link aliasSample}.
 */
export function categorical(s: Stream, weights: ArrayLike<number>): number
export function categorical(s: Stream, weights: Tensor | ArrayLike<number>, options?: SampleOptions): number | Tensor
export function categorical(s: Stream, weights: Tensor | ArrayLike<number>, options?: SampleOptions): number | Tensor {
  const { batch, k, values } = eventRows(weights, 'categorical')
  if (batch.length === 0 && options?.shape === undefined) return categoricalIndex(units(s, 1)[0], values, 0, k)
  const shape = options?.shape ?? batch
  checkBroadcast('categorical', batch, shape)
  const rows = batchIndex(batch, shape)
  const u = units(s, rows.length)
  return fromData(
    Int32Array.from(rows, (row, i) => categoricalIndex(u[i], values, row * k, k)),
    shape,
  )
}

/** Walker's alias table for a categorical distribution (Vose's construction). */
export type AliasTable = {
  /** Probability of keeping column k rather than taking its alias, length K. */
  readonly probability: Float64Array
  /** The alias of column k, length K. */
  readonly alias: Int32Array
}

/**
 * Build Walker's alias table for non-negative weights (length K, an array or a rank-1 tensor), in O(K) (Vose 1991, "A
 * linear algorithm for generating random numbers with a given distribution", IEEE TSE 17(9)). Each
 * {@link aliasSample} is then O(1).
 */
export function aliasTable(weights: Tensor | ArrayLike<number>): AliasTable {
  const w = vectorValues(weights, 'aliasTable')
  const n = w.length
  let total = 0
  for (let k = 0; k < n; k++) {
    if (w[k] < 0) throw new DomainError('aliasTable', 'aliasTable needs non-negative weights')
    total += w[k]
  }
  if (!(total > 0) || !Number.isFinite(total))
    throw new DomainError('aliasTable', 'aliasTable needs finite weights with a positive sum')
  const scaled = new Float64Array(n)
  for (let k = 0; k < n; k++) scaled[k] = (w[k] * n) / total
  const probability = new Float64Array(n)
  const alias = new Int32Array(n)
  const small: number[] = []
  const large: number[] = []
  for (let k = n - 1; k >= 0; k--) (scaled[k] < 1 ? small : large).push(k)
  while (small.length && large.length) {
    const l = small.pop()!
    const g = large.pop()!
    probability[l] = scaled[l]
    alias[l] = g
    scaled[g] = scaled[g] + scaled[l] - 1
    ;(scaled[g] < 1 ? small : large).push(g)
  }
  // Leftovers are 1 up to rounding.
  for (const k of large) probability[k] = 1
  for (const k of small) probability[k] = 1
  for (let k = 0; k < n; k++) if (probability[k] === 1) alias[k] = k
  return { probability, alias }
}

/**
 * Categorical draws from an alias table: a uniform column, then keep it or take its alias (one bounded integer and
 * one uniform per draw). A number without `shape`, else an int32 tensor of that shape.
 */
export function aliasSample(s: Stream, table: AliasTable): number
export function aliasSample(s: Stream, table: AliasTable, options: SampleOptions & { shape: readonly number[] }): Tensor
export function aliasSample(s: Stream, table: AliasTable, options?: SampleOptions): number | Tensor
export function aliasSample(s: Stream, table: AliasTable, options?: SampleOptions): number | Tensor {
  const shape = options?.shape
  const n = shape === undefined ? 1 : shape.reduce((a, b) => a * b, 1)
  const columns = boundedIntegers(s, new Float64Array(n).fill(table.probability.length), 'aliasSample')
  const coins = units(s, n)
  const out = Int32Array.from(columns, (k, i) => (coins[i] < table.probability[k] ? k : table.alias[k]))
  return shape === undefined ? out[0] : fromData(out, shape)
}

/** Shuffle an array (or typed array) in place, uniformly over permutations (Fisher–Yates, Durstenfeld). Returns it. */
export function shuffle<A extends { length: number; [i: number]: unknown }>(s: Stream, array: A): A {
  const n = array.length
  if (n < 2) return array
  // Swap partners for i = n − 1, …, 1, drawn in one block.
  const partners = boundedIntegers(
    s,
    Float64Array.from({ length: n - 1 }, (_, k) => n - k),
    'shuffle',
  )
  for (let i = n - 1, k = 0; i > 0; i--, k++) {
    const j = partners[k]
    const tmp = array[i]
    array[i] = array[j]
    array[j] = tmp
  }
  return array
}

/** A uniformly random permutation of 0, …, n − 1, as an int32 tensor of shape [n]. */
export function permutation(s: Stream, n: Size): Tensor {
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) out[i] = i
  return fromData(shuffle(s, out))
}

/** Options for {@link choice}. */
export type ChoiceOptions = {
  /** Sample with replacement (default true). */
  replace?: boolean
  /** Non-negative weights (length n, an array or a rank-1 tensor); uniform when omitted. */
  weights?: Tensor | ArrayLike<number>
}

/**
 * Indices drawn from {0, …, n − 1}, as an int32 tensor of shape `[size]` (or the given shape; filled in row-major
 * order). With replacement: uniform, or weighted through an alias table. Without replacement: a partial Fisher–Yates
 * shuffle, or for weights the exponential-keys method of Efraimidis and Spirakis (2006), "Weighted random sampling
 * with a reservoir", IPL 97(5) (keys log(u)/wᵢ, largest first), which returns indices in the order successive
 * weighted draws would pick them. Throws if more indices are asked for than there are available items.
 */
export function choice(s: Stream, n: Size, size: Size | readonly Size[], options: ChoiceOptions = {}): Tensor {
  const { replace = true } = options
  const weights = options.weights === undefined ? undefined : vectorValues(options.weights, 'choice')
  const shape = typeof size === 'number' ? [size] : [...size]
  const count = shape.reduce((a, b) => a * b, 1)
  if (weights && weights.length !== n) throw new DomainError('choice', 'choice: weights must have length n')
  if (replace) {
    if (count > 0 && n < 1) throw new DomainError('choice', 'choice: nothing to choose from')
    if (!weights) return fromData(Int32Array.from(boundedIntegers(s, new Float64Array(count).fill(n), 'choice')), shape)
    return aliasSample(s, aliasTable(weights), { shape })
  }
  const out = new Int32Array(count)
  if (!weights) {
    if (count > n) throw new DomainError('choice', 'choice: size exceeds n without replacement')
    const pool = new Int32Array(n)
    for (let i = 0; i < n; i++) pool[i] = i
    const offsets = boundedIntegers(
      s,
      Float64Array.from({ length: count }, (_, i) => n - i),
      'choice',
    )
    for (let i = 0; i < count; i++) {
      const j = i + offsets[i]
      const tmp = pool[i]
      pool[i] = pool[j]
      pool[j] = tmp
      out[i] = pool[i]
    }
    return fromData(out, shape)
  }
  const u = units(s, n)
  const keyed: { k: number; key: number }[] = []
  for (let k = 0; k < n; k++) if (weights[k] > 0) keyed.push({ k, key: Math.log1p(-u[k]) / weights[k] })
  if (count > keyed.length) throw new DomainError('choice', 'choice: size exceeds the number of positive weights')
  keyed.sort((x, y) => y.key - x.key)
  for (let i = 0; i < count; i++) out[i] = keyed[i].k
  return fromData(out, shape)
}

// ── Multivariate normal ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The spread of a multivariate normal: its covariance Σ (d × d, symmetric positive definite), or a lower-triangular
 * Cholesky factor L with Σ = L Lᵀ (entries above the diagonal are ignored).
 */
export type Spread = { covariance: Tensor } | { choleskyFactor: Tensor }
