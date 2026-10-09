/**
 * Draws built on `randomBits` (design K §6). Every sampler takes the stream first and draws a whole block of words at
 * once, so a draw depends only on the stream's key and position, and a batch of $n$ draws costs one pass. None is
 * differentiable.
 *
 * Word use (fixed, so later draws never shift with the values drawn):
 * - a uniform uses 2 words (53 random bits); normals use 4 words per pair (Box–Muller on two uniforms, both outputs
 *   kept), so `normal(s, 0, 1, { shape: [n] })` uses $4\lceil n/2 \rceil$ words and a scalar normal uses 4 (the sine
 *   is discarded);
 * - a bounded integer uses 1 word (2 for a bound above $2^{32}$); the rare rejected word is redrawn from that element's
 *   own child key;
 * - samplers with rejection loops (`drawEach`) give element $i$ the child key `child(s, '~', position + i)` and
 *   advance the stream by one word per element, so a variable number of trials never moves another element's draws.
 *
 * Elementwise samplers take number or tensor parameters, broadcast together and to an optional `{ shape }` (which
 * wins, as NumPy's `size`). Numbers in and no shape give a number out; otherwise a float64 tensor, filled in
 * row-major order. A batch is not the same as successive scalar calls: a scalar normal discards the second output of
 * its pair.
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

/** `true` when any parameter type in `P` is a tensor, else `false`. */
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

/**
 * Check that a parameter shape broadcasts to the requested shape (the requested shape wins, as NumPy's `size`): the
 * two must broadcast together to exactly `to`. Throws `ShapeError` otherwise.
 *
 * @param name The sampler's name for error messages.
 * @param from The shape of the parameters (their broadcast shape, or a batch shape).
 * @param to The output shape requested.
 *
 * @example A batch of parameters fits a larger output, not a smaller one
 * checkBroadcast('normal', [3], [2, 3])
 * print('[3] to [2, 3]: ok')
 * try {
 *   checkBroadcast('normal', [2, 3], [3])
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
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

/**
 * The output shape of an elementwise sampler: `options.shape` (checked against the parameters), or the parameters'
 * broadcast shape.
 *
 * @param name The sampler's name for error messages.
 * @param params The parameters; numbers count as scalars.
 * @param options The sampler's options; only `shape` is read.
 * @returns The shape of the draw.
 */
function outputShape(name: string, params: readonly Param[], options: SampleOptions | undefined): readonly number[] {
  const joint = broadcastShapes([], ...params.filter(isTensor).map((t) => t.shape))
  if (options?.shape === undefined) return joint
  checkBroadcast(name, joint, options.shape)
  return options.shape
}

/**
 * A vector-valued parameter (a categorical's weights, a multivariate normal's mean) as its batch shape, event length
 * and row-major values. The last axis is the event; leading axes are a batch. Throws `ShapeError` for a scalar tensor.
 *
 * @param x The parameter: a tensor of rank at least 1, or an array of numbers (one vector, no batch).
 * @param name The sampler's name for error messages.
 * @returns `batch`, the shape of `x` without its last axis; `k`, the length of that axis; `values`, a fresh
 *   `Float64Array` of every value, row $b$ of the batch occupying entries $bk$ to $bk + k - 1$.
 *
 * @example Two weight vectors of length 3
 * const { batch, k, values } = eventRows(tensor([[1, 2, 3], [4, 5, 6]]), 'categorical')
 * print('batch =', batch)
 * print('k =', k)
 * print('values =', values)
 */
export function eventRows(
  x: Tensor | ArrayLike<number>,
  name: string,
): { batch: number[]; k: number; values: Float64Array } {
  if (!isTensor(x)) return { batch: [], k: x.length, values: Float64Array.from(x) }
  if (x.shape.length === 0) throw new ShapeError(name, `${name}: needs a vector (or a batch of vectors), not a scalar`)
  return { batch: x.shape.slice(0, -1), k: x.shape[x.shape.length - 1], values: Float64Array.from(toFlat(x)) }
}

/**
 * For each element of an output of shape `shape` (in row-major order), the row-major index of the batch element it
 * broadcasts from. `batch` must broadcast to `shape`.
 *
 * @param batch The batch shape of the parameters.
 * @param shape The output shape.
 * @returns One batch index per output element, as plain numbers.
 *
 * @example A batch of 2 broadcast to a 3 × 2 output, and a single vector to 3 draws
 * print('[2] to [3, 2]:', batchIndex([2], [3, 2]))
 * print('[] to [3]:', batchIndex([], [3]))
 */
export function batchIndex(batch: readonly number[], shape: readonly number[]): number[] {
  const n = batch.reduce((a, b) => a * b, 1)
  return toFlat(broadcastTo(reshape(arange(n), batch), shape))
}

/**
 * The values of a vector parameter as a plain `ArrayLike` of numbers. An array is returned as it is (not copied); a
 * tensor must have rank 1 (else `ShapeError`) and is flattened to a new array.
 *
 * @param x The vector: a rank-1 tensor or an array of numbers.
 * @param name The sampler's name for error messages.
 * @returns The values of `x`, in order.
 */
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

/**
 * $n$ uniforms in $[0, 1)$ with 53 random bits each: the top 27 bits of one word and the top 26 of the next, as
 * MT19937's `genrand_res53`. The base of every continuous sampler.
 *
 * @param s The stream; advanced by $2n$ words.
 * @param n The number of uniforms.
 * @returns A new `Float64Array` of $n$ values, each a multiple of $2^{-53}$ in $[0, 1)$.
 *
 * @example Three uniforms use six words
 * const s = stream(0)
 * print('u =', units(s, 3))
 * print('position =', s.position)
 */
export function units(s: Stream, n: Size): Float64Array {
  const w = randomBits(s, 2 * n)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = ((w[2 * i] >>> 5) * TWO_POW_26 + (w[2 * i + 1] >>> 6)) / TWO_POW_53
  return out
}

/**
 * $n$ standard normals by Box–Muller (Box and Muller, 1958), using both outputs of each pair: from uniforms $u_1, u_2$,
 * $z_1 = r\cos(2\pi u_2)$ and $z_2 = r\sin(2\pi u_2)$ with $r = \sqrt{-2\log(1 - u_1)}$. There is no clamp on the tail:
 * $1 - u_1$ lies in $(0, 1]$, so $\lvert z \rvert$ can reach about 8.57 (the limit set by 53-bit uniforms, probability
 * 1e-17).
 *
 * @param s The stream; advanced by $4\lceil n/2 \rceil$ words (two uniforms per pair, even when $n$ is odd).
 * @param n The number of normals.
 * @returns A new `Float64Array` of $n$ independent standard normal values.
 *
 * @example Three normals use two pairs, so eight words
 * const s = stream(0)
 * print('z =', standardNormals(s, 3))
 * print('position =', s.position)
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

/**
 * One uniform integer below $m$ from the words at `at`, or null when they fall in the rejected tail (the top partial
 * copy of $\{0, \dots, m - 1\}$, which would bias the residues).
 *
 * @param words The random words.
 * @param at The index of the first word to use: one word for $m \le 2^{32}$, two (53 bits) above.
 * @param m The bound, an integer with $1 \le m \le 2^{53}$.
 * @returns An integer in $[0, m)$, or null to ask for fresh words.
 */
function boundedFrom(words: Uint32Array, at: number, m: number): number | null {
  if (m <= TWO_POW_32) {
    // Reject the top partial copy of {0, …, m − 1} so every residue is equally likely.
    const u = words[at]
    return u < TWO_POW_32 - (TWO_POW_32 % m) ? u % m : null
  }
  const u = (words[at] >>> 5) * TWO_POW_26 + (words[at + 1] >>> 6)
  return u < TWO_POW_53 - (TWO_POW_53 % m) ? u % m : null
}

/**
 * The number of words one bounded integer uses.
 *
 * @param m The bound.
 * @returns 1 for $m \le 2^{32}$, else 2.
 */
const wordsFor = (m: number) => (m <= TWO_POW_32 ? 1 : 2)

/**
 * Throws `DomainError` unless $m$ is an integer bound with $1 \le m \le 2^{53}$.
 *
 * @param m The bound to check.
 * @param name The sampler's name for error messages.
 */
function checkBound(m: number, name: string): void {
  if (!(Number.isInteger(m) && m >= 1 && m <= TWO_POW_53))
    throw new DomainError(name, `${name}: needs an integer bound 1 ≤ n ≤ 2^53, got ${m}`)
}

/**
 * Uniform integers, element $i$ below `bounds[i]`, without modulo bias. Element $i$ uses its own words (1, or 2 for a
 * bound above $2^{32}$); a rejected word is redrawn from the child stream `child(s, '~', p)`, $p$ the position of
 * the element's first word, so the words each element uses are fixed in advance. Throws `DomainError` for a bound
 * that is not an integer in $[1, 2^{53}]$.
 *
 * @param s The stream; advanced by the total number of words the bounds use (one each up to $2^{32}$).
 * @param bounds The exclusive upper bound of each element, one per draw.
 * @param name The caller's name for error messages (default `'integers'`).
 * @returns A new `Float64Array` with element $i$ in $\{0, \dots, \text{bounds}_i - 1\}$.
 *
 * @example A coin, a die and a percentage in one block
 * print('draws =', boundedIntegers(stream(0), [2, 6, 100]))
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

/**
 * Parameters broadcast to an output. Numbers stay numbers; tensors are broadcast to the output shape and flattened.
 *
 * @param name The sampler's name for error messages.
 * @param params The parameters, numbers or tensors.
 * @param options The sampler's options; only `shape` is read.
 * @returns `shape`, the output shape (null for a scalar draw: every parameter a number and no shape); `n`, the number
 *   of draws; `columns`, each parameter as a number or as its row-major values over the output.
 */
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

/**
 * A broadcast parameter's value at one element of the output.
 *
 * @param c The parameter: a number (the same for every element) or its values over the output.
 * @param k The row-major index of the element.
 * @returns The parameter's value there.
 */
const at = (c: ArrayLike<number> | number, k: number) => (typeof c === 'number' ? c : c[k])

/**
 * Draw elementwise from a block: `base(s, n)` draws $n$ base variates at once (uniforms, normals), and
 * `f(z, ...values)` maps element $k$'s variate and its broadcast parameters to the draw. How samplers with a fixed
 * number of words per draw are written.
 *
 * @param name The sampler's name for error messages.
 * @param params The parameters (numbers or tensors), broadcast together and to `options.shape`.
 * @param options The sampler's options: `shape`, the output shape (default: the parameters' broadcast shape).
 * @param base Draws $n$ base variates from the stream, e.g. `units` or `standardNormals`.
 * @param s The stream, passed to `base` and advanced by it.
 * @param f Maps one base variate and that element's parameter values, in the order of `params`, to the draw.
 * @returns A number when every parameter is a number and no shape is given, else a float64 tensor of the output shape.
 *
 * @example A Rayleigh sampler by inversion
 * const rayleigh = (s, sigma, options) =>
 *   drawBlock('rayleigh', [sigma], options, units, s, (u, sig) => sig * Math.sqrt(-2 * Math.log1p(-u)))
 * print('one draw =', rayleigh(stream(0), 1))
 * print('per scale =', rayleigh(stream(0), tensor([1, 10, 100])))
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
 * Draw elementwise with a variable number of words per element (rejection samplers): element $k$ draws from its own
 * child stream `child(s, '~', position + k)`, and `s` advances by one word per element, so how many trials one element
 * needs never moves another's draws.
 *
 * @param name The sampler's name for error messages.
 * @param params The parameters (numbers or tensors), broadcast together and to `options.shape`.
 * @param options The sampler's options: `shape`, the output shape (default: the parameters' broadcast shape).
 * @param s The stream; advanced by one word per element (the words themselves are not drawn).
 * @param draw Draws one element from its own stream, given that element's parameter values in the order of `params`.
 * @returns A number when every parameter is a number and no shape is given, else a float64 tensor of the output shape.
 *
 * @example A geometric sampler by repeated trials
 * const trials = (r, p) => {
 *   let k = 1
 *   while (uniform(r) >= p) k++
 *   return k
 * }
 * print('trials to a success =', drawEach('geometric', [0.5], { shape: [8] }, stream(0), trials))
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

/**
 * Uniform draws on $[a, b)$ (default $[0, 1)$), $a + (b - a)u$ elementwise over broadcast $a$ and $b$. Two words per
 * draw.
 *
 * @param s The stream; advanced by 2 words per draw.
 * @param a The lower end of the interval (default 0): a number or a tensor.
 * @param b The upper end (default 1): a number or a tensor, broadcast with `a`.
 * @param options `shape`, the output shape (default: the broadcast shape of `a` and `b`).
 * @returns A number when `a` and `b` are numbers and no shape is given, else a float64 tensor.
 *
 * @example One draw, a vector of draws, and one draw per interval
 * const s = stream(0)
 * print('u =', uniform(s))
 * print('on [-1, 1) =', uniform(s, -1, 1, { shape: [3] }))
 * print('per upper end =', uniform(s, 0, tensor([1, 10, 100])))
 */
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
 * Normal draws $\mu + \sigma z$ with the given mean $\mu$ and standard deviation $\sigma$ (default $\Gauss(0, 1)$),
 * elementwise over broadcast parameters, by Box–Muller (see `standardNormals`).
 *
 * @param s The stream; advanced by $4\lceil n/2 \rceil$ words for $n$ draws (4 for a single one).
 * @param mean The mean $\mu$ (default 0): a number or a tensor.
 * @param sd The standard deviation $\sigma$ (default 1): a number or a tensor, broadcast with `mean`.
 * @param options `shape`, the output shape (default: the broadcast shape of `mean` and `sd`).
 * @returns A number when both parameters are numbers and no shape is given, else a float64 tensor.
 *
 * @example A few draws, and one per mean
 * const s = stream(0)
 * print('z =', normal(s, 0, 1, { shape: [4] }))
 * print('per mean =', normal(s, tensor([0, 100]), 1))
 *
 * @example The sample moments of many draws
 * const x = normal(stream(1), 5, 2, { shape: [10000] })
 * const m = sum(x) / 10000
 * print('mean =', m)
 * print('sd =', Math.sqrt(sum(mul(sub(x, m), sub(x, m))) / 10000))
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
 * `normal(s, mean, sd, { shape })` with the shape spelled as a length. Always a tensor, even for numbers.
 *
 * @param s The stream; advanced as by `normal`.
 * @param n The number of draws, or the output shape.
 * @param mean The mean (default 0): a number, or a tensor that broadcasts to the output shape.
 * @param sd The standard deviation (default 1): a number, or a tensor that broadcasts to the output shape.
 * @returns A float64 tensor of shape `[n]`, or of shape `n` when it is a list.
 *
 * @example A vector and a matrix of normals
 * const s = stream(0)
 * print('3 draws =', normals(s, 3))
 * print('2 × 2, mean 10, sd 0.1 =', normals(s, [2, 2], 10, 0.1))
 */
export function normals(s: Stream, n: Size | readonly Size[], mean: Param = 0, sd: Param = 1): Tensor {
  return normal(s, mean, sd, { shape: typeof n === 'number' ? [n] : n })
}

/**
 * Exponential draws with rate $\lambda > 0$ (mean $1/\lambda$), by inversion, $-\log(1 - u)/\lambda$, elementwise
 * over broadcast $\lambda$. The rate is not checked.
 *
 * @param s The stream; advanced by 2 words per draw.
 * @param rate The rate $\lambda$ (default 1): a number or a tensor.
 * @param options `shape`, the output shape (default: the shape of `rate`).
 * @returns A number when `rate` is a number and no shape is given, else a float64 tensor.
 *
 * @example Draws, and the mean of many at rate 2
 * print('rate 2 =', exponential(stream(0), 2, { shape: [4] }))
 * print('mean =', sum(exponential(stream(1), 2, { shape: [10000] })) / 10000)
 */
export function exponential<R extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  rate?: R,
  options?: O,
): Drawn<[R], O> {
  return drawBlock('exponential', [rate ?? 1], options, units, s, (u, r) => -Math.log1p(-u) / r) as Drawn<[R], O>
}

// ── Discrete samplers ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $\Bern(p)$ draws: 1 with probability $p$ (when a uniform $u < p$), else 0, elementwise over broadcast $p$.
 * Probabilities outside $[0, 1]$ are not checked: they act as 0 or 1.
 *
 * @param s The stream; advanced by 2 words per draw.
 * @param p The probability of a 1: a number or a tensor.
 * @param options `shape`, the output shape (default: the shape of `p`).
 * @returns A number when `p` is a number and no shape is given, else a float64 tensor of 0s and 1s.
 *
 * @example Ten coin flips at 0.3, and one draw per probability
 * const s = stream(0)
 * print('p = 0.3:', bernoulli(s, 0.3, { shape: [10] }))
 * print('p = 0, 0.5, 1:', bernoulli(s, tensor([0, 0.5, 1])))
 */
export function bernoulli<P extends Param, O extends SampleOptions = object>(
  s: Stream,
  p: P,
  options?: O,
): Drawn<[P], O> {
  return drawBlock('bernoulli', [p], options, units, s, (u, q) => (u < q ? 1 : 0)) as Drawn<[P], O>
}

/**
 * Uniform integers in $\{0, \dots, n - 1\}$ for an integer $1 \le n \le 2^{53}$, without modulo bias (see
 * `boundedIntegers`). Throws `DomainError` for any other $n$.
 *
 * @param s The stream; advanced by one word per draw (two when $n > 2^{32}$).
 * @param n The number of possible values.
 * @param options `shape`, the output shape; omitted for a single draw.
 * @returns A number without `shape`; else a tensor of that shape, int32 (float64 when $n > 2^{31}$).
 *
 * @example A die, and ten rolls
 * const s = stream(0)
 * print('one roll =', integers(s, 6) + 1)
 * print('ten rolls =', integers(s, 6, { shape: [10] }))
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

/**
 * The categorical index of a uniform $u$ under non-negative weights, by a linear scan of their running sum. Throws
 * `DomainError` for a negative weight, or weights whose sum is not finite and positive.
 *
 * @param u A uniform in $[0, 1)$.
 * @param w The array holding the weights.
 * @param offset The index in `w` of the first weight.
 * @param k The number of weights, entries `offset` to `offset + k - 1` of `w`.
 * @returns The first $j$ whose running sum exceeds $u$ times the total, in $[0, k)$; never an index of weight 0.
 */
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
 * Categorical draws: index $k$ with probability $w_k / \sum_j w_j$, for non-negative weights (unnormalised is fine) of
 * length $K$, or a batch of weight vectors. One uniform and a linear scan per draw; for many draws from the same
 * weights use {@link aliasTable} and {@link aliasSample}. Throws `DomainError` for a negative weight or a zero sum.
 *
 * @param s The stream; advanced by 2 words per draw.
 * @param weights The weights: an array of $K$, or a tensor whose last axis (length $K$) holds the weights and whose
 *   leading axes are a batch of distributions.
 * @param options `shape`, the output shape, to which the batch shape must broadcast (default: the batch shape).
 * @returns A number for one weight vector and no shape, else an int32 tensor of indices of the output shape.
 *
 * @example One draw, and ten
 * const s = stream(0)
 * print('one =', categorical(s, [1, 2, 7]))
 * print('ten =', categorical(s, [1, 2, 7], { shape: [10] }))
 *
 * @example One draw per row of a batch
 * print('per row =', categorical(stream(0), tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]])))
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
 * Build Walker's alias table for non-negative weights, in $O(K)$ (Vose 1991, "A linear algorithm for generating
 * random numbers with a given distribution", IEEE TSE 17(9)). Each {@link aliasSample} is then $O(1)$. Throws
 * `DomainError` for a negative weight, or weights whose sum is not finite and positive.
 *
 * @param weights The $K$ weights, unnormalised: an array or a rank-1 tensor.
 * @returns The table: for each column, the probability of keeping it and its alias.
 *
 * @example The table for weights 1, 2, 7
 * const { probability, alias } = aliasTable([1, 2, 7])
 * print('probability =', probability)
 * print('alias =', alias)
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
 * one uniform per draw), in $O(1)$ per draw.
 *
 * @param s The stream; advanced by 3 words per draw (all the columns are drawn first, then all the uniforms).
 * @param table The alias table, from `aliasTable`.
 * @param options `shape`, the output shape; omitted for a single draw.
 * @returns A number without `shape`, else an int32 tensor of indices of that shape.
 *
 * @example Frequencies match the weights 0.1, 0.2, 0.7
 * const table = aliasTable([1, 2, 7])
 * print('ten draws =', aliasSample(stream(0), table, { shape: [10] }))
 * const many = aliasSample(stream(1), table, { shape: [10000] }).data
 * print('frequencies =', [0, 1, 2].map((k) => many.filter((x) => x === k).length / 10000))
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

/**
 * Shuffle an array (or typed array) in place, uniformly over permutations (the Fisher–Yates shuffle in Durstenfeld's
 * form). The swap partners are drawn in one block.
 *
 * @param s The stream; advanced by one word per element after the first (none for fewer than two elements).
 * @param array The array to shuffle; modified in place.
 * @returns `array` itself, shuffled.
 *
 * @example Shuffle a list in place
 * const cards = ['A', 'K', 'Q', 'J', '10']
 * print('returned =', shuffle(stream(0), cards))
 * print('cards now =', cards)
 */
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

/**
 * A uniformly random permutation of $0, \dots, n - 1$, by `shuffle`.
 *
 * @param s The stream; advanced by $n - 1$ words.
 * @param n The number of items.
 * @returns An int32 tensor of shape $[n]$ holding each of $0, \dots, n - 1$ once.
 *
 * @example A random order of five items
 * const order = permutation(stream(0), 5)
 * print('order =', order)
 */
export function permutation(s: Stream, n: Size): Tensor {
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) out[i] = i
  return fromData(shuffle(s, out))
}

/** Options for {@link choice}. */
export type ChoiceOptions = {
  /** Sample with replacement (default true). */
  replace?: boolean
  /** Non-negative weights (length $n$, an array or a rank-1 tensor); uniform when omitted. */
  weights?: Tensor | ArrayLike<number>
}

/**
 * Indices drawn from $\{0, \dots, n - 1\}$, as an int32 tensor of shape `[size]` (or the given shape; filled in
 * row-major order). With replacement: uniform, or weighted through an alias table. Without replacement: a partial
 * Fisher–Yates shuffle, or for weights the exponential-keys method of Efraimidis and Spirakis (2006), "Weighted random
 * sampling with a reservoir", IPL 97(5) (keys $\log(1 - u_i)/w_i$, largest first), which returns indices in the order
 * successive weighted draws would pick them. Throws `DomainError` if more indices are asked for than there are
 * available items (without replacement, items of positive weight), or if `weights` does not have length $n$.
 *
 * @param s The stream to draw from; advanced by the words used.
 * @param n The number of items to choose from.
 * @param size The number of indices to draw, or the output shape.
 * @param options Whether to replace, and the weights; see `ChoiceOptions`.
 * @returns An int32 tensor of indices of shape `[size]`, or of shape `size` when it is a list.
 *
 * @example With and without replacement
 * print('with =', choice(stream(0), 5, 8))
 * print('without =', choice(stream(0), 5, 5, { replace: false }))
 *
 * @example Weighted, with and without replacement
 * const weights = [0.1, 0.1, 0.8]
 * print('with =', choice(stream(0), 3, 10, { weights }))
 * print('without =', choice(stream(0), 3, 3, { weights, replace: false }))
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
 * The spread of a multivariate normal: `{ covariance }`, its covariance $\Sigmamat$ ($d \times d$, symmetric positive
 * definite), or `{ choleskyFactor }`, a lower-triangular Cholesky factor $\Lmat$ with $\Sigmamat = \Lmat\Lmat^\top$
 * (entries above the diagonal are ignored).
 */
export type Spread = { covariance: Tensor } | { choleskyFactor: Tensor }
