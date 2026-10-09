/**
 * Positional information for attention, which is otherwise blind to order. Two families:
 *
 * - **Absolute**, added to the token embeddings: sinusoidal encodings (Vaswani et al., 2017, §3.5) and learned
 *   position embeddings (a table of one vector per position, as GPT-2; `LearnedPositions`).
 * - **Relative**, inside attention: rotary embeddings, which rotate each query and key by an angle proportional to its
 *   position so that $\qvec^\top\kvec$ depends only on the offset (RoPE; Su et al., 2021), with the context-extension
 *   rescalings of position interpolation (Chen et al., 2023), NTK-aware base scaling and YaRN (Peng et al., 2024);
 *   linear biases $-m \lvert p - q \rvert$ added to the scores (ALiBi; Press et al., 2022); and T5's learned bias per
 *   bucket of relative distance (Raffel et al., 2020). Using none of them (NoPE; Kazemnejad et al., 2023) leaves a
 *   causal decoder to infer position from the mask alone.
 *
 * Positions are absolute token indices, given as arrays, so a call that continues a cached sequence passes the
 * positions after the cache's (`continuePositions`). Throughout, $p$ is a query's position and $q$ a key's.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  matmul,
  mul,
  permute,
  shapeOfValue,
  take,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { normalInit, type Initialiser } from 'aifn-compute/nn/init'
import { tap, type Layer } from 'aifn-compute/nn/layers'
import { positionRange } from './masks'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The positional schemes a transformer can use, as a name: where position enters the model (`none` is NoPE). The
 * absolute ones (`sinusoidal`, `learned`) are added to the embeddings before the first block; `TransformerBlock` takes
 * the relative ones as its `RelativePosition`.
 */
export type PositionScheme = 'none' | 'sinusoidal' | 'learned' | 'rope' | 'alibi' | 't5'

// ── Absolute ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Sinusoidal position encodings `[T, d]` for `positions` (Vaswani et al., 2017, §3.5):
 * $\mathrm{PE}_{p, 2i} = \sin(p\,\omega_i)$ and $\mathrm{PE}_{p, 2i + 1} = \cos(p\,\omega_i)$ with
 * $\omega_i = b^{-2i/d}$ ($b$ = `base`), wavelengths from $2\pi$ up to $2\pi b^{1 - 2/d}$. $\mathrm{PE}_{p + k}$ is a
 * fixed rotation of $\mathrm{PE}_p$ in each (sin, cos) pair, so a linear map can read relative offsets. Throws
 * `DomainError` when $d$ is odd.
 *
 * @param positions The positions to encode, or a count $T$ for positions $0, \dots, T - 1$.
 * @param d The encoding width $d$ (even), the model width it is added to.
 * @param options The frequencies' base.
 * @param options.base The base $b$ of the frequencies (default 10000).
 * @returns The encodings `[T, d]`, one row per position, sines in the even columns and cosines in the odd ones.
 *
 * @example Three positions in width 4: position 0 is all sines 0 and cosines 1
 * const pe = sinusoidalPositions(3, 4)
 * print('shape:', shapeOf(pe))
 * print('PE:', pe)
 * print('squared norm of each row (d/2 = 2):', sum(square(pe), 1))
 */
export function sinusoidalPositions(positions: readonly number[] | Size, d: Size, { base = 10000 } = {}): Tensor {
  if (d % 2 !== 0) throw new DomainError('sinusoidalPositions', `sinusoidalPositions: the dimension ${d} must be even`)
  const ps = typeof positions === 'number' ? positionRange(positions) : positions
  const out = new Float64Array(ps.length * d)
  ps.forEach((p, r) => {
    for (let i = 0; i < d / 2; i++) {
      const angle = p * base ** ((-2 * i) / d)
      out[r * d + 2 * i] = Math.sin(angle)
      out[r * d + 2 * i + 1] = Math.cos(angle)
    }
  })
  return fromData(out, [ps.length, d])
}

/** Parameters of `LearnedPositions`: `weight`, the table of one vector per position, `[maxLength, d]`. */
export type LearnedPositionParams = { weight: Tensor }

/**
 * Learned absolute position embeddings (as BERT and GPT-2): a table $\Pmat$ of `maxLength` vectors added to the token
 * embeddings, $\Xmat \mapsto \Xmat + \Pmat_{0:T}$ for $\Xmat$ of shape `[..., T, d]` (positions $0, \dots, T - 1$).
 * Initialised $\Gauss(0, 0.02^2)$ by default. Positions beyond `maxLength` have no vector, which is why learned
 * positions do not extrapolate: a longer input throws `DomainError`. `learnedPositions` adds the rows of given
 * positions. The output is tapped at the layer's path.
 *
 * @param maxLength The number of positions in the table, the longest sequence the layer accepts.
 * @param d The width $d$ of each position vector, equal to the model width.
 * @param options How the table is initialised.
 * @param options.init The table's initialiser (default `normalInit(0.02)`).
 * @returns The layer: `init` draws the table, `apply` adds its first $T$ rows to the input.
 *
 * @example On a zero input the output is the table's first rows
 * const layer = LearnedPositions(8, 4)
 * const params = layer.init(stream(0))
 * const y = layer.apply(params, zeros([3, 4]))
 * print('table:', shapeOf(params.weight), 'output:', shapeOf(y))
 * print('output:', y)
 * print('table rows 0 to 2:', take(params.weight, [0, 1, 2]))
 */
export function LearnedPositions(
  maxLength: Size,
  d: Size,
  { init = normalInit(0.02) }: { init?: Initialiser } = {},
): Layer<LearnedPositionParams> {
  return {
    kind: 'LearnedPositions',
    label: `LearnedPositions(${maxLength} × ${d})`,
    init: (s: Stream) => ({ weight: init(child(s, 'weight'), [maxLength, d], { fanIn: maxLength, fanOut: d }) }),
    apply: (p, x, ctx) => {
      const s = shapeOfValue(x)
      return tap(ctx, learnedPositions(p.weight, x, positionRange(s[s.length - 2])))
    },
  }
}

/**
 * The input plus the rows of a position table at `positions`: the functional form of `LearnedPositions`, for a call
 * that does not start at position 0 (decoding after a cache). Throws `DomainError` for a position outside the table.
 * Differentiable in the table and the input.
 *
 * @param table The position table `[maxLength, d]`, one row per position.
 * @param x The token embeddings `[..., T, d]`.
 * @param positions The $T$ positions of the tokens, each in $0, \dots$, `maxLength` $- 1$; row $t$ of `x` gets the
 *   table row `positions[t]`.
 * @returns `x` plus the gathered rows, `[..., T, d]`.
 *
 * @example Rows 2 and 3 of the table added to two tokens
 * const table = tensor([[0, 0], [10, 10], [20, 20], [30, 30]])
 * print('x + rows 2 and 3:', learnedPositions(table, tensor([[1, 2], [3, 4]]), [2, 3]))
 */
export function learnedPositions(table: Value, x: Value, positions: readonly number[]): Value {
  const max = shapeOfValue(table)[0]
  const beyond = positions.find((p) => p < 0 || p >= max)
  if (beyond !== undefined)
    throw new DomainError('learnedPositions', `learnedPositions: position ${beyond} is outside the table of ${max}`)
  return add(x, take(table, positions))
}

// ── Rotary ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * How rotary frequencies are rescaled by a `factor` $s$ to run past the training length $L$ (`originalLength`, which
 * only `yarn` reads):
 *
 * - `linear` (position interpolation; Chen et al., 2023): every frequency divided by $s$, so positions up to $sL$ map
 *   into the trained range;
 * - `ntk` (NTK-aware): the base multiplied by $s^{d/(d - 2)}$, which divides the lowest frequency by exactly $s$ and
 *   leaves the highest ($\theta_0 = 1$) unchanged;
 * - `yarn` (Peng et al., 2024): frequencies that turn more than $\beta$ times over $L$ are kept, those turning fewer
 *   than $\alpha$ times are interpolated by $s$, with a ramp linear in the dimension index between (as Hugging Face's
 *   `_compute_yarn_parameters`; `alpha` $\alpha = 1$ and `beta` $\beta = 32$ by default); when $s > 1$ queries and
 *   keys are also scaled by $0.1 \ln s + 1$ (the attention temperature).
 */
export type RopeScaling =
  | { kind: 'linear'; factor: number }
  | { kind: 'ntk'; factor: number }
  | { kind: 'yarn'; factor: number; originalLength: Size; alpha?: number; beta?: number }

/** Options of the rotary embedding. */
export type RopeOptions = {
  /** The base $b$ of the frequencies $\theta_i = b^{-2i/d}$. Default 10000. */
  base?: number
  /** Rescaling for longer contexts. Default none. */
  scaling?: RopeScaling
  /**
   * Which coordinates form the rotated pairs: `half` pairs $i$ with $i + d/2$ (GPT-NeoX, LLaMA as implemented in
   * Hugging Face), `interleaved` pairs $2i$ with $2i + 1$ (RoFormer, GPT-J). Default `half`.
   */
  layout?: 'half' | 'interleaved'
}

/**
 * The rotary frequencies and magnitude of `ropeFrequencies`: `frequencies` holds the $d/2$ values $\theta_i$ (radians
 * per position, highest first), and `magnitude` multiplies the cos and sin tables, so it scales both queries and keys
 * (1 except under YaRN).
 */
export type RopeFrequencies = { frequencies: number[]; magnitude: number }

/**
 * The rotary frequencies $\theta_i = b^{-2i/d}$, $i = 0, \dots, d/2 - 1$, of a head width $d$, rescaled as
 * `RopeScaling` describes. Throws `DomainError` when $d$ is odd.
 *
 * @param d The head width $d$ (even): the width of the vectors that are rotated.
 * @param options The base and the context-extension rescaling (`layout` is not read).
 * @returns The $d/2$ frequencies and the magnitude applied to queries and keys.
 *
 * @example Plain, linear and NTK-aware frequencies of width 4 for twice the context
 * print('plain:', ropeFrequencies(4).frequencies)
 * print('linear x2:', ropeFrequencies(4, { scaling: { kind: 'linear', factor: 2 } }).frequencies)
 * print('ntk x2:', ropeFrequencies(4, { scaling: { kind: 'ntk', factor: 2 } }).frequencies)
 *
 * @example YaRN keeps fast frequencies, interpolates slow ones and raises the magnitude
 * const yarn = ropeFrequencies(8, { scaling: { kind: 'yarn', factor: 4, originalLength: 64 } })
 * print('plain:', ropeFrequencies(8).frequencies)
 * print('yarn x4:', yarn.frequencies)
 * print('magnitude 0.1 ln 4 + 1:', yarn.magnitude)
 */
export function ropeFrequencies(d: Size, options: RopeOptions = {}): RopeFrequencies {
  if (d % 2 !== 0) throw new DomainError('ropeFrequencies', `ropeFrequencies: the dimension ${d} must be even`)
  const { scaling } = options
  let base = options.base ?? 10000
  if (scaling?.kind === 'ntk') base *= scaling.factor ** (d / (d - 2))
  const theta = Array.from({ length: d / 2 }, (_, i) => base ** ((-2 * i) / d))
  if (!scaling || scaling.kind === 'ntk') return { frequencies: theta, magnitude: 1 }
  if (scaling.kind === 'linear') return { frequencies: theta.map((t) => t / scaling.factor), magnitude: 1 }
  const { factor: s, originalLength, alpha = 1, beta = 32 } = scaling
  // The dimension index at which a frequency turns `rotations` times over the original length (Hugging Face's
  // `find_correction_dim`); the ramp is linear in the index between the β and α dimensions, rounded outwards.
  const at = (rotations: number) => (d * Math.log(originalLength / (rotations * 2 * Math.PI))) / (2 * Math.log(base))
  const low = Math.max(Math.floor(at(beta)), 0)
  let high = Math.min(Math.ceil(at(alpha)), d - 1)
  if (high === low) high += 0.001
  const frequencies = theta.map((t, i) => {
    const kept = 1 - Math.min(1, Math.max(0, (i - low) / (high - low)))
    return (t / s) * (1 - kept) + t * kept
  })
  return { frequencies, magnitude: s > 1 ? 0.1 * Math.log(s) + 1 : 1 }
}

/**
 * The constant `[d, d]` matrix $\Mmat$ with $\xvec^\top\Mmat$ the quarter-turn of each pair: a pair $(x_a, x_b)$
 * becomes $(-x_b, x_a)$.
 *
 * @param d The head width $d$ (even).
 * @param layout Which coordinates form the pairs, as in `RopeOptions.layout`.
 * @returns The matrix, row index the input coordinate and column index the output one.
 */
function rotationMatrix(d: Size, layout: 'half' | 'interleaved'): Tensor {
  const m = new Float64Array(d * d)
  const h = d / 2
  for (let i = 0; i < h; i++) {
    const [a, b] = layout === 'half' ? [i, i + h] : [2 * i, 2 * i + 1]
    // out[a] = −x[b], out[b] = x[a]; with x·M, M[row = input, col = output].
    m[b * d + a] = -1
    m[a * d + b] = 1
  }
  return fromData(m, [d, d])
}

/**
 * The cos and sin tables `[T, d]` of the angles $p\,\theta_i$, laid out to match the pairs (both coordinates of pair
 * $i$ hold the same value), times the magnitude. `applyRope` multiplies by them.
 *
 * @param positions The $T$ positions $p$, one per row.
 * @param d The head width $d$ (even).
 * @param options The base, rescaling and pair layout.
 * @returns `cos` and `sin`, each `[T, d]`.
 *
 * @example Width 4, half layout: columns 0 and 2 turn one radian per position
 * const { cos, sin } = ropeTables([0, 1, 2], 4)
 * print('cos:', cos)
 * print('sin:', sin)
 */
export function ropeTables(
  positions: readonly number[],
  d: Size,
  options: RopeOptions = {},
): { cos: Tensor; sin: Tensor } {
  const { frequencies, magnitude } = ropeFrequencies(d, options)
  const layout = options.layout ?? 'half'
  const cos = new Float64Array(positions.length * d)
  const sin = new Float64Array(positions.length * d)
  positions.forEach((p, r) => {
    frequencies.forEach((f, i) => {
      const c = magnitude * Math.cos(p * f)
      const s = magnitude * Math.sin(p * f)
      const [a, b] = layout === 'half' ? [i, i + d / 2] : [2 * i, 2 * i + 1]
      cos[r * d + a] = cos[r * d + b] = c
      sin[r * d + a] = sin[r * d + b] = s
    })
  })
  return { cos: fromData(cos, [positions.length, d]), sin: fromData(sin, [positions.length, d]) }
}

/**
 * Rotary position embedding (Su et al., 2021): rotate each pair of coordinates of a row of $\Xmat$ (`[..., T, d]`) by
 * the angle $p\,\theta_i$ of its position $p$. Applied to queries and keys, the score
 * $(\Rmat_p\qvec)^\top(\Rmat_q\kvec) = \qvec^\top\Rmat_{q - p}\kvec$ then depends on $q - p$ only. Computed as
 * $\Xmat \odot \cos + \mathrm{rotate}(\Xmat) \odot \sin$, so it differentiates in $\Xmat$. Throws `ShapeError`
 * when the number of positions is not $T$.
 *
 * @param x The queries or keys `[..., T, d]`, $d$ even (one head's width).
 * @param positions The $T$ absolute positions of the rows.
 * @param options The base, rescaling and pair layout (default: base 10000, `half`, no rescaling).
 * @returns The rotated `x`, same shape.
 *
 * @example The first pair of [1, 0, 0, 0] turns one radian per position
 * const x = tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
 * print('rotated:', applyRope(x, [0, 1, 2]))
 *
 * @example A rotated score depends only on the offset between positions
 * const q = tensor([[1, 2, 3, 4]])
 * const k = tensor([[0.5, -1, 2, 1]])
 * const score = (p, r) => sum(mul(applyRope(q, [p]), applyRope(k, [r])))
 * print('query at 3, key at 1:', score(3, 1))
 * print('query at 7, key at 5:', score(7, 5))
 * print('query at 2, key at 1:', score(2, 1))
 */
export function applyRope(x: Value, positions: readonly number[], options: RopeOptions = {}): Value {
  const s = shapeOfValue(x)
  const d = s[s.length - 1]
  if (s[s.length - 2] !== positions.length)
    throw new ShapeError('applyRope', `applyRope: ${positions.length} positions for a sequence of ${s[s.length - 2]}`)
  const { cos, sin } = ropeTables(positions, d, options)
  return add(mul(x, cos), mul(matmul(x, rotationMatrix(d, options.layout ?? 'half')), sin))
}

// ── Additive biases ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * ALiBi's head slopes (Press et al., 2022): for $h$ a power of two the geometric sequence
 * $2^{-8/h}, 2^{-16/h}, \dots, 2^{-8}$; otherwise the slopes of the nearest lower power of two followed by every other
 * slope of the next one.
 *
 * @param heads The number of heads $h$.
 * @returns The $h$ slopes $m_h$, steepest first.
 *
 * @example Eight heads halve their slope each time; six take four slopes and then two of the eight-head ones
 * print('8 heads:', alibiSlopes(8))
 * print('6 heads:', alibiSlopes(6))
 */
export function alibiSlopes(heads: Size): number[] {
  const powerOfTwo = (n: number) => {
    const start = 2 ** -(2 ** -(Math.log2(n) - 3))
    return Array.from({ length: n }, (_, i) => start * start ** i)
  }
  if (Number.isInteger(Math.log2(heads))) return powerOfTwo(heads)
  const closest = 2 ** Math.floor(Math.log2(heads))
  return [
    ...powerOfTwo(closest),
    ...alibiSlopes(2 * closest)
      .filter((_, i) => i % 2 === 0)
      .slice(0, heads - closest),
  ]
}

/**
 * ALiBi biases `[h, Tq, Tk]`: $-m_h \lvert p - q \rvert$ for query position $p$ and key position $q$ (Press et al.,
 * 2022), added to the attention scores, so each head prefers recent keys at its own rate and nothing is learned. In a
 * causal model only $q \le p$ is visible, where this is $-m_h (p - q)$.
 *
 * @param heads The number of heads $h$; the slopes are `alibiSlopes(heads)`.
 * @param queries The $T_q$ query positions $p$.
 * @param keys The $T_k$ key positions $q$.
 * @returns The biases `[h, Tq, Tk]`, to pass as `bias` to attention.
 *
 * @example Two heads over three tokens: zero on the diagonal, falling with distance
 * print('bias:', alibiBias(2, [0, 1, 2], [0, 1, 2]))
 */
export function alibiBias(heads: Size, queries: readonly number[], keys: readonly number[]): Tensor {
  const slopes = alibiSlopes(heads)
  const out = new Float64Array(heads * queries.length * keys.length)
  let k = 0
  for (const m of slopes) for (const p of queries) for (const q of keys) out[k++] = -m * Math.abs(p - q)
  return fromData(out, [heads, queries.length, keys.length])
}

/** Options of T5's relative-position buckets (`t5RelativeBucket`, `t5RelativeBias`). */
export type T5BucketOptions = {
  /** Distinguish keys before and after the query (encoder); false for a causal decoder. Default true. */
  bidirectional?: boolean
  /** Number of buckets. Default 32. */
  buckets?: Size
  /** Distances from this on share the last bucket. Default 128. */
  maxDistance?: Size
}

/**
 * T5's bucket of a relative position $r = q - p$, key minus query (Raffel et al., 2020, §2.1; Hugging Face's
 * `_relative_position_bucket`): exact buckets for small distances, logarithmically wider ones up to `maxDistance`,
 * and with `bidirectional` separate halves for keys after the query. Without `bidirectional`, every key after the query
 * falls in bucket 0 with the query itself.
 *
 * @param relative The relative position $r$ (negative for a key before the query).
 * @param options The number of buckets, the distance at which they stop growing, and whether direction counts.
 * @returns The bucket index, from 0 to `buckets` $- 1$.
 *
 * @example Eight buckets up to distance 16: exact near zero, shared far away
 * const rs = [-20, -8, -4, -2, -1, 0, 1, 2, 4, 8, 20]
 * print('r:', rs)
 * print('bidirectional:', rs.map((r) => t5RelativeBucket(r, { buckets: 8, maxDistance: 16 })))
 * print('causal:', rs.map((r) => t5RelativeBucket(r, { bidirectional: false, buckets: 8, maxDistance: 16 })))
 */
export function t5RelativeBucket(relative: number, options: T5BucketOptions = {}): number {
  const { bidirectional = true, maxDistance = 128 } = options
  let n = options.buckets ?? 32
  let bucket = 0
  let r: number
  if (bidirectional) {
    n = Math.floor(n / 2)
    if (relative > 0) bucket += n
    r = Math.abs(relative)
  } else r = -Math.min(relative, 0)
  const exact = Math.floor(n / 2)
  if (r < exact) return bucket + r
  const large = exact + Math.trunc((Math.log(r / exact) / Math.log(maxDistance / exact)) * (n - exact))
  return bucket + Math.min(large, n - 1)
}

/**
 * T5 relative-position biases `[h, Tq, Tk]` from a learned table `[buckets, h]` (Raffel et al., 2020): the bias of
 * query position $p$ and key position $q$ in a head is the table's entry at row $\mathrm{bucket}(q - p)$ and that
 * head's column. A gather from the table, so its gradient reaches the table.
 *
 * @param table The learned biases `[buckets, h]`, one row per bucket and one column per head.
 * @param queries The $T_q$ query positions $p$.
 * @param keys The $T_k$ key positions $q$.
 * @param options The bucketing, as in `t5RelativeBucket`; `buckets` must match the table's rows.
 * @returns The biases `[h, Tq, Tk]`, to pass as `bias` to attention.
 *
 * @example A causal table of four buckets and two heads
 * const table = tensor([[0, 0], [1, 10], [2, 20], [3, 30]])
 * const bias = t5RelativeBias(table, [0, 1, 2], [0, 1, 2], { bidirectional: false, buckets: 4, maxDistance: 8 })
 * print('shape:', shapeOf(bias))
 * print('bias:', bias)
 */
export function t5RelativeBias(
  table: Value,
  queries: readonly number[],
  keys: readonly number[],
  options: T5BucketOptions = {},
): Value {
  const ids = new Int32Array(queries.length * keys.length)
  queries.forEach((p, i) => keys.forEach((q, j) => (ids[i * keys.length + j] = t5RelativeBucket(q - p, options))))
  const rows = take(table, fromData(ids, [queries.length, keys.length]))
  return permute(rows, [2, 0, 1])
}
