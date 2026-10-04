/**
 * Positional information for attention, which is otherwise blind to order. Two families:
 *
 * - **Absolute**, added to the token embeddings: sinusoidal encodings (Vaswani et al., 2017, §3.5) and learned
 *   position embeddings (a table of one vector per position, as GPT-2; `LearnedPositions`).
 * - **Relative**, inside attention: rotary embeddings, which rotate each query and key by an angle proportional to its
 *   position so that q·k depends only on the offset (RoPE; Su et al., 2021), with the context-extension rescalings of
 *   position interpolation (Chen et al., 2023), NTK-aware base scaling and YaRN (Peng et al., 2024); linear biases
 *   −m·|p − q| added to the scores (ALiBi; Press et al., 2022); and T5's learned bias per bucket of relative distance
 *   (Raffel et al., 2020). Using none of them (NoPE; Kazemnejad et al., 2023) leaves a causal decoder to infer
 *   position from the mask alone.
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

/** The positional schemes of `TransformerBlock` and the lab: where position enters the model. */
export type PositionScheme = 'none' | 'sinusoidal' | 'learned' | 'rope' | 'alibi' | 't5'

// ── Absolute ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Sinusoidal position encodings [T, d] for `positions` (Vaswani et al., 2017, §3.5): PE[p, 2i] = sin(p·ω_i) and
 * PE[p, 2i + 1] = cos(p·ω_i) with ω_i = base^(−2i/d), wavelengths from 2π to 2π·base. PE[p + k] is a fixed rotation of
 * PE[p] in each (sin, cos) pair, so a linear map can read relative offsets. `d` must be even.
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

/** Parameters of `LearnedPositions`: one vector per position, [maxLength, d]. */
export type LearnedPositionParams = { weight: Tensor }

/**
 * Learned absolute position embeddings (as BERT and GPT-2): a table of `maxLength` vectors added to the token
 * embeddings, x [..., T, d] ↦ x + P[0:T]. Initialised N(0, 0.02²) by default. Positions beyond `maxLength` have no
 * vector, which is why learned positions do not extrapolate. `learnedPositions` adds the rows of given positions.
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

/** x [..., T, d] plus the rows of the position table [maxLength, d] at `positions` (length T). */
export function learnedPositions(table: Value, x: Value, positions: readonly number[]): Value {
  const max = shapeOfValue(table)[0]
  const beyond = positions.find((p) => p < 0 || p >= max)
  if (beyond !== undefined)
    throw new DomainError('learnedPositions', `learnedPositions: position ${beyond} is outside the table of ${max}`)
  return add(x, take(table, positions))
}

// ── Rotary ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * How rotary frequencies are rescaled to run past the training length `originalLength` by a `factor` s:
 *
 * - `linear` (position interpolation; Chen et al., 2023): every frequency divided by s, so positions up to s·L map
 *   into the trained range;
 * - `ntk` (NTK-aware): the base raised to base·s^(d/(d − 2)), stretching low frequencies most and leaving the highest
 *   nearly unchanged;
 * - `yarn` (Peng et al., 2024): frequencies that turn more than β times over L are kept, those turning fewer than α
 *   times are interpolated by s, with a ramp linear in the dimension index between (as Hugging Face's
 *   `_compute_yarn_parameters`, defaults α = 1, β = 32); queries and keys are also scaled by 0.1·ln s + 1 (the
 *   attention temperature).
 */
export type RopeScaling =
  | { kind: 'linear'; factor: number }
  | { kind: 'ntk'; factor: number }
  | { kind: 'yarn'; factor: number; originalLength: Size; alpha?: number; beta?: number }

/** Options of the rotary embedding. */
export type RopeOptions = {
  /** The base of the frequencies θ_i = base^(−2i/d). Default 10000. */
  base?: number
  /** Rescaling for longer contexts. Default none. */
  scaling?: RopeScaling
  /**
   * Which coordinates form the rotated pairs: `half` pairs i with i + d/2 (GPT-NeoX, LLaMA as implemented in Hugging
   * Face), `interleaved` pairs 2i with 2i + 1 (RoFormer, GPT-J). Default `half`.
   */
  layout?: 'half' | 'interleaved'
}

/** The rotary frequencies θ_i (d/2 of them, radians per position) and the magnitude applied to q and k. */
export type RopeFrequencies = { frequencies: number[]; magnitude: number }

/** The rotary frequencies of a head dimension `d` (even) under `options`; see `RopeScaling`. */
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

/** The constant [d, d] map x ↦ rotate(x): (−x₂, x₁) in each pair. */
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

/** cos and sin tables [T, d] of the angles p·θ_i, laid out to match the pairs, times the magnitude. */
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
 * Rotary position embedding (Su et al., 2021): rotate each pair of coordinates of x [..., T, d] by the angle p·θ_i
 * of its position p (`positions`, length T). Applied to queries and keys, the score q_p·k_q then depends on p − q
 * only: R_pᵀR_q = R_{q−p}. x·cos + rotate(x)·sin, so it differentiates in x.
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
 * ALiBi's head slopes (Press et al., 2022): for h a power of two the geometric sequence 2^(−8/h), 2^(−16/h), …,
 * 2^(−8); otherwise the slopes of the nearest lower power of two followed by every other slope of the next one.
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
 * ALiBi biases [h, Tq, Tk]: −m_h·|p − q| for query position p and key position q (Press et al., 2022), added to the
 * attention scores, so each head prefers recent keys at its own rate and nothing is learned. In a causal model only
 * q ≤ p is visible, where this is −m_h·(p − q).
 */
export function alibiBias(heads: Size, queries: readonly number[], keys: readonly number[]): Tensor {
  const slopes = alibiSlopes(heads)
  const out = new Float64Array(heads * queries.length * keys.length)
  let k = 0
  for (const m of slopes) for (const p of queries) for (const q of keys) out[k++] = -m * Math.abs(p - q)
  return fromData(out, [heads, queries.length, keys.length])
}

/** Options of T5's relative-position buckets. */
export type T5BucketOptions = {
  /** Distinguish keys before and after the query (encoder); false for a causal decoder. Default true. */
  bidirectional?: boolean
  /** Number of buckets. Default 32. */
  buckets?: Size
  /** Distances from this on share the last bucket. Default 128. */
  maxDistance?: Size
}

/**
 * T5's bucket of a relative position r = key − query (Raffel et al., 2020, §2.1; Hugging Face's
 * `_relative_position_bucket`): exact buckets for small distances, logarithmically wider ones up to `maxDistance`,
 * and with `bidirectional` separate halves for keys after the query.
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
 * T5 relative-position biases [h, Tq, Tk] from a learned table [buckets, h] (Raffel et al., 2020): the bias of query
 * position p and key position q is table[bucket(q − p), head]. A gather from the table, so its gradient reaches the
 * table.
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
