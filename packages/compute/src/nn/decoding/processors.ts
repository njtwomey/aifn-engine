/**
 * Logit processors: how a sampler reshapes a model's next-token distribution before drawing (Holtzman et al., 2020,
 * "The curious case of neural text degeneration"). Each takes the logits of a vocabulary of $V$ tokens as a plain
 * array and returns a new array, with $-\infty$ on the tokens it removes, so they compose; `nextTokenDistribution`
 * applies them in Hugging Face's order (repetition penalty, temperature, top-k, top-p) and reports the distribution
 * before and after. Processors work on concrete numbers: decoding is not differentiated.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A language model as decoding sees it: the next-token logits `[V]` after a prefix of token ids (the prompt and the
 * tokens generated so far).
 */
export type LogitsFn = (prefix: readonly number[]) => Value

/**
 * Logits of a value as plain numbers. Throws `ShapeError` unless the value is a vector.
 *
 * @param v The logits `[V]`: a tensor, or a traced value whose primal is read.
 * @returns The $V$ logits as an array.
 *
 * @example A tensor of logits as an array
 * print('logits:', logitsOf(tensor([1, 2, 3])))
 */
export function logitsOf(v: Value): number[] {
  const r = unwrap(v)
  if (typeof r === 'number') throw new ShapeError('decoding', 'decoding: logits must be a vector [V]')
  const t = r as Tensor
  if (t.shape.length !== 1)
    throw new ShapeError('decoding', `decoding: logits must be a vector [V], got [${t.shape.join(', ')}]`)
  return toFlat(t)
}

/**
 * The softmax of logits, with $-\infty$ entries at probability 0, computed after subtracting the largest logit so it
 * does not overflow. Throws `DomainError` when every logit is $-\infty$ (every token was removed).
 *
 * @param logits The $V$ logits.
 * @returns The $V$ probabilities, summing to one.
 *
 * @example Odds of one to three, and a removed token
 * print('probs:', softmaxOf([0, Math.log(3), -Infinity]))
 */
export function softmaxOf(logits: readonly number[]): number[] {
  // A loop, not Math.max(...logits): spreading a large vocabulary (> ~1e5 tokens) overflows the call stack.
  const m = logits.reduce((a, l) => Math.max(a, l), -Infinity)
  if (m === -Infinity) throw new DomainError('decoding', 'decoding: every token was removed')
  const e = logits.map((l) => (l === -Infinity ? 0 : Math.exp(l - m)))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

/**
 * Logits divided by a temperature $T > 0$: $T < 1$ sharpens the distribution, $T > 1$ flattens it. Throws
 * `DomainError` unless $T > 0$.
 *
 * @param logits The $V$ logits.
 * @param temperature The temperature $T$.
 * @returns The logits divided by $T$.
 *
 * @example The same logits at three temperatures
 * const logits = [2, 1, 0]
 * print('T = 1:', softmaxOf(logits))
 * print('T = 0.5:', softmaxOf(applyTemperature(logits, 0.5)))
 * print('T = 2:', softmaxOf(applyTemperature(logits, 2)))
 */
export function applyTemperature(logits: readonly number[], temperature: number): number[] {
  if (!(temperature > 0))
    throw new DomainError('applyTemperature', `applyTemperature: temperature ${temperature} must be positive`)
  return logits.map((l) => l / temperature)
}

/**
 * Keep the $k$ largest logits (ties at the $k$-th value kept), the rest $-\infty$ (Fan, Lewis and Dauphin, 2018).
 *
 * @param logits The $V$ logits.
 * @param k The number of tokens to keep; 0 or less, or $V$ or more, keeps every token.
 * @returns The logits with all but the top $k$ set to $-\infty$.
 *
 * @example The top two, and a tie at the second
 * print('top 2:', applyTopK([1, 3, 2, 0], 2))
 * print('tie at the 2nd:', applyTopK([1, 3, 1, 0], 2))
 */
export function applyTopK(logits: readonly number[], k: Size): number[] {
  if (k <= 0 || k >= logits.length) return [...logits]
  const threshold = [...logits].sort((a, b) => b - a)[k - 1]
  return logits.map((l) => (l >= threshold ? l : -Infinity))
}

/**
 * Nucleus sampling (Holtzman et al., 2020): keep the smallest set of most probable tokens whose probability reaches
 * $p$, the rest $-\infty$; at least `minKeep` tokens stay.
 *
 * @param logits The $V$ logits.
 * @param p The probability mass $p$ to keep; 1 or more keeps every token.
 * @param minKeep The fewest tokens kept, whatever their mass.
 * @returns The logits with the tokens outside the nucleus set to $-\infty$.
 *
 * @example Probabilities 0.5, 0.3, 0.15 and 0.05: the nucleus grows with p
 * const logits = [Math.log(0.5), Math.log(0.3), Math.log(0.15), Math.log(0.05)]
 * print('p = 0.7:', softmaxOf(applyTopP(logits, 0.7)))
 * print('p = 0.9:', softmaxOf(applyTopP(logits, 0.9)))
 * print('p = 0.3, at least 2:', softmaxOf(applyTopP(logits, 0.3, 2)))
 */
export function applyTopP(logits: readonly number[], p: number, minKeep: Size = 1): number[] {
  if (p >= 1) return [...logits]
  const probs = softmaxOf(logits)
  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a])
  const keep = new Set<number>()
  let mass = 0
  for (const i of order) {
    if (mass >= p && keep.size >= minKeep) break
    keep.add(i)
    mass += probs[i]
  }
  return logits.map((l, i) => (keep.has(i) ? l : -Infinity))
}

/**
 * The repetition penalty of CTRL (Keskar et al., 2019), as Hugging Face applies it: every token already in the prefix
 * has its logit divided by $\theta$ when positive and multiplied by $\theta$ otherwise, so $\theta > 1$ discourages
 * repeats. A token is penalised once however often it occurs.
 *
 * @param logits The $V$ logits.
 * @param prefix The token ids so far; each distinct id is penalised.
 * @param penalty The penalty $\theta$; 1 leaves the logits unchanged.
 * @returns The penalised logits.
 *
 * @example Tokens 0 and 1 have been seen: a positive logit is halved, a negative one doubled
 * print('penalty 2:', applyRepetitionPenalty([2, -1, 3], [0, 1, 0], 2))
 */
export function applyRepetitionPenalty(
  logits: readonly number[],
  prefix: readonly number[],
  penalty: number,
): number[] {
  if (penalty === 1) return [...logits]
  const seen = new Set(prefix)
  return logits.map((l, i) => (seen.has(i) ? (l > 0 ? l / penalty : l * penalty) : l))
}

/** Options of the sampling processors (all off by default). */
export type SamplingOptions = {
  /** The temperature $T > 0$ the logits are divided by (`applyTemperature`). */
  temperature?: number
  /** Keep only the $k$ most probable tokens (`applyTopK`). */
  topK?: Size
  /** Keep only the nucleus of probability $p$ (`applyTopP`). */
  topP?: number
  /** The repetition penalty $\theta$ of tokens already in the prefix (`applyRepetitionPenalty`). */
  repetitionPenalty?: number
}

/** A next-token distribution before and after processing. */
export type NextTokenDistribution = {
  /** The softmax of the raw logits. */
  probs: number[]
  /** The distribution actually sampled from (after every processor). */
  filtered: number[]
  /** Tokens left in the support after every processor (top-k and top-p remove tokens). */
  kept: boolean[]
}

/**
 * The next-token distribution after `prefix` with the processors applied in Hugging Face's order: repetition penalty,
 * temperature, top-k, top-p. Throws `DomainError` when the processors remove every token.
 *
 * @param logits The $V$ raw logits after the prefix.
 * @param prefix The token ids so far, read by the repetition penalty only.
 * @param options Which processors to apply, and how strongly.
 * @returns The raw distribution, the processed one and which tokens remain.
 *
 * @example Token 0 was seen: the penalty and a temperature of 0.5, then the top three
 * const logits = [2, 1, 0.5, 0, -1]
 * const d = nextTokenDistribution(logits, [0], { repetitionPenalty: 2, temperature: 0.5, topK: 3 })
 * print('raw:', d.probs)
 * print('filtered:', d.filtered)
 * print('kept:', d.kept)
 */
export function nextTokenDistribution(
  logits: readonly number[],
  prefix: readonly number[],
  options: SamplingOptions = {},
): NextTokenDistribution {
  let l = [...logits]
  if (options.repetitionPenalty !== undefined) l = applyRepetitionPenalty(l, prefix, options.repetitionPenalty)
  if (options.temperature !== undefined) l = applyTemperature(l, options.temperature)
  if (options.topK !== undefined) l = applyTopK(l, options.topK)
  if (options.topP !== undefined) l = applyTopP(l, options.topP)
  const filtered = softmaxOf(l)
  return { probs: softmaxOf(logits), filtered, kept: l.map((v) => v !== -Infinity) }
}
