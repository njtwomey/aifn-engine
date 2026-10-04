/**
 * Logit processors: how a sampler reshapes a model's next-token distribution before drawing (Holtzman et al., 2020,
 * "The curious case of neural text degeneration"). Each takes logits [V] and returns logits, with −∞ on the tokens it
 * removes, so they compose; `nextTokenDistribution` applies them in Hugging Face's order (repetition penalty,
 * temperature, top-k, top-p) and reports the distribution before and after.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A language model as decoding sees it: the next-token logits [V] after a prefix of token ids. */
export type LogitsFn = (prefix: readonly number[]) => Value

/** Logits of a value as plain numbers. */
export function logitsOf(v: Value): number[] {
  const r = unwrap(v)
  if (typeof r === 'number') throw new ShapeError('decoding', 'decoding: logits must be a vector [V]')
  const t = r as Tensor
  if (t.shape.length !== 1)
    throw new ShapeError('decoding', `decoding: logits must be a vector [V], got [${t.shape.join(', ')}]`)
  return toFlat(t)
}

/** softmax of logits, with −∞ entries at probability 0. */
export function softmaxOf(logits: readonly number[]): number[] {
  // A loop, not Math.max(...logits): spreading a large vocabulary (> ~1e5 tokens) overflows the call stack.
  const m = logits.reduce((a, l) => Math.max(a, l), -Infinity)
  if (m === -Infinity) throw new DomainError('decoding', 'decoding: every token was removed')
  const e = logits.map((l) => (l === -Infinity ? 0 : Math.exp(l - m)))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

/** Logits divided by a temperature T > 0: T < 1 sharpens the distribution, T > 1 flattens it. */
export function applyTemperature(logits: readonly number[], temperature: number): number[] {
  if (!(temperature > 0))
    throw new DomainError('applyTemperature', `applyTemperature: temperature ${temperature} must be positive`)
  return logits.map((l) => l / temperature)
}

/** Keep the k largest logits (ties at the k-th value kept), the rest −∞ (Fan, Lewis and Dauphin, 2018). */
export function applyTopK(logits: readonly number[], k: Size): number[] {
  if (k <= 0 || k >= logits.length) return [...logits]
  const threshold = [...logits].sort((a, b) => b - a)[k - 1]
  return logits.map((l) => (l >= threshold ? l : -Infinity))
}

/**
 * Nucleus sampling (Holtzman et al., 2020): keep the smallest set of most probable tokens whose probability reaches
 * p, the rest −∞; at least `minKeep` tokens stay (default 1).
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
 * has its logit divided by θ when positive and multiplied by θ when negative, so θ > 1 discourages repeats.
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
  temperature?: number
  topK?: Size
  topP?: number
  repetitionPenalty?: number
}

/** A next-token distribution before and after processing. */
export type NextTokenDistribution = {
  /** softmax of the raw logits. */
  probs: number[]
  /** The distribution actually sampled from (after every processor). */
  filtered: number[]
  /** Tokens left in the support by top-k and top-p. */
  kept: boolean[]
}

/**
 * The next-token distribution after `prefix` with the processors applied in Hugging Face's order: repetition penalty,
 * temperature, top-k, top-p.
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
