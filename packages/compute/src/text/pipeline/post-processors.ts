/**
 * Post-processing, between the model and the encoding a network reads: special tokens added by a template
 * ("[CLS] $A [SEP]" for one sequence, "[CLS] $A [SEP] $B:1 [SEP]:1" for a pair, with token-type ids), truncation to a
 * maximum length (longest-first for pairs, with a stride: overflowing windows that overlap by `stride` tokens), and
 * padding to a fixed length, a multiple, or the longest of a batch, with the attention and special-token masks.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/** One item of a template: a sequence ($A or $B) or a special token, each with a token-type id. */
export type TemplateItem =
  { readonly sequence: 'A' | 'B'; readonly typeId: number } | { readonly special: string; readonly typeId: number }

/** A template post-processor (Hugging Face's `TemplateProcessing`). */
export interface PostProcessor {
  readonly type: 'template'
  readonly single: readonly TemplateItem[]
  readonly pair: readonly TemplateItem[]
}

/** Truncation: the longest an encoding may be (special tokens included), and what to do beyond it. */
export interface Truncation {
  readonly maxLength: number
  /** Tokens shared by consecutive overflowing windows (default 0). */
  readonly stride: number
  /** Which sequence of a pair gives tokens up (default `longestFirst`). */
  readonly strategy: 'longestFirst' | 'onlyFirst' | 'onlySecond'
  /** Keep the start (`right`, default: cut the end) or the end (`left`). */
  readonly direction: 'right' | 'left'
}

/** Padding: to a fixed length, or to the longest encoding of a batch, rounded up to a multiple. */
export interface Padding {
  /** The length to pad to; null pads a batch to its longest. */
  readonly length: number | null
  readonly multipleOf: number | null
  readonly token: string
  readonly id: number
  readonly typeId: number
  readonly direction: 'right' | 'left'
}

function parseItem(s: string): TemplateItem {
  const [name, type] = s.split(':')
  const typeId = type === undefined ? 0 : Number(type)
  if (!Number.isInteger(typeId)) throw new DomainError('template', `template: bad type id in '${s}'`)
  if (name === '$A' || name === '$0' || name === '$') return { sequence: 'A', typeId }
  if (name === '$B' || name === '$1') return { sequence: 'B', typeId }
  return { special: name, typeId }
}

/**
 * A template post-processor from Hugging Face's template strings: items separated by spaces, `$A` and `$B` the two
 * sequences, anything else a special token, each optionally `:typeId`. The pair template defaults to the single one
 * followed by `$B:1`.
 */
export function templateProcessor(single: string, pair?: string): PostProcessor {
  const s = single.trim().split(/\s+/u).map(parseItem)
  const p = pair ? pair.trim().split(/\s+/u).map(parseItem) : [...s, { sequence: 'B' as const, typeId: 1 }]
  return { type: 'template', single: s, pair: p }
}

/** BERT's template: [CLS] $A [SEP], and [CLS] $A [SEP] $B:1 [SEP]:1 for pairs. */
export function bertProcessor(cls = '[CLS]', sep = '[SEP]'): PostProcessor {
  return templateProcessor(`${cls} $A ${sep}`, `${cls} $A ${sep} $B:1 ${sep}:1`)
}

/** Truncation settings (stride 0, longest first, cut the end). */
export function truncation(maxLength: number, options: Partial<Omit<Truncation, 'maxLength'>> = {}): Truncation {
  return {
    maxLength,
    stride: options.stride ?? 0,
    strategy: options.strategy ?? 'longestFirst',
    direction: options.direction ?? 'right',
  }
}

/** Padding settings (to the longest of a batch, on the right, type id 0). */
export function padding(token: string, id: number, options: Partial<Omit<Padding, 'token' | 'id'>> = {}): Padding {
  return {
    length: options.length ?? null,
    multipleOf: options.multipleOf ?? null,
    token,
    id,
    typeId: options.typeId ?? 0,
    direction: options.direction ?? 'right',
  }
}

/** The number of special tokens a post-processor adds to one sequence or a pair. */
export function addedTokens(p: PostProcessor | null, pair: boolean): number {
  if (!p) return 0
  return (pair ? p.pair : p.single).filter((x) => 'special' in x).length
}

/**
 * The windows [start, end) of a sequence of `length` tokens cut to `max` with `stride` tokens shared between
 * neighbours: the first window, then each next one starting `max − stride` later, until one reaches the end
 * (direction `left` mirrors this from the end). One window when the sequence fits.
 */
export function truncationWindows(
  length: number,
  max: number,
  stride: number,
  direction: 'right' | 'left' = 'right',
): [number, number][] {
  if (length <= max) return [[0, length]]
  if (max <= 0) return [[0, 0]]
  if (stride >= max)
    throw new DomainError('truncation', `truncation: the stride (${stride}) must be less than the window (${max})`)
  const step = max - stride
  const out: [number, number][] = []
  for (let s = 0; ; s += step) {
    const e = Math.min(length, s + max)
    out.push([s, e])
    if (e >= length) break
  }
  return direction === 'right' ? out : out.map(([s, e]) => [length - e, length - s] as [number, number])
}

/**
 * The lengths two sequences are cut to so that together they fit `max` (Hugging Face's `longest_first`): the shorter
 * keeps up to half (or all of itself), the longer the rest.
 */
export function pairLengths(a: number, b: number, max: number): [number, number] {
  if (a + b <= max) return [a, b]
  const swap = a > b
  let n1 = swap ? b : a
  let n2: number
  if (n1 > max) n2 = n1
  else n2 = Math.max(n1, max - n1)
  if (n1 + n2 > max) {
    n1 = Math.floor(max / 2)
    n2 = n1 + (max % 2)
  }
  return swap ? [n2, n1] : [n1, n2]
}
