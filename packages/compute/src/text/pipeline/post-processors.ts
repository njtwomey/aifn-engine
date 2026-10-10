/**
 * Post-processing, between the model and the encoding a network reads: special tokens added by a template
 * (`[CLS] $A [SEP]` for one sequence, `[CLS] $A [SEP] $B:1 [SEP]:1` for a pair, with token-type ids), truncation to a
 * maximum length (longest-first for pairs, with a stride: overflowing windows that overlap by `stride` tokens), and
 * padding to a fixed length, a multiple, or the longest of a batch, with the attention and special-token masks.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * One item of a template: a `sequence` (`A` or `B`, written `$A` and `$B`) or a `special` token, each with the
 * token-type id (`typeId`) its tokens get.
 */
export type TemplateItem =
  { readonly sequence: 'A' | 'B'; readonly typeId: number } | { readonly special: string; readonly typeId: number }

/** A template post-processor (Hugging Face's `TemplateProcessing`). */
export interface PostProcessor {
  /** Marks a template post-processor. */
  readonly type: 'template'
  /** The template for one sequence. */
  readonly single: readonly TemplateItem[]
  /** The template for a pair of sequences. */
  readonly pair: readonly TemplateItem[]
}

/** Truncation: the longest an encoding may be (special tokens included), and what to do beyond it. */
export interface Truncation {
  /** The most tokens an encoding may hold, special tokens included. */
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
  /** Round the padded length up to a multiple of this, or null. */
  readonly multipleOf: number | null
  /** The padding token's string. */
  readonly token: string
  /** The padding token's id. */
  readonly id: number
  /** The token-type id of padding. */
  readonly typeId: number
  /** Pad at the end (`right`) or the start (`left`). */
  readonly direction: 'right' | 'left'
}

/**
 * One item of a template string: `$A`, `$0` or `$` for the first sequence, `$B` or `$1` for the second, anything else
 * a special token; a `:n` suffix gives the type id (default 0). Throws `DomainError` when the type id is not an
 * integer.
 *
 * @param s The item, such as `[SEP]:1`.
 * @returns The item.
 */
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
 * followed by `$B:1`. Throws `DomainError` when a type id is not an integer.
 *
 * @param single The template for one sequence, such as `<s> $A </s>`.
 * @param pair The template for a pair; left out, `single` then `$B:1`.
 * @returns The post-processor.
 *
 * @example RoBERTa's templates
 * const p = templateProcessor('<s> $A </s>', '<s> $A </s> </s> $B </s>')
 * const show = (items) => items.map((x) => ('special' in x ? x.special : '$' + x.sequence) + ':' + x.typeId)
 * print('single =', show(p.single))
 * print('pair =', show(p.pair))
 *
 * @example The default pair template
 * const p = templateProcessor('$A <eos>')
 * print(p.pair.map((x) => ('special' in x ? x.special : '$' + x.sequence) + ':' + x.typeId))
 */
export function templateProcessor(single: string, pair?: string): PostProcessor {
  const s = single.trim().split(/\s+/u).map(parseItem)
  const p = pair ? pair.trim().split(/\s+/u).map(parseItem) : [...s, { sequence: 'B' as const, typeId: 1 }]
  return { type: 'template', single: s, pair: p }
}

/**
 * BERT's template: `[CLS] $A [SEP]`, and `[CLS] $A [SEP] $B:1 [SEP]:1` for pairs.
 *
 * @param cls The classification token put first.
 * @param sep The separator put after each sequence.
 * @returns The post-processor.
 *
 * @example The pair template, with type ids
 * print(bertProcessor().pair.map((x) => ('special' in x ? x.special : '$' + x.sequence) + ':' + x.typeId))
 */
export function bertProcessor(cls = '[CLS]', sep = '[SEP]'): PostProcessor {
  return templateProcessor(`${cls} $A ${sep}`, `${cls} $A ${sep} $B:1 ${sep}:1`)
}

/**
 * Truncation settings (stride 0, longest first, cut the end).
 *
 * @param maxLength The most tokens an encoding may hold, special tokens included.
 * @param options `stride`, `strategy` and `direction`, each defaulting as above.
 * @returns The settings, for a tokeniser's `truncation`.
 *
 * @example Seven words into windows of six tokens, sharing one
 * const t = tokeniser({
 *   preTokeniser: whitespaceSplitPreTokeniser(),
 *   model: wordLevelStage(['[UNK]', '[CLS]', '[SEP]', 'a', 'b', 'c', 'd', 'e', 'f', 'g'], {
 *     specials: ['[UNK]', '[CLS]', '[SEP]'],
 *   }),
 *   postProcessor: bertProcessor(),
 *   truncation: truncation(6, { stride: 1 }),
 * })
 * const e = encodeText(t, 'a b c d e f g')
 * print('kept =', e.tokens)
 * print('overflowing =', e.overflowing.map((o) => o.tokens))
 * print('a pair =', encodeText(withStages(t, { truncation: truncation(7) }), 'a b c d e', 'f g').tokens)
 */
export function truncation(maxLength: number, options: Partial<Omit<Truncation, 'maxLength'>> = {}): Truncation {
  return {
    maxLength,
    stride: options.stride ?? 0,
    strategy: options.strategy ?? 'longestFirst',
    direction: options.direction ?? 'right',
  }
}

/**
 * Padding settings (to the longest of a batch, on the right, type id 0).
 *
 * @param token The padding token's string.
 * @param id The padding token's id.
 * @param options `length` (null: the longest of a batch), `multipleOf`, `typeId` and `direction`.
 * @returns The settings, for a tokeniser's `padding`.
 *
 * @example To the longest of a batch, and to a multiple of 4 on the left
 * const vocabulary = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', 'the', 'cat', 'sat', 'on', 'mat']
 * const t = tokeniser({
 *   preTokeniser: whitespaceSplitPreTokeniser(),
 *   model: wordLevelStage(vocabulary, { specials: ['[UNK]', '[CLS]', '[SEP]', '[PAD]'] }),
 *   postProcessor: bertProcessor(),
 *   padding: padding('[PAD]', 3),
 * })
 * for (const e of encodeBatch(t, ['the cat sat', 'the cat'])) print(e.tokens, e.attentionMask)
 * const left = withStages(t, { padding: padding('[PAD]', 3, { multipleOf: 4, direction: 'left' }) })
 * for (const e of encodeBatch(left, ['the cat', 'the cat sat on the mat'])) print(e.tokens)
 */
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

/**
 * The number of special tokens a post-processor adds to one sequence or a pair.
 *
 * @param p The post-processor, or null for none (0 added).
 * @param pair Whether the input is a pair.
 * @returns The number of special tokens in the template.
 *
 * @example BERT adds 2 to one sequence and 3 to a pair
 * print(addedTokens(bertProcessor(), false), addedTokens(bertProcessor(), true), addedTokens(null, true))
 */
export function addedTokens(p: PostProcessor | null, pair: boolean): number {
  if (!p) return 0
  return (pair ? p.pair : p.single).filter((x) => 'special' in x).length
}

/**
 * The windows [start, end) of a sequence of `length` tokens cut to `max` with `stride` tokens shared between
 * neighbours: the first window, then each next one starting `max - stride` later, until one reaches the end
 * (direction `left` mirrors this from the end). One window when the sequence fits, and one empty window when `max` is
 * not positive. Throws `DomainError` when `stride` is not less than `max`.
 *
 * @param length The number of tokens in the sequence.
 * @param max The most tokens in a window.
 * @param stride The tokens shared by consecutive windows.
 * @param direction `right` keeps the start first; `left` lists the windows from the end, mirrored.
 * @returns The windows, the kept one first.
 *
 * @example Ten tokens in windows of four, sharing one
 * print('right:', truncationWindows(10, 4, 1))
 * print('left:', truncationWindows(10, 4, 1, 'left'))
 * print('fits:', truncationWindows(3, 4, 1))
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
 *
 * @param a The length of the first sequence.
 * @param b The length of the second.
 * @param max The most tokens the two may hold together.
 * @returns The lengths to cut them to, in the order given; unchanged when they fit.
 *
 * @example The shorter kept whole, or both halved
 * print(pairLengths(10, 3, 8))
 * print(pairLengths(10, 12, 8))
 * print(pairLengths(2, 3, 8))
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
