/**
 * Aligned text: a string that remembers where each of its code units came from in an original text, so normalisation,
 * replacement and splitting can change the text while tokens keep offsets into what the user typed (as the
 * `NormalizedString` of Hugging Face `tokenizers`). Every UTF-16 code unit of `text` carries a [start, end) range of
 * `original`; inserted characters carry a zero-width range at the position they were inserted, except a prefix added
 * by `alignedPrepend`, which takes the range of the first unit.
 *
 * Every function returns a new aligned text and leaves its input unchanged. Positions are UTF-16 code units, as
 * JavaScript's string indices.
 */

/** Text with, per UTF-16 code unit, the [start, end) range of the original it came from (int32, two per unit). */
export interface AlignedText {
  /** The text as the user gave it. */
  readonly original: string
  /** The current, rewritten text. */
  readonly text: string
  /** Unit $i$ of `text` came from `original.slice(spans[2 * i], spans[2 * i + 1])`. */
  readonly spans: Int32Array
}

/**
 * The identity alignment: `text` is `original`, each code unit its own range.
 *
 * @param original The text.
 * @returns The aligned text whose unit $i$ has the range $[i, i + 1)$.
 *
 * @example The start of every alignment
 * const a = aligned('Hi!')
 * print(a.text, a.spans)
 */
export function aligned(original: string): AlignedText {
  const spans = new Int32Array(2 * original.length)
  for (let i = 0; i < original.length; i++) {
    spans[2 * i] = i
    spans[2 * i + 1] = i + 1
  }
  return { original, text: original, spans }
}

/**
 * The code units [start, end) of an aligned text, keeping their ranges.
 *
 * @param a The aligned text.
 * @param start The first unit kept.
 * @param end The unit after the last one kept.
 * @returns The slice, with the same original.
 *
 * @example The second word still points into the original
 * const a = alignedReplace(aligned('Hello  World'), /\s+/, ' ')
 * const w = alignedSlice(a, 6, 11)
 * print(w.text, originalSpan(w, 0, w.text.length))
 */
export function alignedSlice(a: AlignedText, start: number, end: number): AlignedText {
  return { original: a.original, text: a.text.slice(start, end), spans: a.spans.slice(2 * start, 2 * end) }
}

/**
 * Aligned texts of one original, joined, each keeping its ranges.
 *
 * @param original The original all the parts come from.
 * @param parts The parts, in order.
 * @returns Their texts and ranges run together.
 *
 * @example Two slices put back in the other order
 * const a = aligned('ab cd')
 * const b = alignedConcat(a.original, [alignedSlice(a, 3, 5), alignedSlice(a, 2, 3), alignedSlice(a, 0, 2)])
 * print(b.text, b.spans)
 */
export function alignedConcat(original: string, parts: readonly AlignedText[]): AlignedText {
  const n = parts.reduce((s, p) => s + p.text.length, 0)
  const spans = new Int32Array(2 * n)
  let at = 0
  for (const p of parts) {
    spans.set(p.spans, 2 * at)
    at += p.text.length
  }
  return { original, text: parts.map((p) => p.text).join(''), spans }
}

/**
 * The range of the original covered by code units [start, end): the hull of their ranges, ignoring zero-width
 * (inserted) units unless every unit is one. An empty range gives the position it sits at.
 *
 * @param a The aligned text.
 * @param start The first unit of `text`.
 * @param end The unit after the last one.
 * @returns The [start, end) range of `original`.
 *
 * @example Where a decoded entity came from
 * const a = alignedReplace(aligned('fish &amp; chips'), /&amp;/, '&')
 * print(a.text, '->', originalSpan(a, 5, 6))
 */
export function originalSpan(a: AlignedText, start: number, end: number): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  let zlo = Infinity
  let zhi = -Infinity
  for (let i = start; i < end; i++) {
    const s = a.spans[2 * i]
    const e = a.spans[2 * i + 1]
    if (e > s) {
      lo = Math.min(lo, s)
      hi = Math.max(hi, e)
    } else {
      zlo = Math.min(zlo, s)
      zhi = Math.max(zhi, e)
    }
  }
  if (lo <= hi) return [lo, hi]
  if (zlo <= zhi) return [zlo, zhi]
  const p = start < a.text.length ? a.spans[2 * start] : a.spans.length > 0 ? a.spans[a.spans.length - 1] : 0
  return [p, p]
}

/** Builder: append output text, all of whose units map to one range. */
class Builder {
  private chunks: string[] = []
  private ranges: number[] = []
  /** Append `s`, every unit of it mapped to the range [start, end) of the original. */
  append(s: string, start: number, end: number): void {
    if (s.length === 0) return
    this.chunks.push(s)
    for (let i = 0; i < s.length; i++) this.ranges.push(start, end)
  }
  /** Append a slice of `a` keeping its own ranges. */
  keep(a: AlignedText, start: number, end: number): void {
    if (end <= start) return
    this.chunks.push(a.text.slice(start, end))
    for (let i = 2 * start; i < 2 * end; i++) this.ranges.push(a.spans[i])
  }
  /** The aligned text built so far, over `original`. */
  done(original: string): AlignedText {
    return { original, text: this.chunks.join(''), spans: Int32Array.from(this.ranges) }
  }
}

/** The units of a text that a chunk-wise map works on. */
export type AlignmentUnit = 'codePoint' | 'grapheme'

/**
 * The chunks of a text: code points, or grapheme clusters by `Intl.Segmenter`.
 *
 * @param text The text to split.
 * @param unit The kind of chunk.
 * @returns The [start, end) code-unit range of each chunk, in order.
 */
function* chunks(text: string, unit: AlignmentUnit): Generator<[number, number]> {
  if (unit === 'grapheme') {
    for (const g of new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(text))
      yield [g.index, g.index + g.segment.length]
    return
  }
  let at = 0
  for (const c of text) {
    yield [at, at + c.length]
    at += c.length
  }
}

/**
 * Map an aligned text chunk by chunk (code points, or grapheme clusters, which Unicode normalisation never composes
 * across): each chunk is replaced by `f(chunk)`, whose units all map to the chunk's range. A chunk that maps to itself
 * keeps its finer ranges.
 *
 * @param a The aligned text.
 * @param f The map of one chunk; it may return a longer or shorter string, or the empty string to delete the chunk.
 * @param unit The chunks `f` is applied to.
 * @returns The mapped text, aligned to the same original.
 *
 * @example Case folding lengthens "ß"; both units of "ss" point at it
 * const a = alignedMap(aligned('Maß'), (c) => (c === 'ß' ? 'ss' : c.toLowerCase()))
 * print(a.text, a.spans)
 */
export function alignedMap(a: AlignedText, f: (chunk: string) => string, unit: AlignmentUnit = 'codePoint') {
  const b = new Builder()
  for (const [s, e] of chunks(a.text, unit)) {
    const chunk = a.text.slice(s, e)
    const out = f(chunk)
    if (out === chunk) b.keep(a, s, e)
    else {
      const [os, oe] = originalSpan(a, s, e)
      b.append(out, os, oe)
    }
  }
  return b.done(a.original)
}

/**
 * The replacement string of one match, with `$` patterns read as `String.prototype.replace` reads them: `$$`, `$&`,
 * `` $` ``, `$'`, `$n` and `$nn` (a group that exists; otherwise the text is kept), and `$<name>` (when the pattern
 * has named groups).
 *
 * @param replacement The replacement, with `$` patterns.
 * @param m The match.
 * @param text The whole text matched against, for `` $` `` and `$'`.
 * @returns The replacement with its patterns filled in.
 */
function substitute(replacement: string, m: RegExpExecArray, text: string): string {
  const groups = m.length - 1
  // `$<` is literal text when the pattern has no named groups.
  const re = m.groups ? /\$(\$|&|`|'|\d{1,2}|<[^>]*>)/g : /\$(\$|&|`|'|\d{1,2})/g
  return replacement.replace(re, (x, p: string) => {
    if (p === '$') return '$'
    if (p === '&') return m[0]
    if (p === '`') return text.slice(0, m.index)
    if (p === "'") return text.slice(m.index + m[0].length)
    if (p[0] === '<') return m.groups![p.slice(1, -1)] ?? ''
    // Two digits name a group when that group exists; otherwise the first digit alone does, and the second is text.
    if (p.length === 2 && Number(p) >= 1 && Number(p) <= groups) return m[Number(p)] ?? ''
    const k = Number(p[0])
    return k >= 1 && k <= groups ? (m[k] ?? '') + p.slice(1) : x
  })
}

/**
 * Replace every match of `pattern` (made global) by `replacement` (a string with `$1`-style groups, or a function of
 * the match); the units of a replacement map to the range of the text it replaced, and an empty match's replacement
 * is a zero-width insertion at its position. A replacement equal to its match keeps the match's own ranges.
 *
 * @param a The aligned text.
 * @param pattern The regular expression; a `g` flag is added when missing.
 * @param replacement A string with the `$` patterns of `String.prototype.replace` (`$&` the match, `$1` … `$99` its
 *   groups, `$<name>` a named group, `` $` `` and `$'` the text before and after it, `$$` a dollar sign), or a function
 *   of the match.
 * @returns The rewritten text, aligned to the same original.
 *
 * @example Collapse white space, then map a token back
 * const a = alignedReplace(aligned('the   cat'), /\s+/, ' ')
 * print(JSON.stringify(a.text), 'cat at', originalSpan(a, 4, 7))
 */
export function alignedReplace(
  a: AlignedText,
  pattern: RegExp,
  replacement: string | ((m: RegExpExecArray) => string),
): AlignedText {
  const re = new RegExp(pattern.source, pattern.flags.includes('g') ? pattern.flags : pattern.flags + 'g')
  const b = new Builder()
  let at = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(a.text)) !== null) {
    const s = m.index
    const e = s + m[0].length
    b.keep(a, at, s)
    const out = typeof replacement === 'string' ? substitute(replacement, m, a.text) : replacement(m)
    if (out === m[0]) b.keep(a, s, e)
    else {
      const [os, oe] = originalSpan(a, s, e)
      b.append(out, os, oe)
    }
    at = e
    if (m[0].length === 0) re.lastIndex++
  }
  b.keep(a, at, a.text.length)
  return b.done(a.original)
}

/**
 * Insert `s` before the text. Its units take the range of the first unit (as Hugging Face aligns a prepended "▁" or
 * "Ġ"), so a token made of the prefix alone still points at the start of the text. Before an empty text the prefix
 * gets the zero-width range at 0.
 *
 * @param a The aligned text.
 * @param s The prefix.
 * @returns The prefixed text, aligned to the same original.
 *
 * @example SentencePiece's word-start marker
 * const a = alignedPrepend(aligned('hello'), '▁')
 * print(a.text, originalSpan(a, 0, 1), originalSpan(a, 0, 3))
 */
export function alignedPrepend(a: AlignedText, s: string): AlignedText {
  const b = new Builder()
  if (a.text.length > 0) b.append(s, a.spans[0], a.spans[1])
  else b.append(s, 0, 0)
  b.keep(a, 0, a.text.length)
  return b.done(a.original)
}
