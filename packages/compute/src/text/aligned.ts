/**
 * Aligned text: a string that remembers where each of its code units came from in an original text, so normalisation,
 * replacement and splitting can change the text while tokens keep offsets into what the user typed (as the
 * `NormalizedString` of Hugging Face `tokenizers`). Every UTF-16 code unit of `text` carries a [start, end) range of
 * `original`; inserted characters carry a zero-width range at the position they were inserted.
 */

/** Text with, per UTF-16 code unit, the [start, end) range of the original it came from (int32, two per unit). */
export interface AlignedText {
  readonly original: string
  readonly text: string
  readonly spans: Int32Array
}

/** The identity alignment: `text` is `original`, each code unit its own range. */
export function aligned(original: string): AlignedText {
  const spans = new Int32Array(2 * original.length)
  for (let i = 0; i < original.length; i++) {
    spans[2 * i] = i
    spans[2 * i + 1] = i + 1
  }
  return { original, text: original, spans }
}

/** The code units [start, end) of an aligned text, keeping their ranges. */
export function alignedSlice(a: AlignedText, start: number, end: number): AlignedText {
  return { original: a.original, text: a.text.slice(start, end), spans: a.spans.slice(2 * start, 2 * end) }
}

/** Aligned texts of one original, joined. */
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
  done(original: string): AlignedText {
    return { original, text: this.chunks.join(''), spans: Int32Array.from(this.ranges) }
  }
}

/** The units of a text that a chunk-wise map works on. */
export type AlignmentUnit = 'codePoint' | 'grapheme'

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
 * Replace every match of `pattern` (made global) by `replacement` (a string with `$1`-style groups, or a function of
 * the match); the units of a replacement map to the range of the text it replaced, and an empty match's replacement
 * is a zero-width insertion at its position.
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
    const out =
      typeof replacement === 'string'
        ? replacement.replace(/\$(\d)|\$&/g, (_x, g: string | undefined) => (g ? (m![Number(g)] ?? '') : m![0]))
        : replacement(m)
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
 * "Ġ"), so a token made of the prefix alone still points at the start of the text.
 */
export function alignedPrepend(a: AlignedText, s: string): AlignedText {
  const b = new Builder()
  if (a.text.length > 0) b.append(s, a.spans[0], a.spans[1])
  else b.append(s, 0, 0)
  b.keep(a, 0, a.text.length)
  return b.done(a.original)
}
