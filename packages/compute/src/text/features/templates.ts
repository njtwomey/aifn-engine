/**
 * Feature templates in the syntax of CRF++ (Kudo 2005, "CRF++: Yet another CRF toolkit", the "Preparing feature
 * templates" section), the classic way to declare the features of a sequence labeller.
 *
 * The input is a table: one row per token (position $n$), one column per attribute (the token itself, a part of
 * speech, a character class, ...). A template line is literal text with macros `%x[r,c]`, each the attribute in column
 * $c$ of the row $r$ positions away from the current one. Expanding a template at position $n$ replaces every macro by
 * that cell and gives one feature string, e.g. `U01:%x[-1,0]` at the second token of "the cat" gives `U01:the`. Rows
 * before the start read as `_B-1`, `_B-2`, … and rows after the end as `_B+1`, `_B+2`, …
 *
 * - A line starting with `U` is a **unigram** template: its string is conjoined with the current label $y_n$, so it
 *   stands for $K$ feature functions $f(y_n, \xvec, n) = [\text{string at } n = s][y_n = k]$.
 * - A line starting with `B` is a **bigram** template: conjoined with the label pair $(y_{n-1}, y_n)$, $K^2$
 *   functions. A bare `B` expands to the same string everywhere, so it is the transition matrix; `B01:%x[0,0]` makes
 *   the transitions depend on the current token. Bigram strings are used for $n \ge 1$, where there is an edge.
 * - Empty lines and lines starting with `#` are skipped (as are lines starting with a space, as in CRF++).
 *
 * Expanded strings are collected into a `FeatureIndex` (string to id, with counts) over training data; strings seen
 * fewer than `minFrequency` times are dropped (CRF++'s `-f`). Unknown strings at prediction time fire nothing.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The largest $\lvert r \rvert$ CRF++ accepts in `%x[r,c]`. */
export const MAX_TEMPLATE_OFFSET = 8

/**
 * A malformed template: `line` and `column` are 1-based positions in the template source. Thrown by `parseTemplates`,
 * whose name it carries as the operation.
 *
 * @param message What is wrong, without the position (the position is prefixed to the error's message).
 * @param line The 1-based line of the source.
 * @param column The 1-based column of the line.
 *
 * @example Catch a malformed macro and read its position
 * try {
 *   parseTemplates('U00:%x[0,0]\nU01:%x[0]')
 * } catch (e) {
 *   print(e instanceof TemplateSyntaxError, e.line, e.column)
 *   print(e.message)
 * }
 */
export class TemplateSyntaxError extends DomainError {
  /** The 1-based line of the source. */
  readonly line: number
  /** The 1-based column of the line. */
  readonly column: number

  constructor(message: string, line: number, column: number) {
    super('parseTemplates', `parseTemplates: line ${line}, column ${column}: ${message}`)
    this.name = 'TemplateSyntaxError'
    this.line = line
    this.column = column
  }
}

/** One `%x[row,column]` macro: the relative row $r$, the column $c$, and its character span in the template text. */
export interface TemplateMacro {
  /** The row offset $r$ from the current position (negative for earlier rows). */
  readonly row: number
  /** The column $c$, from 0. */
  readonly column: number
  /** 0-based start in `text`. */
  readonly start: Size
  /** 0-based end in `text`, exclusive. */
  readonly end: Size
}

/** One parsed template line. */
export interface FeatureTemplate {
  /** `unigram` for a `U` line, `bigram` for a `B` line. */
  readonly kind: 'unigram' | 'bigram'
  /** The line as written (trimmed), e.g. `U05:%x[-1,0]/%x[0,0]`. */
  readonly text: string
  /** The identifier: the text before the first `:` (`U05`), or the whole line when it has none (`B`). */
  readonly id: string
  /** The macros, in order. */
  readonly macros: readonly TemplateMacro[]
  /** The literal text between the macros: `pieces.length = macros.length + 1`. */
  readonly pieces: readonly string[]
  /** 1-based line in the source. */
  readonly line: number
}

/** A parsed template file. */
export interface FeatureTemplates {
  /** The tag `'feature-templates'`. */
  readonly kind: 'feature-templates'
  /** The template lines, in source order. */
  readonly templates: readonly FeatureTemplate[]
  /** The number of input columns the templates read: one more than the largest $c$. */
  readonly columns: Size
  /** The largest $\lvert r \rvert$ any macro reads. */
  readonly reach: Size
}

/** The input of a sequence: rows (one per token) of string columns. */
export type TokenRows = readonly (readonly string[])[]

/**
 * Parse a CRF++ template file (see the file comment). Throws `TemplateSyntaxError`, with the line and column, on an
 * unknown line type (not `U` or `B`), a malformed macro (`%x[r,c]` with integer $r$ and non-negative integer $c$),
 * $\lvert r \rvert$ above `MAX_TEMPLATE_OFFSET`, or a `%` followed by anything but `x[`. A file with no templates is an
 * error too.
 *
 * @param source The template file, one template per line (`\n` or `\r\n` line ends).
 * @returns The templates, with the number of columns they read and their reach.
 *
 * @example Three unigram templates and the transition template
 * const templates = parseTemplates('U00:%x[-1,0]\nU01:%x[0,0]\nU02:%x[0,0]/%x[0,1]\nB')
 * for (const t of templates.templates) print(t.id, t.kind, t.macros.map((m) => [m.row, m.column]), t.pieces)
 * print('columns', templates.columns, ' reach', templates.reach)
 */
export function parseTemplates(source: string): FeatureTemplates {
  const templates: FeatureTemplate[] = []
  const lines = source.split(/\r?\n/)
  lines.forEach((raw, i) => {
    const line = i + 1
    if (raw.length === 0 || raw[0] === '#' || raw[0] === ' ' || raw[0] === '\t') return
    const text = raw.trimEnd()
    if (text.length === 0) return
    const head = text[0]
    if (head !== 'U' && head !== 'B')
      throw new TemplateSyntaxError(
        `unknown template type '${head}': a line starts with U (unigram) or B (bigram)`,
        line,
        1,
      )
    const macros: TemplateMacro[] = []
    const pieces: string[] = []
    let literal = ''
    let k = 0
    while (k < text.length) {
      const ch = text[k]
      if (ch !== '%') {
        literal += ch
        k++
        continue
      }
      const m = /^%x\[(-?\d+),(\d+)\]/.exec(text.slice(k))
      if (!m) {
        const col = k + 1
        if (text[k + 1] !== 'x') throw new TemplateSyntaxError(`'%' must be followed by x[row,column]`, line, col)
        if (text[k + 2] !== '[') throw new TemplateSyntaxError(`expected '[' after %x`, line, k + 3)
        const close = text.indexOf(']', k)
        throw new TemplateSyntaxError(
          close < 0
            ? `unterminated macro: expected ']'`
            : `malformed macro '${text.slice(k, close + 1)}': expected %x[row,column] with integers, column ≥ 0`,
          line,
          col,
        )
      }
      const row = Number(m[1])
      const column = Number(m[2])
      if (Math.abs(row) > MAX_TEMPLATE_OFFSET)
        throw new TemplateSyntaxError(`row offset ${row} is beyond ±${MAX_TEMPLATE_OFFSET}`, line, k + 4)
      pieces.push(literal)
      literal = ''
      macros.push({ row, column, start: k, end: k + m[0].length })
      k += m[0].length
    }
    pieces.push(literal)
    const colon = text.indexOf(':')
    templates.push({
      kind: head === 'U' ? 'unigram' : 'bigram',
      text,
      id: colon < 0 ? text : text.slice(0, colon),
      macros,
      pieces,
      line,
    })
  })
  if (templates.length === 0) throw new TemplateSyntaxError('no templates: add a U or B line', 1, 1)
  const all = templates.flatMap((t) => t.macros)
  return {
    kind: 'feature-templates',
    templates,
    columns: all.reduce((a, m) => Math.max(a, m.column + 1), 0),
    reach: all.reduce((a, m) => Math.max(a, Math.abs(m.row)), 0),
  }
}

/**
 * The cell `%x[r,c]` read at position $n$: the attribute, or `_B-k` / `_B+k` outside the sequence. Throws `DomainError`
 * when the row has no column $c$.
 *
 * @param rows The sequence, one row of attributes per token.
 * @param n The current position.
 * @param row The row offset $r$.
 * @param column The column $c$.
 * @returns The attribute in row $n + r$, column $c$, or the boundary marker.
 *
 * @example Inside and outside the sequence
 * const rows = [['the', 'DT'], ['cat', 'NN'], ['sat', 'VBD']]
 * print('%x[-1,0] at 1:', templateCell(rows, 1, -1, 0), ' %x[0,1] at 1:', templateCell(rows, 1, 0, 1))
 * print('%x[-2,0] at 0:', templateCell(rows, 0, -2, 0), ' %x[1,0] at 2:', templateCell(rows, 2, 1, 0))
 */
export function templateCell(rows: TokenRows, n: Size, row: number, column: Size): string {
  const at = n + row
  if (at < 0) return `_B${at}`
  if (at >= rows.length) return `_B+${at - rows.length + 1}`
  const cell = rows[at][column]
  if (cell === undefined)
    throw new DomainError(
      'expandTemplate',
      `expandTemplate: the template reads column ${column}, but row ${at} has ${rows[at].length} columns`,
    )
  return cell
}

/**
 * The feature string of one template at position $n$: its literal text with every macro replaced by its cell.
 *
 * @param template One parsed template.
 * @param rows The sequence, one row of attributes per token.
 * @param n The position.
 * @returns The feature string.
 *
 * @example `U01:%x[-1,0]` at the second token of "the cat"
 * const [t] = parseTemplates('U01:%x[-1,0]').templates
 * print(expandTemplate(t, [['the'], ['cat']], 1))
 */
export function expandTemplate(template: FeatureTemplate, rows: TokenRows, n: Size): string {
  let out = template.pieces[0]
  template.macros.forEach((m, k) => {
    out += templateCell(rows, n, m.row, m.column) + template.pieces[k + 1]
  })
  return out
}

/**
 * The expanded strings at every position: unigram strings at every $n$, bigram strings at $n \ge 1$ (none at $n = 0$).
 */
export interface ExpandedSequence {
  /** Entry $n$: the strings of the unigram templates at position $n$, in template order. */
  readonly unigram: readonly (readonly string[])[]
  /** Entry $n$: the strings of the bigram templates at position $n$, in template order; empty at $n = 0$. */
  readonly bigram: readonly (readonly string[])[]
}

/**
 * Expand every template at every position of a sequence.
 *
 * @param templates The parsed templates.
 * @param rows The sequence, one row of attributes per token.
 * @returns The unigram and bigram strings at each position.
 *
 * @example The features of "the cat sat"
 * const templates = parseTemplates('U00:%x[-1,0]\nU01:%x[0,0]\nU02:%x[0,0]/%x[0,1]\nB')
 * const rows = [['the', 'DT'], ['cat', 'NN'], ['sat', 'VBD']]
 * const e = expandTemplates(templates, rows)
 * e.unigram.forEach((fs, n) => print(n, fs, e.bigram[n]))
 */
export function expandTemplates(templates: FeatureTemplates, rows: TokenRows): ExpandedSequence {
  const uni = templates.templates.filter((t) => t.kind === 'unigram')
  const bi = templates.templates.filter((t) => t.kind === 'bigram')
  return {
    unigram: rows.map((_, n) => uni.map((t) => expandTemplate(t, rows, n))),
    bigram: rows.map((_, n) => (n === 0 ? [] : bi.map((t) => expandTemplate(t, rows, n)))),
  }
}

/**
 * The feature strings of training data, each with an id and a count. Unigram and bigram strings have their own id
 * spaces (from 0 to `unigram.length - 1` and from 0 to `bigram.length - 1`), in order of first appearance.
 */
export interface FeatureIndex {
  /** The tag `'feature-index'`. */
  readonly kind: 'feature-index'
  /** The templates the strings were expanded from. */
  readonly templates: FeatureTemplates
  /** The kept unigram strings, by id. */
  readonly unigram: readonly string[]
  /** The kept bigram strings, by id. */
  readonly bigram: readonly string[]
  /** How often each unigram string occurred in the training data (positions, over all sequences). */
  readonly unigramCounts: Int32Array
  /** How often each bigram string occurred in the training data. */
  readonly bigramCounts: Int32Array
  /** Strings dropped by `minFrequency` (unigram, bigram). */
  readonly dropped: { readonly unigram: Size; readonly bigram: Size }
  /** The least count a string needed to be kept. */
  readonly minFrequency: Size
  /** Unigram string to id. */
  readonly unigramIds: ReadonlyMap<string, number>
  /** Bigram string to id. */
  readonly bigramIds: ReadonlyMap<string, number>
}

/** Options of `featureIndex`. */
export interface FeatureIndexOptions {
  /** Keep strings seen at least this many times (CRF++'s `-f`, default 1). */
  minFrequency?: Size
}

/**
 * Count the expanded strings of training sequences and keep those seen at least `minFrequency` times.
 *
 * @param templates The parsed templates.
 * @param sequences The training sequences, each one row of attributes per token.
 * @param options The frequency cut-off; see {@link FeatureIndexOptions}.
 * @returns The kept strings with their ids and counts.
 *
 * @example Index two short sentences, keeping strings seen twice
 * const templates = parseTemplates('U00:%x[-1,0]\nU01:%x[0,0]\nU02:%x[0,0]/%x[0,1]\nB')
 * const data = [[['the', 'DT'], ['cat', 'NN']], [['the', 'DT'], ['dog', 'NN']]]
 * const index = featureIndex(templates, data, { minFrequency: 2 })
 * print('unigram', index.unigram, index.unigramCounts)
 * print('bigram ', index.bigram, index.bigramCounts)
 * print('dropped', index.dropped)
 */
export function featureIndex(
  templates: FeatureTemplates,
  sequences: readonly TokenRows[],
  options: FeatureIndexOptions = {},
): FeatureIndex {
  const { minFrequency = 1 } = options
  const uni = new Map<string, number>()
  const bi = new Map<string, number>()
  for (const rows of sequences) {
    const e = expandTemplates(templates, rows)
    for (const fs of e.unigram) for (const f of fs) uni.set(f, (uni.get(f) ?? 0) + 1)
    for (const fs of e.bigram) for (const f of fs) bi.set(f, (bi.get(f) ?? 0) + 1)
  }
  const keep = (m: Map<string, number>) => [...m].filter(([, c]) => c >= minFrequency)
  const u = keep(uni)
  const b = keep(bi)
  return {
    kind: 'feature-index',
    templates,
    unigram: u.map(([s]) => s),
    bigram: b.map(([s]) => s),
    unigramCounts: Int32Array.from(u.map(([, c]) => c)),
    bigramCounts: Int32Array.from(b.map(([, c]) => c)),
    dropped: { unigram: uni.size - u.length, bigram: bi.size - b.length },
    minFrequency,
    unigramIds: new Map(u.map(([s], i) => [s, i])),
    bigramIds: new Map(b.map(([s], i) => [s, i])),
  }
}

/**
 * A sequence's features as ids, in compressed rows: the unigram ids at position n are
 * `unigramIds[unigramStart[n] … unigramStart[n + 1])`, likewise for bigrams (empty at n = 0). Strings not in the index
 * are left out.
 */
export interface EncodedSequence {
  /** The number of positions $N$. */
  readonly length: Size
  /** The unigram ids of all positions, run together. */
  readonly unigramIds: Int32Array
  /** Where each position's unigram ids start in `unigramIds` ($N+1$ entries, the last its length). */
  readonly unigramStart: Int32Array
  /** The bigram ids of all positions, run together. */
  readonly bigramIds: Int32Array
  /** Where each position's bigram ids start in `bigramIds` ($N+1$ entries). */
  readonly bigramStart: Int32Array
}

/**
 * Encode a sequence with an index (see `EncodedSequence`). Strings the index does not hold fire nothing.
 *
 * @param index The feature index built from training data.
 * @param rows The sequence, one row of attributes per token.
 * @returns The ids at each position, in compressed rows.
 *
 * @example A new sentence: unseen words fire only the strings the index knows
 * const templates = parseTemplates('U00:%x[-1,0]\nU01:%x[0,0]\nU02:%x[0,0]/%x[0,1]\nB')
 * const index = featureIndex(templates, [[['the', 'DT'], ['cat', 'NN']], [['a', 'DT'], ['dog', 'NN']]])
 * const enc = encodeTemplateRows(index, [['the', 'DT'], ['dog', 'NN'], ['ran', 'VBD']])
 * print('unigram ids', enc.unigramIds, ' starts', enc.unigramStart)
 * print('bigram ids ', enc.bigramIds, ' starts', enc.bigramStart)
 */
export function encodeTemplateRows(index: FeatureIndex, rows: TokenRows): EncodedSequence {
  const e = expandTemplates(index.templates, rows)
  const pack = (all: readonly (readonly string[])[], ids: ReadonlyMap<string, number>) => {
    const out: number[] = []
    const start = new Int32Array(all.length + 1)
    all.forEach((fs, n) => {
      for (const f of fs) {
        const id = ids.get(f)
        if (id !== undefined) out.push(id)
      }
      start[n + 1] = out.length
    })
    return { ids: Int32Array.from(out), start }
  }
  const u = pack(e.unigram, index.unigramIds)
  const b = pack(e.bigram, index.bigramIds)
  return { length: rows.length, unigramIds: u.ids, unigramStart: u.start, bigramIds: b.ids, bigramStart: b.start }
}
