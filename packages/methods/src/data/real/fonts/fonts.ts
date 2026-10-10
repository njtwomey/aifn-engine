/**
 * Capital letters of 66 real font instances (23 families across six design classes, at several weights, widths and
 * postures) as outline vectors in dense correspondence: sample $k$ of a glyph's contour sits at the same place of the
 * letter in every font (a stem foot, a serif tip, the apex of an A), so the vectors can be averaged, interpolated and
 * modelled, as in the manifold-of-fonts note (a GPLVM on these data; after Campbell and Kautz, 2014, "Learning a
 * Manifold of Fonts", ACM Transactions on Graphics 33(4)). The data are vendored from the AI Field
 * Notes site's font builder (provenance and licences in `glyphs.ts`). Every font comes from Google Fonts under the SIL
 * Open Font License 1.1 or the Apache License 2.0; each font's licence is in its `FontInfo`.
 *
 * A font vector holds, for each character in turn, the $x$ and $y$ of every sample of every contour (outer contour
 * first, then the counters), in font units with the cap height at 700, and then one advance width per character.
 * `GlyphLayout` says where each character sits in it.
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import { labels, matrix, type Dataset, type DatasetMeta } from '../../types'
import { ANCHOR, CAP_HEIGHT, DISPLAY, DROPPED, FONTS, GLYPHS, HANDLE, SOURCE, VECTORS, WIDTH } from './glyphs'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The six design classes, in the order of the class labels $0, \dots, 5$ of a font dataset's `y`: `sans` (grotesque and
 * geometric sans serif, e.g. Roboto), `humanist` (humanist sans, e.g. Open Sans, Fira Sans), `serif` (old-style and
 * transitional serif), `didone` (high-contrast modern serif, e.g. Bodoni Moda), `slab` (slab serif, e.g. Roboto Slab)
 * and `mono` (monospaced).
 */
export const FONT_CLASSES = ['sans', 'humanist', 'serif', 'didone', 'slab', 'mono'] as const
/** A design class: one of `FONT_CLASSES`. */
export type FontClass = (typeof FONT_CLASSES)[number]

/** One font instance: a family at one weight, width and posture. */
export interface FontInfo {
  /** Family name, e.g. "Roboto". */
  readonly family: string
  /** Style label, e.g. "Bold Italic". */
  readonly style: string
  /** Design class. */
  readonly cls: FontClass
  /** CSS weight, 100 to 900. */
  readonly weight: number
  /** Width as a percentage of normal. */
  readonly width: number
  /** Whether the instance is italic (or oblique). */
  readonly italic: boolean
  /** SPDX identifier of the font's licence (OFL-1.1 or Apache-2.0). */
  readonly licence: string
}

/** Where one character sits in a font vector. */
export interface GlyphLayout {
  /** The character, a capital letter. */
  readonly char: string
  /** Samples per contour: the outer contour first, then its counters (holes). */
  readonly contours: readonly number[]
  /** Index of the first value (x of sample 0 of contour 0); x and y alternate. */
  readonly offset: number
  /** Index of the advance width, or $-1$ when the vector holds none. */
  readonly advance: number
}

/** The whole vendored table: every font's full vector, decoded. */
export interface FontVectors {
  /** The 66 font instances, one per row of `values`. */
  readonly fonts: readonly FontInfo[]
  /** Where each of the 23 characters sits in a row, advance width included. */
  readonly glyphs: readonly GlyphLayout[]
  /** Row-major int16 [fonts, width]: every glyph's x, y samples, then one advance width per glyph. */
  readonly values: Int16Array
  /** Values per row: every glyph's samples, then the advance widths. */
  readonly width: number
  /** Font units: every font is scaled so that the top of its H is at this height. */
  readonly capHeight: number
  /** The word the note's figure sets ("GPLVM"). */
  readonly display: string
  /** Sample index on the display word's P (outer contour) of the foot-serif tip, dragged in the note's figure. */
  readonly handle: number
  /** Sample index on the display word's P (outer contour) of a point held fixed while `handle` moves. */
  readonly anchor: number
  /** The pinned google/fonts commit. */
  readonly source: string
  /** Fonts the builder dropped because a character's contour count differed from the majority. */
  readonly dropped: readonly { font: string; reason: string }[]
}

/** The decoded table, filled on first use by `fontVectors`. */
let cache: FontVectors | null = null

/**
 * The vendored table, decoded once from base64 into int16 values; later calls return the same object.
 *
 * @returns Every font's full vector (66 rows of `width` values, row-major), the fonts, where each glyph sits, and the
 *   provenance.
 *
 * @example The table and one font
 * const t = fontVectors()
 * print('fonts:', t.fonts.length, ' values per font:', t.width, ' cap height:', t.capHeight)
 * print('characters:', t.glyphs.map((g) => g.char).join(''))
 * print('first font:', t.fonts[0])
 * print('its first three samples of A (x, y):', t.values.slice(0, 6))
 */
export function fontVectors(): FontVectors {
  if (cache) return cache
  const binary = atob(VECTORS)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const view = new DataView(bytes.buffer)
  const values = new Int16Array(bytes.length / 2)
  for (let i = 0; i < values.length; i++) values[i] = view.getInt16(2 * i, true)
  cache = {
    fonts: FONTS as readonly FontInfo[],
    glyphs: GLYPHS,
    values,
    width: WIDTH,
    capHeight: CAP_HEIGHT,
    display: DISPLAY,
    handle: HANDLE,
    anchor: ANCHOR,
    source: SOURCE,
    dropped: DROPPED,
  }
  return cache
}

/** Options for `fonts`. */
export interface FontsOptions {
  /** The characters whose outlines form the features, in this order (default every one: "ABCDEFGHIJLMNOPRSTUVXYZ"). */
  chars?: string
  /** Append each character's advance width (default false). */
  advances?: boolean
  /** Rows (font indices) to keep, in this order (default all 66). */
  rows?: readonly number[]
}

/** Metadata of a font dataset: the fonts of its rows and where each character sits in `x`. */
export interface FontDatasetMeta extends DatasetMeta {
  /** The font of each row of `x`. */
  readonly fonts: readonly FontInfo[]
  /** Layouts with offsets into the columns of `x` (advance $-1$ without `advances`). */
  readonly glyphs: readonly GlyphLayout[]
  /** The cap height in font units (700). */
  readonly capHeight: number
}

/** A font dataset: one row per font. */
export interface FontDataset extends Dataset {
  /** The dataset's metadata, with the fonts of the rows and the glyph layouts. */
  readonly meta: FontDatasetMeta
}

/**
 * Fonts as a dataset: rows are fonts, features the flattened outline samples of the chosen characters ($x$, $y$ of
 * every sample of every contour, in font units with the cap height at 700), and `y` the design class (int32,
 * $0, \dots, 5$: sans, humanist, serif, didone, slab, mono; see `FONT_CLASSES`). The fonts' family, class, weight,
 * width, posture and licence are in `meta.fonts`.
 *
 * Only capital letters are available, and not all of them: the 23 capitals "ABCDEFGHIJLMNOPRSTUVXYZ" (K, Q and W are
 * left out because their contour counts or shapes vary between designs: a split K, a free Q tail, a crossed W). There
 * are no lowercase letters, digits or punctuation; asking for one throws a `DomainError` naming the characters
 * available, as does a row that is not a font index. Fonts from Google Fonts (SIL OFL 1.1 or Apache 2.0).
 *
 * @param options The characters to take, whether to append their advance widths, and which fonts to keep.
 * @returns The dataset: `x` (fonts $\times$ values), `y` the design classes, and `meta` with the column names
 *   (`A0.3.x` is $x$ of sample 3 of contour 0 of A; `A.advance` an advance width), the fonts and the glyph layouts.
 *
 * @example Shape, classes and the first values
 * const d = fonts({ chars: 'HO' })
 * print('x:', d.x.shape, ' first columns:', d.meta.featureNames.slice(0, 4))
 * print('first row starts:', toArray(d.x)[0].slice(0, 4))
 * print('first fonts:', d.meta.fonts.slice(0, 3).map((f) => `${f.family} ${f.style}`))
 * const y = toArray(d.y)
 * print('fonts per class:', d.meta.labelNames.map((c, k) => `${c} ${y.filter((v) => v === k).length}`))
 *
 * @example A letter that is not available throws
 * try {
 *   fonts({ chars: 'K' })
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function fonts(options: FontsOptions = {}): FontDataset {
  const table = fontVectors()
  const byChar = new Map(table.glyphs.map((g) => [g.char, g]))
  const chars = [...(options.chars ?? table.glyphs.map((g) => g.char).join(''))]
  const rows = options.rows ?? table.fonts.map((_, i) => i)
  const advances = options.advances ?? false
  const layout: GlyphLayout[] = []
  const columns: number[] = []
  const names: string[] = []
  for (const c of chars) {
    const g = byChar.get(c)
    if (!g) throw new DomainError('fonts', `fonts: no outlines for "${c}"; available: ${[...byChar.keys()].join('')}`)
    const offset = columns.length
    const samples = g.contours.reduce((a, b) => a + b, 0)
    for (let k = 0; k < 2 * samples; k++) columns.push(g.offset + k)
    g.contours.forEach((m, contour) => {
      for (let j = 0; j < m; j++) names.push(`${c}${contour}.${j}.x`, `${c}${contour}.${j}.y`)
    })
    layout.push({ char: c, contours: g.contours, offset, advance: -1 })
  }
  if (advances)
    chars.forEach((c, i) => {
      layout[i] = { ...layout[i], advance: columns.length }
      columns.push(byChar.get(c)!.advance)
      names.push(`${c}.advance`)
    })
  const d = columns.length
  const x = new Float64Array(rows.length * d)
  rows.forEach((r, i) => {
    if (!(Number.isInteger(r) && r >= 0 && r < table.fonts.length))
      throw new DomainError('fonts', `fonts: no font ${r}`)
    for (let j = 0; j < d; j++) x[i * d + j] = table.values[r * table.width + columns[j]]
  })
  const kept = rows.map((r) => table.fonts[r])
  return {
    kind: 'dataset',
    x: matrix(x, rows.length, d),
    y: labels(kept.map((f) => FONT_CLASSES.indexOf(f.cls))),
    meta: {
      name: 'fonts',
      description: `Outlines of ${chars.join('')} in ${rows.length} font instances, sampled in dense correspondence (${d} values per font).`,
      task: 'embedding',
      featureNames: names,
      labelNames: [...FONT_CLASSES],
      source:
        'Google Fonts (github.com/google/fonts, SIL OFL 1.1 and Apache 2.0), outlines put in correspondence by the AI Field Notes font builder; after Campbell and Kautz (2014), "Learning a Manifold of Fonts"',
      url: `https://${table.source.replace('@', '/tree/')}`,
      fonts: kept,
      glyphs: layout,
      capHeight: table.capHeight,
    },
  }
}

/** One contour's samples: x and y coordinates in font units. */
export interface Contour {
  /** The $x$ coordinate of each sample, in contour order. */
  readonly x: Float64Array
  /** The $y$ coordinate of each sample. */
  readonly y: Float64Array
}

/**
 * The contours of one glyph in a font vector (a row of a font dataset's `x`, or a GPLVM's predicted outline), for
 * drawing: the outer contour first, then the counters.
 *
 * @param row The font vector; read from `glyph.offset` on, with $x$ and $y$ alternating. Not modified.
 * @param glyph Where the glyph sits in `row`: from `meta.glyphs` of a font dataset for one of its rows, or from
 *   `fontVectors().glyphs` for a full vector.
 * @returns One contour per entry of `glyph.contours`, each with its samples' $x$ and $y$ in font units.
 *
 * @example The two contours of an O
 * const d = fonts({ chars: 'O' })
 * const [outer, counter] = glyphContours(toArray(d.x)[0], d.meta.glyphs[0])
 * print('samples:', outer.x.length, 'outer,', counter.x.length, 'counter')
 * const height = (c) => Math.max(...c.y) - Math.min(...c.y)
 * print('outer height:', height(outer), ' counter height:', height(counter))
 */
export function glyphContours(row: ArrayLike<number>, glyph: GlyphLayout): Contour[] {
  let at = glyph.offset
  return glyph.contours.map((m) => {
    const x = new Float64Array(m)
    const y = new Float64Array(m)
    for (let j = 0; j < m; j++) {
      x[j] = row[at + 2 * j]
      y[j] = row[at + 2 * j + 1]
    }
    at += 2 * m
    return { x, y }
  })
}

/**
 * Font-vector rows as a float64 tensor, for callers that want the whole table ([fonts, width], advances included).
 *
 * @returns A fresh $66 \times 4479$ float64 tensor of the values of `fontVectors`.
 *
 * @example The whole table
 * const t = fontTable()
 * print('shape:', t.shape)
 * print('first values of the first font:', toArray(t)[0].slice(0, 4))
 */
export function fontTable() {
  const t = fontVectors()
  return fromData(Float64Array.from(t.values), [t.fonts.length, t.width])
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/real/fonts')

dataset(
  {
    key: 'fonts',
    name: 'Fonts',
    summary: 'Capital-letter outlines of 66 fonts in dense correspondence, one row per font.',
    task: 'embedding',
    output: 'dataset',
    knobs: space({ advances: bool() }),
    truth: false,
    random: false,
    notes: ['manifold-of-fonts'],
  },
  fonts,
)
