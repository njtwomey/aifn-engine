/**
 * Capital letters of 66 real font instances (23 families across six design classes, at several weights, widths and
 * postures) as outline vectors in dense correspondence: sample k of a glyph's contour sits at the same place of the
 * letter in every font (a stem foot, a serif tip, the apex of an A), so the vectors can be averaged, interpolated and
 * modelled, as in the manifold-of-fonts note (a GPLVM on these data; after Campbell and Kautz, 2014, "Learning a
 * Manifold of Fonts", ACM Transactions on Graphics 33(4)). The data are vendored from the AI Field
 * Notes site's font builder (provenance and licences in `glyphs.ts`).
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import { labels, matrix, type Dataset, type DatasetMeta } from '../../types'
import { ANCHOR, CAP_HEIGHT, DISPLAY, DROPPED, FONTS, GLYPHS, HANDLE, SOURCE, VECTORS, WIDTH } from './glyphs'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The six design classes, in the order of the class labels `y` = 0 … 5: `sans` (grotesque and geometric sans serif,
 * e.g. Roboto), `humanist` (humanist sans, e.g. Open Sans, Fira Sans), `serif` (old-style and transitional serif),
 * `didone` (high-contrast modern serif, e.g. Bodoni Moda), `slab` (slab serif, e.g. Roboto Slab) and `mono`
 * (monospaced).
 */
export const FONT_CLASSES = ['sans', 'humanist', 'serif', 'didone', 'slab', 'mono'] as const
export type FontClass = (typeof FONT_CLASSES)[number]

/** One font instance: a family at one weight, width and posture. */
export interface FontInfo {
  readonly family: string
  /** Style label, e.g. "Bold Italic". */
  readonly style: string
  readonly cls: FontClass
  /** CSS weight, 100 to 900. */
  readonly weight: number
  /** Width as a percentage of normal. */
  readonly width: number
  readonly italic: boolean
  /** SPDX identifier of the font's licence (OFL-1.1 or Apache-2.0). */
  readonly licence: string
}

/** Where one character sits in a font vector. */
export interface GlyphLayout {
  readonly char: string
  /** Samples per contour: the outer contour first, then its counters (holes). */
  readonly contours: readonly number[]
  /** Index of the first value (x of sample 0 of contour 0); x and y alternate. */
  readonly offset: number
  /** Index of the advance width, or −1 when the vector holds none. */
  readonly advance: number
}

/** The whole vendored table: every font's full vector, decoded. */
export interface FontVectors {
  readonly fonts: readonly FontInfo[]
  readonly glyphs: readonly GlyphLayout[]
  /** Row-major int16 [fonts, width]: every glyph's x, y samples, then one advance width per glyph. */
  readonly values: Int16Array
  readonly width: number
  /** Font units: every font is scaled so that the top of its H is at this height. */
  readonly capHeight: number
  /** The word the note's figure sets ("GPLVM"). */
  readonly display: string
  /** Sample indices on the display word's P used by the note's drag interaction (the foot-serif tip, a fixed point). */
  readonly handle: number
  readonly anchor: number
  /** The pinned google/fonts commit. */
  readonly source: string
  /** Fonts the builder dropped because a character's contour count differed from the majority. */
  readonly dropped: readonly { font: string; reason: string }[]
}

let cache: FontVectors | null = null

/** The vendored table, decoded once. */
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
  readonly fonts: readonly FontInfo[]
  /** Layouts with offsets into the columns of `x` (advance −1 without `advances`). */
  readonly glyphs: readonly GlyphLayout[]
  readonly capHeight: number
}

/** A font dataset: one row per font. */
export interface FontDataset extends Dataset {
  readonly meta: FontDatasetMeta
}

/**
 * Fonts as a dataset: rows are fonts, features the flattened outline samples of the chosen characters (x, y of every
 * sample of every contour, in font units with the cap height at 700), and `y` the design class (int32, 0 … 5: sans,
 * humanist, serif, didone, slab, mono; see `FONT_CLASSES`). The fonts' family, class, weight, width, posture and
 * licence are in `meta.fonts`.
 *
 * Only capital letters are available, and not all of them: the 23 capitals "ABCDEFGHIJLMNOPRSTUVXYZ" (K, Q and W are left
 * out because their contour counts or shapes vary between designs: a split K, a free Q tail, a crossed W). There are no lowercase letters, digits or
 * punctuation; asking for one throws a `DomainError` naming the characters available.
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
  readonly x: Float64Array
  readonly y: Float64Array
}

/**
 * The contours of one glyph in a font vector (a row of a font dataset's `x`, or a GPLVM's predicted outline), for
 * drawing: the outer contour first, then the counters.
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

/** Font-vector rows as a float64 tensor, for callers that want the whole table ([fonts, width], advances included). */
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
