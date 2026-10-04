import { describe, expect, it } from 'vitest'
import { FONT_CLASSES, fontTable, fontVectors, fonts, glyphContours } from 'aifn-methods/data/real/fonts'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'

const table = fontVectors()
const ALL = 'ABCDEFGHIJLMNOPRSTUVXYZ'

describe('the vendored table', () => {
  it('holds 66 fonts × 23 capitals: 2228 samples of x and y, then 23 advance widths', () => {
    expect(table.fonts.length).toBe(66)
    expect(table.glyphs.map((g) => g.char).join('')).toBe(ALL)
    const samples = table.glyphs.reduce((a, g) => a + g.contours.reduce((p, m) => p + m, 0), 0)
    expect(samples).toBe(2228)
    expect(table.width).toBe(2 * samples + 23)
    expect(table.values.length).toBe(66 * table.width)
    expect(table.capHeight).toBe(700)
    expect(table.source).toMatch(/^github\.com\/google\/fonts@[0-9a-f]{8,}/)
    // Glyph blocks are laid out back to back, advances after them.
    let at = 0
    for (const g of table.glyphs) {
      expect(g.offset).toBe(at)
      at += 2 * g.contours.reduce((p, m) => p + m, 0)
    }
    expect(table.glyphs.map((g) => g.advance)).toEqual(table.glyphs.map((_, i) => 2 * samples + i))
    expect(fontVectors()).toBe(table)
  })

  it('every font has a class, a weight, a width and an open licence; every class is represented', () => {
    for (const f of table.fonts) {
      expect(FONT_CLASSES).toContain(f.cls)
      expect(f.weight).toBeGreaterThanOrEqual(100)
      expect(f.weight).toBeLessThanOrEqual(900)
      expect(f.width).toBeGreaterThan(0)
      expect(['OFL-1.1', 'Apache-2.0']).toContain(f.licence)
    }
    expect(new Set(table.fonts.map((f) => f.cls)).size).toBe(FONT_CLASSES.length)
    expect(new Set(table.fonts.map((f) => f.family)).size).toBe(23)
  })

  it('the top of every H is at the cap height and the letters sit on the baseline', () => {
    const H = table.glyphs.find((g) => g.char === 'H')!
    const rows = toRows(fontTable())
    expect(fontTable().shape).toEqual([66, table.width])
    for (const row of rows) {
      const ys = glyphContours(row, H).flatMap((c) => Array.from(c.y))
      expect(Math.max(...ys)).toBeCloseTo(700, -1)
      expect(Math.min(...ys)).toBeCloseTo(0, -1)
    }
  })

  it('samples are in dense correspondence: sample k of a glyph is nearer its match in another font than a neighbour is', () => {
    const rows = toRows(fontTable())
    const A = table.glyphs.find((g) => g.char === 'A')!
    const [a, b] = [glyphContours(rows[0], A)[0], glyphContours(rows[1], A)[0]]
    const m = a.x.length
    let matched = 0
    let shifted = 0
    for (let k = 0; k < m; k++) {
      const j = (k + Math.floor(m / 4)) % m
      matched += Math.hypot(a.x[k] - b.x[k], a.y[k] - b.y[k])
      shifted += Math.hypot(a.x[k] - b.x[j], a.y[k] - b.y[j])
    }
    expect(matched).toBeLessThan(shifted / 3)
  })
})

describe('fonts()', () => {
  it('rows are fonts and features the chosen outlines; y is the design class', () => {
    const d = fonts()
    expect(d.kind).toBe('dataset')
    expect(d.x.shape).toEqual([66, 2 * 2228])
    expect(d.y!.dtype).toBe('int32')
    expect(Array.from(toFlat(d.y!))).toEqual(table.fonts.map((f) => FONT_CLASSES.indexOf(f.cls)))
    expect(d.meta.featureNames!.length).toBe(2 * 2228)
    expect(d.meta.featureNames![0]).toBe('A0.0.x')
    expect(d.meta.labelNames).toEqual([...FONT_CLASSES])
    expect(d.meta.fonts).toEqual(table.fonts)
    // Row i of x is the font vector without its advances.
    expect(Array.from(toFlat(d.x)).slice(0, 2 * 2228)).toEqual(Array.from(table.values.slice(0, 2 * 2228)))
  })

  it('selects characters, rows and advances, and draws each glyph back from its row', () => {
    const d = fonts({ chars: 'PA', rows: [5, 0], advances: true })
    const P = table.glyphs.find((g) => g.char === 'P')!
    const A = table.glyphs.find((g) => g.char === 'A')!
    const sizeOf = (g: typeof P) => 2 * g.contours.reduce((p, m) => p + m, 0)
    expect(d.x.shape).toEqual([2, sizeOf(P) + sizeOf(A) + 2])
    expect(d.meta.fonts).toEqual([table.fonts[5], table.fonts[0]])
    const [layoutP, layoutA] = d.meta.glyphs
    expect([layoutP.char, layoutP.offset, layoutA.offset]).toEqual(['P', 0, sizeOf(P)])
    expect([layoutP.advance, layoutA.advance]).toEqual([sizeOf(P) + sizeOf(A), sizeOf(P) + sizeOf(A) + 1])
    const row = toRows(d.x)[0]
    const full = Array.from(table.values.slice(5 * table.width, 6 * table.width))
    expect(glyphContours(row, layoutP)).toEqual(glyphContours(full, P))
    expect(glyphContours(row, layoutA)).toEqual(glyphContours(full, A))
    expect(row[layoutA.advance]).toBe(full[A.advance])
    // P has an outer contour and a counter.
    expect(layoutP.contours.length).toBe(2)
  })

  it('rejects unknown characters and rows', () => {
    expect(() => fonts({ chars: 'K' })).toThrow(/no outlines for "K"/)
    expect(() => fonts({ rows: [66] })).toThrow(/no font 66/)
  })
})
