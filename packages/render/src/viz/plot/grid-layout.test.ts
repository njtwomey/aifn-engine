import { describe, expect, it } from 'vitest'
import { equalRowWidths, layoutColumn, layoutEqualColumn, type ColumnInput } from './grid-layout'

/** Pixels per data unit on x and y of the equal-aspect panel, from the ECharts grid's pixel extents. */
function unitsOf(input: ColumnInput) {
  const l = layoutColumn(input)
  const e = input.equal.row
  const gridWidth = l.width - input.margins.left - input.margins.right
  const gridHeight = l.heights[e] - input.chrome[e] - input.margins.top[e] - input.margins.bottom[e]
  return { l, x: gridWidth / input.equal.xSpan, y: gridHeight / input.equal.ySpan }
}

const base: ColumnInput = {
  width: 960,
  frameHeight: 480,
  cap: 900,
  ratios: [2, 1],
  equal: { row: 0, xSpan: 12, ySpan: 4 },
  margins: { left: 66, right: 20, top: [32, 32], bottom: [14, 44] },
  chrome: [28, 28],
  gap: 8,
  minHeight: 160,
}

describe('layoutColumn', () => {
  it('gives equal pixels per unit when the column fits', () => {
    const { l, x, y } = unitsOf(base)
    expect(Math.abs(x / y - 1)).toBeLessThan(0.01)
    expect(l.width).toBe(960)
  })

  it('narrows the column to the cap, keeping equal units, when the panel would be too tall', () => {
    const { l, x, y } = unitsOf({ ...base, equal: { row: 0, xSpan: 12, ySpan: 11 } })
    expect(Math.abs(x / y - 1)).toBeLessThan(0.01)
    expect(l.width).toBeLessThan(960)
    expect(l.heights.reduce((a, b) => a + b, 0) + base.gap).toBeLessThanOrEqual(900)
  })

  it('keeps the other rows at their ratio share of what remains, or the minimum', () => {
    const { l } = unitsOf({
      ...base,
      ratios: [1, 2, 1],
      margins: { ...base.margins, top: [32, 32, 32], bottom: [14, 14, 44] },
      chrome: [28, 28, 28],
    })
    expect(l.heights[1]).toBeGreaterThanOrEqual(160)
    expect(l.heights[2]).toBeGreaterThanOrEqual(160)
  })

  it('sizes the other rows from the equal panel with ratiosOf equal, and fits the cap', () => {
    const input: ColumnInput = {
      ...base,
      ratios: [1, 0.2],
      equal: { row: 0, xSpan: 1, ySpan: 1 },
      margins: { ...base.margins, top: [32, 8], bottom: [2, 44] },
      chrome: [0, 0],
      minHeight: 40,
      ratiosOf: 'equal',
    }
    const { l, x, y } = unitsOf(input)
    expect(Math.abs(x / y - 1)).toBeLessThan(0.01)
    expect(l.heights[1] - 8 - 44).toBe(Math.round(0.2 * l.plot.height))
    expect(l.heights.reduce((a, b) => a + b, 0) + base.gap).toBeLessThanOrEqual(900)
  })
})

describe('layoutEqualColumn', () => {
  const input = {
    width: 600,
    frameHeight: 400,
    cap: 2000,
    aspects: [1, 1],
    margins: { left: 60, right: 20, top: [20, 20], bottom: [14, 44] },
    gap: 8,
  }
  it('gives every panel one plot width, so equal aspects give equal plots', () => {
    const l = layoutEqualColumn(input)
    expect(l.width).toBe(600)
    expect(l.heights[0] - 34).toBe(l.heights[1] - 64)
  })
  it('narrows the column to fit the cap', () => {
    const l = layoutEqualColumn({ ...input, cap: 500, frameHeight: 300 })
    expect(l.heights.reduce((a, b) => a + b, 0) + 8).toBeLessThanOrEqual(500)
    expect(l.width).toBeLessThan(600)
  })
})

describe('equalRowWidths', () => {
  it('leaves the same plot width in every column', () => {
    const margins = [
      { left: 60, right: 20 },
      { left: 16, right: 20 },
      { left: 16, right: 70 },
    ]
    const w = equalRowWidths(900, 8, margins)
    const plots = w.map((v, i) => v - margins[i].left - margins[i].right)
    expect(Math.max(...plots) - Math.min(...plots)).toBeLessThanOrEqual(1)
  })
})
