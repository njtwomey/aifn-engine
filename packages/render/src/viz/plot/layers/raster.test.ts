import { describe, expect, it } from 'vitest'
import { AxisModel } from '../axis'
import type { LayerContext } from '../layer'
import { Raster, stretchedImage, type RasterProps } from './raster'

/** A layer context as a Plot builds it: a plot area of 400 × 300 px over the raster's own bounds. */
function context(p: RasterProps): LayerContext {
  const dx = p.x.length > 1 ? p.x[1] - p.x[0] : 1
  const dy = p.y.length > 1 ? p.y[1] - p.y[0] : 1
  const lo = (v: ArrayLike<number>, d: number) => Math.min(v[0], v[v.length - 1]) - Math.abs(d) / 2
  const hi = (v: ArrayLike<number>, d: number) => Math.max(v[0], v[v.length - 1]) + Math.abs(d) / 2
  return {
    mode: 'light',
    color: '#000',
    slot: 0,
    id: 'r',
    box: { x: [lo(p.x, dx), hi(p.x, dx)], y: [lo(p.y, dy), hi(p.y, dy)], xLog: false, yLog: false },
    plot: { width: 400, height: 300 },
    range: () => undefined,
    x: new AxisModel({ label: 'a' }),
    y: new AxisModel({ label: 'b' }),
  }
}

// z[i][j] = 10 i + j: every cell's value names its row and column.
const props: RasterProps = {
  x: [0, 1, 2, 3],
  y: [5, 6, 7],
  z: [0, 1, 2].map((i) => [0, 1, 2, 3].map((j) => 10 * i + j)),
  valueLabel: 'v',
}

describe('Raster hover after the canvas-image rewrite', () => {
  const out = Raster.layer.build(props, context(props))
  const cells = out.series.find((s) => s.id === 'r:cells') as { data: number[][]; symbolSize: number[] }

  it('keeps the image silent and puts one transparent hit target on every cell centre', () => {
    const image = out.series.find((s) => s.id === 'r:image') as { silent: boolean; tooltip: { show: boolean } }
    expect(image.silent).toBe(true)
    expect(image.tooltip.show).toBe(false)
    expect(cells.data).toHaveLength(12)
    for (const [x, y, v] of cells.data) expect(v).toBe(10 * (y - 5) + x)
    // Each target covers its cell: 400 px over 4 columns, 300 px over 3 rows, plus a pixel so no seam is left.
    expect(cells.symbolSize).toEqual([101, 101])
  })

  it('formats the tooltip of a hovered cell with its coordinates and value', () => {
    const format = out.tooltip?.['r:cells']
    expect(format).toBeDefined()
    const dataIndex = cells.data.findIndex(([x, y]) => x === 2 && y === 6)
    const html = format!({ seriesName: '__cells', value: cells.data[dataIndex], marker: '', dataIndex })
    expect(html).toBe('a 2, b 6<br/>v: <b>12</b>')
  })

  it('reads the cell under a pointer anywhere inside it, and nothing outside the grid', () => {
    expect(out.pointer?.([2.4, 6.45])?.rows[0].value).toBe('12')
    expect(out.pointer?.([-0.49, 4.51])?.rows[0].value).toBe('0')
    expect(out.pointer?.([3.4, 7.4])?.rows[0].value).toBe('23')
    expect(out.pointer?.([4.6, 6])).toBeNull()
  })

  it('reads descending axes the same way', () => {
    const flipped: RasterProps = { ...props, y: [7, 6, 5] }
    const o = Raster.layer.build(flipped, context(flipped))
    // Row 0 is now at y = 7.
    expect(o.pointer?.([1, 7.2])?.rows[0].value).toBe('1')
    expect(o.pointer?.([1, 5])?.rows[0].value).toBe('21')
  })

  it('names categories in the tooltip', () => {
    const classes: RasterProps = {
      x: [0, 1],
      y: [0],
      z: [[0, 1]],
      scale: 'categorical',
      categoryNames: ['cat', 'dog'],
      valueLabel: 'class',
    }
    const o = Raster.layer.build(classes, context(classes))
    const html = o.tooltip?.['r:cells']({ seriesName: '__cells', value: [1, 0, 1], marker: '', dataIndex: 1 })
    expect(html).toContain('class: <b>dog</b>')
  })
})

describe('Raster image element', () => {
  it('stretches the image to its box by a transform, so the SVG renderer cannot keep its aspect', () => {
    // A 5 × 40 grid at 64 px per cell, into a box wider than tall: the SVG renderer's <image> would fit it at 1:8 and
    // centre it if the box went into its width and height.
    const el = stretchedImage({ width: 320, height: 2560 }, 58, 24, 464.5, 408)
    expect(el.style).toMatchObject({ x: 0, y: 0, width: 320, height: 2560 })
    expect([el.x, el.y]).toEqual([58, 24])
    expect(el.style.width * el.scaleX).toBeCloseTo(406.5)
    expect(el.style.height * el.scaleY).toBeCloseTo(384)
  })
})
