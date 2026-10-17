/**
 * aifn-compute/nn/sequence scan orders on an 8 by 8 grid (the state-space-models-beyond-text note's checks): every order
 * a permutation, snake and Hilbert paths, 112 neighbour pairs, the mean and largest sequence distance of raster, snake
 * and cross scans, and how many cells a forward or bidirectional raster output sees.
 */
import { describe, expect, it } from 'vitest'
import {
  neighbourPairs,
  routePositions,
  scanOrder,
  sequenceDistance,
  visibleTo,
  type ScanRoute,
} from 'aifn-compute/nn/sequence'

const n = 8
const adjacent = (a: number, b: number) =>
  Math.abs((a % n) - (b % n)) + Math.abs(Math.floor(a / n) - Math.floor(b / n)) === 1

describe('scan orders', () => {
  it('gives permutations, and paths for snake and Hilbert orders', () => {
    for (const kind of ['row', 'column', 'snake', 'hilbert'] as const) {
      const o = scanOrder(kind, n, n)
      expect(new Set(o).size).toBe(n * n)
      expect(o.every((c) => c >= 0 && c < n * n)).toBe(true)
    }
    for (const kind of ['snake', 'hilbert'] as const) {
      const o = scanOrder(kind, n, n)
      expect(o.slice(1).every((c, i) => adjacent(o[i], c))).toBe(true)
    }
    const row = scanOrder('row', n, n)
    expect(row.slice(1).every((c, i) => adjacent(row[i], c))).toBe(false)
    expect(() => scanOrder('hilbert', 6, 6)).toThrow(/power of two/)
  })

  it('measures neighbour distances: raster 4.5 (56 at 1), snake 4.5 and 15, cross scan all 1', () => {
    const pairs = neighbourPairs(n, n)
    expect(pairs.length).toBe(112)
    const schemes: Record<string, ScanRoute[]> = {
      row: [{ order: 'row' }],
      both: [{ order: 'row' }, { order: 'row', reverse: true }],
      snake: [{ order: 'snake' }],
      cross: [
        { order: 'row' },
        { order: 'row', reverse: true },
        { order: 'column' },
        { order: 'column', reverse: true },
      ],
    }
    const stats = Object.fromEntries(
      Object.entries(schemes).map(([name, routes]) => {
        const pos = routePositions(routes, n, n)
        const d = pairs.map(([a, b]) => sequenceDistance(pos, a, b))
        const mean = d.reduce((a, x) => a + x, 0) / d.length
        return [name, { mean, max: Math.max(...d), ones: d.filter((x) => x === 1).length }]
      }),
    )
    expect(stats.row).toEqual({ mean: 4.5, max: 8, ones: 56 })
    expect(stats.snake.mean).toBe(4.5)
    expect(stats.snake.max).toBe(15)
    expect(stats.cross.max).toBe(1)
    expect(stats.both.mean).toBe(4.5)
  })

  it('sees 29 cells from (3, 4) on a forward raster scan, and all 64 both ways', () => {
    const seen = (routes: ScanRoute[]) => {
      const pos = routePositions(routes, n, n)
      return Array.from({ length: 64 }, (_, s) => visibleTo(pos, 28, s)).filter(Boolean).length
    }
    expect(seen([{ order: 'row' }])).toBe(29)
    expect(seen([{ order: 'row' }, { order: 'row', reverse: true }])).toBe(64)
  })
})
