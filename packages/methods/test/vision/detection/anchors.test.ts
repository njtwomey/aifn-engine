/**
 * aifn-methods/vision/detection anchors: the offsets of the object-detection note's example and their inverse, anchor
 * grid geometry, IoU matching, FCOS targets and FPN levels (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import {
  anchorFreeTargets,
  anchorGrid,
  decodeBoxes,
  encodeBoxes,
  matchAnchors,
  pyramidLevel,
  type Box,
} from 'aifn-methods/vision/detection'

describe('encodeBoxes and decodeBoxes', () => {
  it('encode the note example to (0.156, -0.078, 0.223, -0.065) and decode it back', () => {
    const anchor: Box = [100 - 32, 100 - 64, 100 + 32, 100 + 64]
    const truth: Box = [110 - 40, 90 - 60, 110 + 40, 90 + 60]
    const [t] = encodeBoxes([anchor], [truth])
    ;[0.156, -0.078, 0.223, -0.065].forEach((v, i) => expect(Math.abs(t[i] - v)).toBeLessThan(1e-3))
    const [back] = decodeBoxes([anchor], [t])
    back.forEach((v, i) => expect(v).toBeCloseTo(truth[i], 9))
  })
})

describe('anchorGrid', () => {
  const g = anchorGrid({ featureShape: [10, 10], stride: 16, sizes: [64], ratios: [1] })

  it('centres cell (5, 7) of a stride-16 map at (120, 88)', () => {
    const a = g.anchors[5 * 10 + 7]
    expect((a[0] + a[2]) / 2).toBe(120)
    expect((a[1] + a[3]) / 2).toBe(88)
    expect(a[2] - a[0]).toBe(64)
    expect(g.index[57]).toEqual({ row: 5, col: 7, size: 0, ratio: 0 })
  })

  it('keeps the area of a ratio-2 anchor of size 64 at 4096', () => {
    const tall = anchorGrid({ featureShape: [1, 1], stride: 16, sizes: [64], ratios: [2] }).anchors[0]
    expect((tall[2] - tall[0]) * (tall[3] - tall[1])).toBeCloseTo(4096, 9)
    expect(tall[3] - tall[1]).toBeGreaterThan(tall[2] - tall[0])
  })

  it('matches a truth equal to an anchor with IoU 1, and far anchors negative', () => {
    const m = matchAnchors(g.anchors, [g.anchors[57]], { positive: 0.5, negative: 0.4 })
    expect(m.iou[57]).toBeCloseTo(1, 12)
    expect(m.labels[57]).toBe(1)
    expect(m.labels[0]).toBe(0)
    expect(m.bestAnchor).toEqual([57])
  })
})

describe('anchorFreeTargets', () => {
  it('gives FCOS distances (20, 28, 80, 12) and centre-ness 0.327', () => {
    const [p] = anchorFreeTargets([[120, 88]], [[100, 60, 200, 100]], [0, 64])
    expect(p.ltrb).toEqual([20, 28, 80, 12])
    expect(p.centreness).toBeCloseTo(Math.sqrt((20 / 80) * (12 / 28)), 12)
    expect(p.centreness).toBeCloseTo(0.327, 3)
    expect(p.positive).toBe(false)
    expect(anchorFreeTargets([[0, 0]], [[100, 60, 200, 100]], [0, 64])[0].matched).toBe(-1)
  })
})

describe('pyramidLevel', () => {
  it('assigns 224 to P4, 112 to P3, 448 to P5 and 32 to P2 (clamped)', () => {
    expect([224, 112, 448, 32].map((s) => pyramidLevel(s, s))).toEqual([4, 3, 5, 2])
  })
})
