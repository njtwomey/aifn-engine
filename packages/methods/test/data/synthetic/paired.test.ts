import { describe, expect, it } from 'vitest'
import { datasetRegistry, generate } from 'aifn-methods/data'
import { PAIRED_RGB, pairedShapes, type PairedViews } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { toFlat } from 'aifn-compute/foundation/tensor'

const flat = (t: Parameters<typeof toFlat>[0]) => Array.from(toFlat(t))

describe('pairedShapes', () => {
  it('is deterministic in its stream and differs between streams', () => {
    const a = pairedShapes(stream(7), { n: 40 })
    const b = pairedShapes(stream(7), { n: 40 })
    const c = pairedShapes(stream(8), { n: 40 })
    expect(flat(a.a)).toEqual(flat(b.a))
    expect(flat(a.b)).toEqual(flat(b.b))
    expect(flat(a.truth.combination)).toEqual(flat(b.truth.combination))
    expect(flat(a.a)).not.toEqual(flat(c.a))
  })

  it('pairs row i of the image view with row i of the caption view: both describe the same object', () => {
    const d = pairedShapes(stream(1), { n: 60, size: 10, pixelNoise: 0, captionNoise: 0 })
    expect(d.a.shape).toEqual([60, 3 * 10 * 10])
    expect(d.b.shape).toEqual([60, 9])
    expect(flat(d.truth.pair)).toEqual(Array.from({ length: 60 }, (_, i) => i))
    const shape = flat(d.truth.attributes.shape)
    const size = flat(d.truth.attributes.size)
    const colour = flat(d.truth.attributes.colour)
    const a = flat(d.a)
    const b = flat(d.b)
    const P = 100
    for (let i = 0; i < 60; i++) {
      // The clean caption is the one-hot of the object's attributes and equals its class prototype.
      const caption = b.slice(i * 9, (i + 1) * 9)
      const hot = [shape[i], 4 + size[i], 6 + colour[i]]
      caption.forEach((v, j) => expect(v).toBe(hot.includes(j) ? 1 : 0))
      const k = flat(d.truth.combination)[i]
      expect(flat(d.prototypes).slice(k * 9, (k + 1) * 9)).toEqual(caption)
      expect(d.meta.combinationCodes[k]).toEqual([shape[i], size[i], colour[i]])
      // The image's channels are the coverage times the colour's RGB, so their ratios are the colour's.
      const rgb = PAIRED_RGB[colour[i]]
      const sums = [0, 1, 2].map((c) => a.slice(i * 3 * P + c * P, i * 3 * P + (c + 1) * P).reduce((s, v) => s + v, 0))
      expect(sums[0]).toBeGreaterThan(0)
      expect(sums[1] / sums[0]).toBeCloseTo(rgb[1] / rgb[0], 10)
      expect(sums[2] / sums[0]).toBeCloseTo(rgb[2] / rgb[0], 10)
    }
    // Large shapes cover more of the image than small ones.
    const cover = (i: number) =>
      a.slice(i * 3 * P, (i + 1) * 3 * P).reduce((s, v) => s + v, 0) / PAIRED_RGB[colour[i]].reduce((s, v) => s + v, 0)
    const mean = (bin: number) => {
      const rows = size.flatMap((s, i) => (s === bin ? [cover(i)] : []))
      return rows.reduce((s, v) => s + v, 0) / rows.length
    }
    expect(mean(1)).toBeGreaterThan(1.5 * mean(0))
  })

  it('holds out shape–colour combinations: never in seen data, only in held-out data', () => {
    const seen = pairedShapes(stream(2), { n: 400, heldOut: 2 })
    const held = pairedShapes(stream(3), { n: 200, heldOut: 2, include: 'heldOut' })
    const all = pairedShapes(stream(4), { n: 600, heldOut: 2, include: 'all' })
    const heldCombos = new Set(seen.meta.heldOutCombinations)
    // Red triangles and blue squares, each at both sizes.
    expect(seen.meta.heldOutCombinations.map((k) => seen.meta.combinationNames[k])).toEqual([
      'small blue square',
      'large blue square',
      'small red triangle',
      'large red triangle',
    ])
    expect(flat(seen.truth.combination).some((k) => heldCombos.has(k))).toBe(false)
    expect(flat(seen.truth.heldOut).every((h) => h === 0)).toBe(true)
    expect(flat(held.truth.combination).every((k) => heldCombos.has(k))).toBe(true)
    expect(flat(held.truth.heldOut).every((h) => h === 1)).toBe(true)
    const allHeld = flat(all.truth.heldOut)
    flat(all.truth.combination).forEach((k, i) => expect(allHeld[i]).toBe(heldCombos.has(k) ? 1 : 0))
    expect(allHeld.filter((h) => h === 1).length).toBeGreaterThan(50)
    expect(() => pairedShapes(stream(1), { heldOut: 0, include: 'heldOut' })).toThrow(/no combinations/)
  })

  it('is registered with pairs output and runs at its default knobs', () => {
    const entry = datasetRegistry.pairedShapes
    expect(entry.info.output).toBe('pairs')
    const d = generate(entry, stream(5), defaults(entry.info.knobs)) as PairedViews
    expect(d.kind).toBe('pairs')
    expect(d.a.shape).toEqual([600, 3 * 12 * 12])
    expect(d.meta.recipe?.base).toBe('pairedShapes')
  })
})
