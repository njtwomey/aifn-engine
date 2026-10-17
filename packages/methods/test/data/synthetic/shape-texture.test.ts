/**
 * aifn-methods/data/synthetic shape-texture images: the pairings, the value range of clean images, the stripes and a
 * Gauss-circle count of a rendered disc (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  insideShape,
  renderShapes,
  SHAPE_NAMES,
  shapeTextureDataset,
  TEXTURE_NAMES,
  textureValue,
} from 'aifn-methods/data/synthetic'

describe('shapeTextureDataset', () => {
  it('matches texture to shape under matched pairing, with clean values in {0, 0.3, 1}', () => {
    const m = shapeTextureDataset(stream(1), { perClass: 20, noise: 0 })
    expect(m.x.shape).toEqual([60, 1, 16, 16])
    expect(Array.from(m.texture)).toEqual(Array.from(m.shape))
    expect(toFlat(m.x).every((v) => v === 0 || v === 0.3 || v === 1)).toBe(true)
  })

  it('never matches under conflict pairing, and uses every texture (53, 48, 49 of 150)', () => {
    const cf = shapeTextureDataset(stream(2), { perClass: 50, pairing: 'conflict' })
    expect(Array.from(cf.texture).every((t, i) => t !== cf.shape[i])).toBe(true)
    const counts = [0, 0, 0]
    cf.texture.forEach((t) => counts[t]++)
    expect(counts).toEqual([53, 48, 49])
  })

  it('with restyle 1 agrees with the shape about a third of the time (0.348)', () => {
    const r = shapeTextureDataset(stream(3), { perClass: 300, restyle: 1 })
    const agree = Array.from(r.texture).filter((t, i) => t === r.shape[i]).length
    expect(agree).toBe(313)
    expect(agree / r.texture.length).toBeCloseTo(0.348, 3)
  })

  it('throws for unequal shapes and textures under matched pairing', () => {
    expect(() => shapeTextureDataset(stream(0), { textures: ['solid', 'dots'] })).toThrow(AifnError)
  })
})

describe('renderShapes', () => {
  it('alternates horizontal stripes by row', () => {
    const stripes = renderShapes(8, [{ shape: 'square', texture: 'horizontal', x: 4, y: 4, radius: 4 }])
    expect(stripes[3 * 8 + 3]).not.toBe(stripes[4 * 8 + 3])
    expect(stripes[3 * 8 + 3]).toBe(stripes[3 * 8 + 4])
  })

  it('covers 29 pixels with a solid disc of radius 3 (the Gauss circle count)', () => {
    const disc = renderShapes(9, [{ shape: 'disc', texture: 'solid', x: 4.5, y: 4.5, radius: 3 }])
    expect(disc.reduce((a, b) => a + b, 0)).toBe(29)
    expect(disc[40]).toBe(1)
    expect(disc[0]).toBe(0)
  })

  it('needs a stream for noise', () => {
    expect(() => renderShapes(4, [], { noise: 0.1 })).toThrow(AifnError)
  })

  it('has six shapes and five textures', () => {
    expect(SHAPE_NAMES.length).toBe(6)
    expect(TEXTURE_NAMES.length).toBe(5)
    expect(insideShape('ring', 0, 0, 4)).toBe(false)
    expect(textureValue('dots', 3, 3)).toBe(1)
    expect(textureValue('dots', 3, 4)).toBe(0.3)
  })
})
