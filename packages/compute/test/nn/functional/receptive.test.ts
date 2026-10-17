/**
 * aifn-compute/nn/functional receptive fields: `receptiveField` on VGG-16 and a dilated context module,
 * `effectiveReceptiveField` of two uniform layers, and `dilatedTapCoverage` / `meanValidTaps` on the ASPP rates of
 * DeepLabv3 (the site's self-checks).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { tensor, toFlat, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  conv1d,
  dilatedTapCoverage,
  effectiveReceptiveField,
  meanValidTaps,
  receptiveField,
  type ReceptiveFieldLayer,
} from 'aifn-compute/nn/functional'

describe('receptiveField', () => {
  it('gives VGG-16 its 196 and 212 pixel fields, jump 32 and start 15.5', () => {
    const vgg: ReceptiveFieldLayer[] = []
    for (const n of [2, 2, 3, 3, 3]) {
      for (let i = 0; i < n; i++) vgg.push({ kernel: 3, padding: 1 })
      vgg.push({ kernel: 2, stride: 2 })
    }
    const rf = receptiveField(vgg)
    expect(rf.length).toBe(vgg.length + 1)
    expect(rf.at(-1)!.size).toBe(212)
    expect(rf.at(-2)!.size).toBe(196)
    expect(rf.at(-1)!.jump).toBe(32)
    expect(rf.at(-1)!.start).toBe(15.5)
  })

  it('doubles with dilations 1, 2, 4, 8 (Yu and Koltun)', () => {
    expect(receptiveField([1, 2, 4, 8].map((d) => ({ kernel: 3, dilation: d }))).map((r) => r.size)).toEqual([
      1, 3, 7, 15, 31,
    ])
  })

  it('gives the docstring example', () => {
    expect(receptiveField([{ kernel: 3 }, { kernel: 3 }, { kernel: 2, stride: 2 }]).map((r) => r.size)).toEqual([
      1, 3, 5, 6,
    ])
  })
})

describe('effectiveReceptiveField', () => {
  it('is the kernel convolved with itself for two uniform 3-tap layers', () => {
    const w = tensor([[[1 / 3, 1 / 3, 1 / 3]]])
    const net = (x: Value) => conv1d(conv1d(x, w, { padding: 1 }), w, { padding: 1 })
    const g = toFlat(effectiveReceptiveField(net, zeros([1, 1, 11]), [0, 0, 5]) as Tensor)
    const want = [0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0]
    g.forEach((v, i) => expect(v * 9).toBeCloseTo(want[i], 9))
  })
})

describe('dilatedTapCoverage', () => {
  it('keeps nine taps in the interior at rate 1 and one at rate 64 on 65 x 65', () => {
    const r1 = dilatedTapCoverage(65, 3, 1)
    expect(r1.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(r1[9]).toBeCloseTo((63 * 63) / (65 * 65), 12)
    expect((63 / 65) ** 2).toBeCloseTo(0.939, 3)
    const r64 = dilatedTapCoverage(65, 3, 64)
    expect(r64[9]).toBe(0)
    expect(r64[1]).toBeCloseTo((63 * 63) / (65 * 65), 12)
    expect(dilatedTapCoverage(65, 3, 32)[9]).toBeCloseTo(1 / (65 * 65), 12)
  })

  it('counts 2, 2, 3, 2, 2 taps per axis on n = 5, d = 2: mean 4.84', () => {
    expect(meanValidTaps(5, 3, 2)).toBeCloseTo((11 / 5) ** 2, 12)
  })

  it('gives 0.405, 0.074 and 0 nine-tap shares at rates 6, 12, 18 on 33 x 33', () => {
    expect(dilatedTapCoverage(33, 3, 6)[9]).toBeCloseTo(0.405, 3)
    expect(dilatedTapCoverage(33, 3, 12)[9]).toBeCloseTo(0.074, 3)
    expect(dilatedTapCoverage(33, 3, 18)[9]).toBe(0)
  })

  it('throws for an even kernel', () => {
    expect(() => dilatedTapCoverage(9, 2, 1)).toThrow(AifnError)
  })
})
