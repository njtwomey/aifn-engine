/**
 * MIL pooling (MILLET's five poolings): shapes; with a linear classifier, instance pooling gives the same logits as
 * global average pooling and its interpretation is the class activation map; conjunctive pooling is the mean of the
 * attention-weighted instance predictions; a mask pools over the kept instances only, as pooling the smaller bag;
 * gradients reach every parameter.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { stream, normals } from 'aifn-compute/foundation/random'
import { dense, fromData, sum, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { MIL_POOLING_KINDS, milPool, MilPooling, type MilPoolingParams } from 'aifn-compute/nn/layers'

const B = 2
const t = 5
const d = 4
const c = 3
const Z = fromData(Float64Array.from(toFlat(normals(stream('z'), B * t * d))), [B, t, d])

describe('milPool', () => {
  const params = MilPooling(d, c, 'conjunctive').init(stream(1))
  it('every kind gives [B, c] logits and an interpretation per instance', () => {
    for (const kind of MIL_POOLING_KINDS) {
      const out = milPool(kind, params, Z)
      expect((out.logits as Tensor).shape).toEqual([B, c])
      expect((out.interpretation as Tensor).shape).toEqual(kind === 'attention' ? [B, t] : [B, t, c])
    }
  })
  it('instance pooling equals global average pooling for a linear classifier', () => {
    const a = toFlat(milPool('instance', params, Z).logits as Tensor)
    const b = toFlat(milPool('embedding', params, Z).logits as Tensor)
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12))
  })
  it('conjunctive logits are the mean of a_j ŷ_j', () => {
    const out = milPool('conjunctive', params, Z)
    const I = dense.data(out.interpretation as Tensor)
    const L = toFlat(out.logits as Tensor)
    for (let b = 0; b < B; b++)
      for (let k = 0; k < c; k++) {
        let s = 0
        for (let j = 0; j < t; j++) s += I[(b * t + j) * c + k]
        expect(L[b * c + k]).toBeCloseTo(s / t, 12)
      }
  })
  it('a mask pools over the kept instances only', () => {
    const keep = [1, 0, 1, 1, 0]
    const mask = fromData(Float64Array.from([...keep, ...keep]), [B, t])
    const kept = [0, 2, 3]
    const Zk = fromData(
      Float64Array.from(
        Array.from({ length: B }, (_, b) =>
          kept.flatMap((j) => Array.from(dense.data(Z).slice((b * t + j) * d, (b * t + j + 1) * d))),
        ).flat(),
      ),
      [B, kept.length, d],
    )
    for (const kind of MIL_POOLING_KINDS) {
      const a = toFlat(milPool(kind, params, Z, mask).logits as Tensor)
      const b = toFlat(milPool(kind, params, Zk).logits as Tensor)
      a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12))
    }
  })
  it('a mask that is not exactly [B, t] throws', () => {
    const extra = fromData(new Float64Array(B * t * 2).fill(1), [B, t, 2])
    expect(() => milPool('instance', params, Z, extra)).toThrow(ShapeError)
    expect(() => milPool('instance', params, Z, fromData(new Float64Array(B).fill(1), [B]))).toThrow(ShapeError)
  })
  it('gradients reach the classifier and the attention head', () => {
    const g = grad((p: MilPoolingParams) => sum(milPool('additive', p, Z).logits))(params) as MilPoolingParams
    expect(toFlat(g.classifier.weight).some((v) => v !== 0)).toBe(true)
    expect(toFlat(g.attention!.hidden.weight).some((v) => v !== 0)).toBe(true)
  })
})
