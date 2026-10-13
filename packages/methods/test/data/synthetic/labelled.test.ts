import { describe, expect, expectTypeOf, it } from 'vitest'
import {
  anisotropicBlobs,
  blobs,
  checkerboard,
  circles,
  gaussians,
  moons,
  rings,
  shuffleDataset,
  spirals,
  xor,
  type LabelledDataset,
} from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'
import type { Supervised } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import type { Tensor } from 'aifn-compute/foundation/tensor'

describe('labelled generators', () => {
  it('return a dataset whose y is required, which cross-validates without a cast', () => {
    const data = gaussians(stream(0), { n: 120 })
    expectTypeOf(data.y).toEqualTypeOf<Tensor>()
    expectTypeOf(data).toExtend<Supervised>()
    const cv = crossValidate(logisticRegression(), data, kFold({ k: 3, shuffle: true }), [accuracy], {
      stream: stream(1),
    })
    expect(cv.folds.length).toBe(3)
    expect(cv.mean.accuracy).toBeGreaterThan(0.5)
    const model = logisticRegression().fit(shuffleDataset(stream(2), blobs(stream(3), { n: 60, centers: 2 })))
    expect(model).toBeDefined()
  })

  it('type every labelled point generator, and shuffling keeps the type', () => {
    const s = stream(4)
    const all: LabelledDataset[] = [
      blobs(s),
      moons(s),
      circles(s),
      rings(s),
      spirals(s),
      xor(s),
      checkerboard(s),
      gaussians(s),
      anisotropicBlobs(s),
    ]
    for (const d of all) expect(d.y.shape).toEqual([d.x.shape[0]])
    expectTypeOf(shuffleDataset(s, all[0])).toEqualTypeOf<LabelledDataset>()
  })
})
