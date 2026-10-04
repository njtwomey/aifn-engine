/**
 * Type and runtime conformance of the unsupervised area to `aifn-compute/foundation/contracts`: iterative fits are factory
 * Algorithms over `Status` states, fitted models are `Model`s with their capabilities, and transductive methods say so.
 */
import { describe, expect, expectTypeOf, it } from 'vitest'
import {
  gaussianMixture,
  kMedoids,
  kmeans,
  kmeansSteps,
  meanShift,
  miniBatchKMeans,
  type KMeansInit,
  type KMeansState,
} from 'aifn-methods/unsupervised/clustering'
import { isomap, tsne } from 'aifn-methods/unsupervised/embedding'
import { pca } from 'aifn-methods/unsupervised/embedding/linear'
import type * as C from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'

type KMeansModel = ReturnType<ReturnType<typeof kmeans>['fit']>

const x = tensor([
  [0, 0],
  [0.1, 0.2],
  [3, 3],
  [3.1, 2.9],
  [6, 0],
  [6.2, 0.1],
  [0.2, 0.1],
  [3.2, 3.1],
  [5.9, 0.2],
])

describe('k-means', () => {
  it("Lloyd's algorithm is a factory Algorithm over Status states", () => {
    expectTypeOf(kmeansSteps).returns.toExtend<C.Algorithm<KMeansInit, KMeansState>>()
    expectTypeOf<KMeansState>().toExtend<C.Status>()
  })
  it('the fitted model is a Model with its capabilities', () => {
    expectTypeOf<KMeansModel>().toExtend<C.Model & C.Decides<Tensor> & C.Transforms<Tensor> & C.Trained<KMeansState>>()
  })
})

describe('fitted models', () => {
  it('are kind "model" with a name; inductive ones decide or transform new points', () => {
    const d = dataset(x)
    const s = { stream: stream(1) }
    const inductive = [
      kmeans({ k: 3 }).fit(d, s),
      miniBatchKMeans({ k: 3, batchSize: 4 }).fit(d, s),
      kMedoids({ k: 3 }).fit(d, s),
      meanShift({ bandwidth: 1.5 }).fit(d, s),
      gaussianMixture({ k: 3 }).fit(d, s),
    ]
    for (const m of inductive) {
      expect(m.kind).toBe('model')
      expect(typeof m.name).toBe('string')
      expect(m.decide(x).shape).toEqual([9])
    }
    const p = pca({ components: 1 }).fit(d)
    expect(p.kind).toBe('model')
    expect(p.transform(x).shape).toEqual([9, 1])
  })
  it('transductive embeddings are marked and keep their embedding', () => {
    const t = tsne({ perplexity: 2, iterations: 20 }).fit(dataset(x), { stream: stream(2) })
    expect(t.transductive).toBe(true)
    expect(t.embedding.shape).toEqual([9, 2])
    const i = isomap({ neighbours: 4 }).fit(dataset(x))
    expect(i.transductive).toBe(true)
  })
})
