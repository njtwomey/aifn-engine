/** Sparse GPs step by step (`sparse.ts`): the L-BFGS fit as an Algorithm, greedy inducing-point growth, registration. */
import { describe, expect, it } from 'vitest'
import { regression1d } from 'aifn-methods/data/synthetic'
import {
  fitSparseGp,
  sparseGp,
  sparseGpAt,
  sparseGpFitSteps,
  sparseGpGrowSteps,
} from 'aifn-methods/learning/gaussian-processes'
import { stream } from 'aifn-compute/foundation/random'
import { isEntry } from 'aifn-compute/foundation/registry'
import { take, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { rbf } from 'aifn-compute/learning/kernels'

const data = regression1d(stream('sparse-steps'), { n: 80, fn: 'sine', range: [0, 10], spacing: 'gapped', noise: 0.2 })
const x = data.x as Tensor
const y = data.y as Tensor
const kernel = rbf({ lengthscale: 1, variance: 1 })

describe('sparseGpFitSteps', () => {
  const z = take(x, [0, 5, 10, 15, 20, 25]) as Tensor
  it.each(['vfe', 'fitc'] as const)('%s: the objective never decreases and states carry Z and θ', (method) => {
    const tr = trace(sparseGpFitSteps(kernel, x, y, z, { method, noiseVariance: 0.1, exact: true }), undefined, 40)
    const states = tr.steps
    expect(states.length).toBeGreaterThan(5)
    for (let t = 1; t < states.length; t++)
      expect(states[t].logMarginal).toBeGreaterThanOrEqual(states[t - 1].logMarginal - 1e-9)
    const last = tr.final
    expect(last.inducing.shape).toEqual([6, 1])
    expect(Object.keys(last.hyper).length).toBe(2)
    // The state rebuilds the sparse GP whose objective it reports.
    expect(sparseGpAt(kernel, x, y, last, { method }).logMarginal).toBeCloseTo(last.logMarginal, 8)
    // The VFE bound is below the exact evidence at every state.
    if (method === 'vfe') for (const s of states) expect(s.exact!).toBeGreaterThanOrEqual(s.logMarginal - 1e-8)
  })

  it('is what fitSparseGp runs', () => {
    const fit = fitSparseGp(kernel, x, y, z, { noiseVariance: 0.1, maxSteps: 30 })
    const tr = trace(sparseGpFitSteps(kernel, x, y, z, { noiseVariance: 0.1 }), undefined, 30)
    expect(fit.model.logMarginal).toBeCloseTo(tr.final.logMarginal, 10)
  })
})

describe('sparseGpGrowSteps', () => {
  it('adds one candidate per step and never lowers the VFE bound with the hyperparameters held', () => {
    const alg = sparseGpGrowSteps(kernel, x, y, { noiseVariance: 0.05, maxInducing: 10, candidates: 40, exact: true })
    const tr = trace(alg, undefined, 20, { stream: stream('grow') })
    const states = tr.steps
    expect(states[0].inducing.shape[0]).toBe(1)
    expect(tr.final.inducing.shape[0]).toBe(10)
    expect(tr.meta.stopped).toBe('done')
    for (let t = 1; t < states.length; t++) {
      const s = states[t]
      expect(s.inducing.shape[0]).toBe(t + 1)
      expect(s.candidates).not.toContain(s.added)
      expect(toFlat(s.inducing).at(-1)).toBe(toFlat(x)[s.added!])
      expect(s.logMarginal).toBeGreaterThanOrEqual(states[t - 1].logMarginal - 1e-9)
      // The gap to the exact evidence (the KL from the variational posterior) shrinks as Z grows.
      expect(s.exact! - s.logMarginal).toBeLessThanOrEqual(states[t - 1].exact! - states[t - 1].logMarginal + 1e-8)
    }
  })

  it('picks the best candidate: no other single addition scores higher', () => {
    const alg = sparseGpGrowSteps(kernel, x, y, { noiseVariance: 0.05, maxInducing: 4, candidates: 25 })
    const tr = trace(alg, undefined, 3, { stream: stream('grow-best') })
    const prev = tr.steps[1]
    const next = tr.steps[2]
    for (const c of prev.candidates.slice(0, 10)) {
      const z = take(x, [...Array.from(toFlat(prev.inducing), (v) => toFlat(x).indexOf(v)), c]) as Tensor
      expect(sparseGp(kernel, x, y, z, { noiseVariance: 0.05 }).logMarginal).toBeLessThanOrEqual(
        next.logMarginal + 1e-9,
      )
    }
  })

  it('re-optimises after each addition when asked, still never lowering the bound', () => {
    const alg = sparseGpGrowSteps(kernel, x, y, { noiseVariance: 0.1, maxInducing: 6, candidates: 30, reoptimise: 5 })
    const tr = trace(alg, undefined, 10, { stream: stream('grow-reopt') })
    for (let t = 1; t < tr.steps.length; t++) {
      expect(tr.steps[t].logMarginal).toBeGreaterThanOrEqual(tr.steps[t].afterAdd - 1e-9)
      expect(tr.steps[t].afterAdd).toBeGreaterThanOrEqual(tr.steps[t - 1].logMarginal - 1e-9)
    }
    expect(tr.final.hyper.lengthscale).not.toBe(1)
  })

  it('both are registered algorithms', () => {
    expect(isEntry(sparseGpFitSteps, 'algorithm')).toBe(true)
    expect(isEntry(sparseGpGrowSteps, 'algorithm')).toBe(true)
  })
})
