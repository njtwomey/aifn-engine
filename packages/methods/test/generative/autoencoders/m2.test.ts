/**
 * aifn-methods/generative/autoencoders: M2, on the values of the site's self-check (semi-supervised-vae.check.ts).
 * The check's 400-step accuracies (M2 0.995 against 0.797 for the classifier alone) take seconds and are not repeated;
 * a 40-step run checks that the loss falls.
 */
import { describe, expect, it } from 'vitest'
import { parallelWaves, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import {
  autoencoderAlgorithms,
  autoencoderFunctions,
  autoencoderLoss,
  initM2,
  labelledBound,
  m2ClassProbabilities,
  m2Decode,
  m2Model,
  semiSupervisedVaeRun,
  semiSupervisedVaeTraining,
  unlabelledBound,
} from 'aifn-methods/generative/autoencoders'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { float64Data, tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { oneHot } from 'aifn-compute/learning/losses'
import { adamRule } from 'aifn-compute/optim/first-order'
import { expectInfo } from '../../registry'

describe('the bounds', () => {
  it('U at q = (0.8, 0.2), L = (3.1, 5.4) is 3.0596; its minimum −log Σ e^−L = 3.0045 is at softmax(−L)', () => {
    expect(unlabelledBound([0.8, 0.2], [3.1, 5.4])).toBeCloseTo(3.0596, 4)
    const q0 = 1 / (1 + Math.exp(-2.3))
    const uStar = unlabelledBound([q0, 1 - q0], [3.1, 5.4])
    expect(Math.abs(uStar + Math.log(Math.exp(-3.1) + Math.exp(-5.4)))).toBeLessThan(1e-12)
    expect(uStar).toBeCloseTo(3.0045, 4)
    for (const q of [0.5, 0.8, 0.95]) expect(unlabelledBound([q, 1 - q], [3.1, 5.4])).toBeGreaterThanOrEqual(uStar)
  })

  it('the labelled bound less log K and the Gaussian normaliser is the cvae loss with the same noise (174.609412)', () => {
    const d = parallelWaves(stream(1), { n: 40 })
    const model = m2Model(2, 2)
    const params = initM2(model, stream(2))
    const s = stream(3)
    const hot = oneHot(d.y, 2)
    const eps = normals(child(s, 'eps'), [40, 1])
    const ours = float64Data(labelledBound(model, params, d.x, hot, eps) as Tensor)
    const mean = ours.reduce((a, b) => a + b, 0) / 40 - Math.log(2) - 2 * Math.log(0.1 * Math.sqrt(2 * Math.PI))
    const raw = autoencoderLoss(model.vae, params.vae, d.x, s, hot).loss as unknown
    const engine = typeof raw === 'number' ? raw : float64Data(raw as Tensor)[0]
    expect(Math.abs(mean - engine)).toBeLessThan(1e-8)
    expect(engine).toBeCloseTo(174.609412, 5)
  })

  it('m2Decode gives one row per code for the class asked', () => {
    const model = m2Model(2, 2, { hidden: [4], classifierHidden: [4] })
    const params = initM2(model, stream(1))
    const z = tensor([[-1], [0], [1]])
    expect(m2Decode(model, params, z, 0).length).toBe(6)
    expect(Array.from(m2Decode(model, params, z, 0))).not.toEqual(Array.from(m2Decode(model, params, z, 1)))
  })
})

describe('semiSupervisedVaeTraining', () => {
  const d = parallelWaves(stream('gssl/m2/waves/1'), {})
  const sp = semiSupervisedSplit(stream('gssl/m2/split/1'), d.y, 4, { x: d.x, draw: 'global', strict: true })
  const problem = { labelledX: sp.labelledX!, labelledY: sp.labelledY, unlabelledX: sp.unlabelledX!, classes: 2 }

  it('the M2 loss falls over the first 40 steps (a = 5)', () => {
    const model = m2Model(2, 2)
    const alg = semiSupervisedVaeTraining(problem, { model, classWeight: 5, optimizer: adamRule({ stepSize: 3e-3 }) })
    const first = run(alg, undefined, 1)
    const later = run(alg, undefined, 40)
    expect(later.loss).toBeLessThan(first.loss)
    expect(Number.isFinite(later.labelled) && Number.isFinite(later.unlabelled)).toBe(true)
  })

  it('the classifier alone minimises the labelled cross-entropy only', () => {
    const model = m2Model(2, 2, { hidden: [8], classifierHidden: [8] })
    const s = run(semiSupervisedVaeTraining(problem, { model, mode: 'classifier' }), undefined, 3)
    expect(s.loss).toBe(s.classification)
    expect(Number.isNaN(s.labelled)).toBe(true)
  })

  it('the plain-data run reproduces the algorithm', () => {
    const options = { hidden: [8], classifierHidden: [8], classWeight: 5 }
    const shots = [
      ...semiSupervisedVaeRun(problem, {
        ...options,
        steps: 6,
        every: 3,
        truth: sp.unlabelledTruth,
        codes: tensor([[0]]),
      }),
    ]
    expect(shots.map((c) => c.step)).toEqual([0, 3, 6])
    const model = m2Model(2, 2, options)
    const s = run(semiSupervisedVaeTraining(problem, { model, classWeight: 5 }), undefined, 6)
    const last = shots.at(-1)!
    expect(last.loss).toBe(s.loss)
    expect(Array.from(last.probabilities)).toEqual(
      Array.from(m2ClassProbabilities(model, s.params, problem.unlabelledX)),
    )
    expect(last.decoded.length).toBe(2)
    expect(last.accuracy).toBeGreaterThanOrEqual(0)
  })

  it('is registered', () => {
    expectInfo(autoencoderAlgorithms, 'algorithm')
    expect(autoencoderFunctions.semiSupervisedVaeRun.info.role).toBe('simulation')
  })
})
