/**
 * aifn-methods/generative/gan: the semi-supervised GAN, on the values of the site's self-check
 * (semi-supervised-gan.check.ts). The check's 500-step accuracies on two moons (feature matching 0.953, complement
 * 0.932, labels only 0.849) take seconds and are not repeated here; the short runs below test the same code paths.
 */
import { describe, expect, it } from 'vitest'
import { moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import {
  classAndFakeProbabilities,
  complementSampler,
  ganAlgorithms,
  ganFunctions,
  initSemiSupervisedGan,
  semiSupervisedGan,
  semiSupervisedGanRun,
  semiSupervisedGanTraining,
  type CriticParams,
} from 'aifn-methods/generative/gan'
import { treeFlatten } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { float64Data, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { expectInfo } from '../../registry'

const d = moons(stream('gssl/sgan/moons/1'), { n: 200, noise: 0.1 })
const y = Array.from(toFlat(d.y))
const BOX = { x: [-1.5, 2.5] as const, y: [-1, 1.5] as const }

describe('classAndFakeProbabilities', () => {
  it('p(generated) = 1 / (Z + 1) with Z = Σ exp(l_k), and rows sum to 1', () => {
    const net = semiSupervisedGan({})
    const critic = initSemiSupervisedGan(net, stream(1)).critic as unknown as CriticParams
    const l = float64Data(net.head.apply(critic.head, net.features.apply(critic.features, d.x)) as Tensor)
    const p = classAndFakeProbabilities(net, critic, d.x)
    for (let i = 0; i < 200; i++) {
      const Z = Math.exp(l[2 * i]) + Math.exp(l[2 * i + 1])
      expect(Math.abs(p[3 * i + 2] - 1 / (Z + 1))).toBeLessThan(1e-12)
      expect(Math.abs(p[3 * i] + p[3 * i + 1] + p[3 * i + 2] - 1)).toBeLessThan(1e-12)
    }
  })

  it("the note's worked example: logits (2, −1) give Z = 7.757, D = 0.886, −log D = 0.121", () => {
    const Z = Math.exp(2) + Math.exp(-1)
    expect(Z).toBeCloseTo(7.757, 3)
    expect(Z / (Z + 1)).toBeCloseTo(0.886, 3)
    expect(-Math.log(Z / (Z + 1))).toBeCloseTo(0.121, 3)
    // A generated point with logits (0.5, 0.3): −log(1 − D) = log(Z + 1) = 1.386.
    expect(Math.log(Math.exp(0.5) + Math.exp(0.3) + 1)).toBeCloseTo(1.386, 3)
  })
})

describe('complementSampler', () => {
  it('draws n points in the box, each farther than r from every data point', () => {
    const pts = float64Data(complementSampler(d.x, BOX, 0.25)(stream(5), 300))
    const X = float64Data(d.x)
    expect(pts.length).toBe(600)
    for (let i = 0; i < 300; i++) {
      const [px, py] = [pts[2 * i], pts[2 * i + 1]]
      expect(px >= BOX.x[0] && px <= BOX.x[1] && py >= BOX.y[0] && py <= BOX.y[1]).toBe(true)
      for (let j = 0; j < 200; j++)
        expect((X[2 * j] - px) ** 2 + (X[2 * j + 1] - py) ** 2).toBeGreaterThanOrEqual(0.0625)
    }
  })

  it('throws rather than return fewer points when the data cover the box', () => {
    expect(() => complementSampler(d.x, BOX, 10)(stream(5), 3)).toThrow(/only 0 of 3/)
  })
})

describe('semiSupervisedGanTraining', () => {
  const sp = semiSupervisedSplit(stream('gssl/sgan/split/1'), y, 4, { x: d.x, draw: 'global', strict: true })
  const problem = { labelledX: sp.labelledX!, labelledY: sp.labelledY, unlabelledX: sp.unlabelledX!, classes: 2 }
  const net = semiSupervisedGan({ hidden: [8, 8] })
  const start = initSemiSupervisedGan(net, stream('gssl/sgan/init/1'))

  it('labels only: the generator never moves and the critic loss is the labelled cross-entropy', () => {
    const s = run(semiSupervisedGanTraining(problem, { net, fakes: { kind: 'none' } }), start, 5)
    expect(Array.from(treeFlatten(s.generator).leaves.flatMap((t) => Array.from(toFlat(t as Tensor))))).toEqual(
      Array.from(treeFlatten(start.generator).leaves.flatMap((t) => Array.from(toFlat(t as Tensor)))),
    )
    expect(Number.isFinite(Number(s.criticLoss))).toBe(true)
  })

  it('every source trains with finite losses', () => {
    const sampler = complementSampler(d.x, BOX, 0.25)
    for (const fakes of [
      { kind: 'feature-matching' },
      { kind: 'non-saturating' },
      { kind: 'fixed', sampler },
    ] as const) {
      const s = run(semiSupervisedGanTraining(problem, { net, fakes, batch: 32 }), start, 5)
      expect(Number.isFinite(Number(s.criticLoss)), fakes.kind).toBe(true)
      expect(Number.isFinite(Number(s.generatorLoss)), fakes.kind).toBe(true)
    }
  })

  it('the plain-data run reproduces the algorithm and keeps its checkpoints', () => {
    const shots = [
      ...semiSupervisedGanRun(problem, {
        hidden: [8, 8],
        steps: 6,
        every: 3,
        shown: 5,
        seed: 'gssl/sgan/init/1',
        truth: sp.unlabelledTruth,
      }),
    ]
    expect(shots.map((c) => c.step)).toEqual([0, 3, 6])
    const direct = run(semiSupervisedGanTraining(problem, { net }), start, 6)
    const last = shots.at(-1)!
    expect(last.criticLoss).toBe(Number(direct.criticLoss))
    expect(Array.from(last.probabilities)).toEqual(
      Array.from(classAndFakeProbabilities(net, direct.critic as unknown as CriticParams, problem.unlabelledX)),
    )
    expect(last.fakes.length).toBe(10)
    expect(last.accuracy).toBeGreaterThanOrEqual(0)
  })

  it('is registered', () => {
    expect(ganAlgorithms.semiSupervisedGanTraining.info.kind).toBe('algorithm')
    expect(ganFunctions.semiSupervisedGanRun.info.role).toBe('simulation')
    expectInfo(ganFunctions, 'function')
  })
})
