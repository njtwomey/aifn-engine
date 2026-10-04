import { describe, expect, it } from 'vitest'
import { pairedShapes } from 'aifn-methods/data/synthetic'
import {
  contrastiveAblation,
  contrastiveLoss,
  contrastiveTrainingRun,
  embed,
  retrieve,
  similarities,
  TwoTower,
  zeroShot,
  type ContrastiveSnapshot,
} from 'aifn-methods/neural/contrastive'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { infoNce } from 'aifn-compute/learning/losses'

const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

describe('two-tower model and loss', () => {
  const data = pairedShapes(stream('t'), { n: 16, size: 8 })
  const model = TwoTower({ inA: data.a.shape[1], inB: data.b.shape[1], hidden: 8, dim: 3 })
  const p = model.init(stream('init'), 0.2)

  it('embeds both views on the unit sphere', () => {
    for (const [x, view] of [
      [data.a, 'a'],
      [data.b, 'b'],
    ] as const) {
      const z = embed(model, p, x, view)
      expect(z.shape).toEqual([16, 3])
      const v = toFlat(z)
      for (let i = 0; i < 16; i++) expect(Math.hypot(v[3 * i], v[3 * i + 1], v[3 * i + 2])).toBeCloseTo(1, 12)
    }
  })

  it("is CLIP's symmetric InfoNCE at the model's temperature, with a gradient for log(1/τ)", () => {
    const za = model.encoderA.apply(p.a, data.a)
    const zb = model.encoderB.apply(p.b, data.b)
    const direct = num(infoNce(za, zb, { temperature: 0.2, symmetric: true }))
    expect(num(contrastiveLoss(model, p, data))).toBeCloseTo(direct, 10)
    expect(num(contrastiveLoss(model, p, data, { temperature: 0.5 }))).toBeCloseTo(
      num(infoNce(za, zb, { temperature: 0.5, symmetric: true })),
      10,
    )
    const lossOf = (q: unknown) => contrastiveLoss(model, q as typeof p, data)
    const g = valueAndGrad(lossOf)(p as unknown).grad as unknown as typeof p
    expect(Math.abs(toFlat(g.logScale)[0])).toBeGreaterThan(0)
  })
})

describe('retrieval and zero-shot helpers', () => {
  const z = tensor([
    [1, 0],
    [0, 1],
    [-1, 0],
  ]) as Tensor
  it('similarities are inner products of unit rows; retrieve ranks by them', () => {
    expect(Array.from(similarities(z, z))).toEqual([1, 0, -1, 0, 1, 0, -1, 0, 1])
    const r = retrieve([Math.SQRT1_2, Math.SQRT1_2], z, 2)
    expect(r.index).toEqual([0, 1])
    expect(r.similarity[0]).toBeCloseTo(Math.SQRT1_2, 12)
  })
  it('zero-shot picks the most similar prototype, among candidates when given', () => {
    const q = tensor([
      [0.9, 0.1],
      [-0.2, 0.9],
    ]) as Tensor
    expect(Array.from(zeroShot(q, z))).toEqual([0, 1])
    expect(Array.from(zeroShot(q, z, [1, 2]))).toEqual([1, 1])
  })
})

describe('training run', () => {
  it('starts at step 0, lowers the loss, aligns pairs and classifies seen classes zero-shot above chance', () => {
    const train = pairedShapes(stream('train'), { n: 800 })
    const test = pairedShapes(stream('test'), { n: 300, include: 'all' })
    const snaps: ContrastiveSnapshot[] = [
      ...contrastiveTrainingRun(train, test, { steps: 400, every: 100, dim: 8, seed: 3 }),
    ]
    const last = snaps.at(-1)!
    expect(snaps[0].step).toBe(0)
    expect(last.checkpoints.map((c) => c.step)).toEqual([0, 100, 200, 300, 400])
    expect(last.losses).toHaveLength(401)
    expect(last.temperatures).toHaveLength(401)
    const [first, end] = [last.checkpoints[0], last.checkpoints.at(-1)!]
    const early = last.losses.slice(0, 20).reduce((a, b) => a + b) / 20
    const late = last.losses.slice(-20).reduce((a, b) => a + b) / 20
    expect(late).toBeLessThan(0.6 * early)
    expect(end.alignment).toBeLessThan(first.alignment)
    // 24 classes: chance is 1/24.
    expect(end.zeroShotSeen).toBeGreaterThan(0.4)
    expect(end.retrievalTop1).toBeGreaterThan(first.retrievalTop1)
    // The learned temperature moves from its initial 0.1.
    expect(end.temperature).not.toBeCloseTo(0.1, 3)
    // Deterministic in its seed.
    const again = [...contrastiveTrainingRun(train, test, { steps: 400, every: 100, dim: 8, seed: 3 })].at(-1)!
    expect(again.losses).toEqual(last.losses)
  }, 60_000)

  it('the ablation yields one scored run per batch size and temperature, without parameters', () => {
    const train = pairedShapes(stream('train'), { n: 200 })
    const test = pairedShapes(stream('test'), { n: 100, include: 'all' })
    const runs = [
      ...contrastiveAblation(train, test, { batchSizes: [4, 16], temperatures: [0.1, 'learned'], steps: 20 }),
    ].at(-1)!
    expect(runs.map((r) => [r.batchSize, r.temperature])).toEqual([
      [4, 0.1],
      [4, 'learned'],
      [16, 0.1],
      [16, 'learned'],
    ])
    expect(runs[0].finalTemperature).toBe(0.1)
    expect('params' in runs[0]).toBe(false)
    for (const r of runs) expect(Number.isFinite(r.uniformity) && Number.isFinite(r.alignment)).toBe(true)
  }, 60_000)
})
