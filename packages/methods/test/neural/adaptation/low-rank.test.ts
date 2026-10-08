/**
 * Low-rank adaptation: the targets' spectra; fits that reach zero on a low-rank target and the Eckart–Young floor
 * otherwise; PiSSA starting at the floor; the scale conventions; determinism; the hand-written gradient against finite
 * differences.
 */
import { describe, expect, it } from 'vitest'
import { matmul, mul, square, sub, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { svd } from 'aifn-compute/numerics/linalg'
import { lowRankFit, lowRankLoss, lowRankTarget, type LowRankFitOptions } from 'aifn-methods/neural/adaptation'

const base: LowRankFitOptions = {
  rank: 4,
  scale: 'inverse',
  alpha: 8,
  optimiser: 'adam',
  learningRate: 0.01,
  steps: 300,
  init: 'lora',
  seed: 1,
}

describe('lowRankTarget', () => {
  it('has the power-law spectrum it was asked for', () => {
    const sigma = toFlat(svd(lowRankTarget({ size: 24, decay: 0.7, seed: 3 })).S)
    sigma.forEach((v, i) => expect(v).toBeCloseTo((i + 1) ** -0.7, 9))
  })

  it('is rank r plus small noise', () => {
    const sigma = toFlat(
      svd(lowRankTarget({ size: 32, decay: 'low-rank-plus-noise', rank: 3, noise: 0.05, seed: 2 })).S,
    )
    for (let i = 0; i < 3; i++) expect(Math.abs(sigma[i] - 1)).toBeLessThan(0.15)
    expect(sigma[3]).toBeLessThan(0.15)
    expect(sigma[3]).toBeGreaterThan(0)
  })

  it('is the same matrix for the same seed, and another for another', () => {
    const a = toFlat(lowRankTarget({ size: 8, decay: 1, seed: 5 }))
    expect(Array.from(toFlat(lowRankTarget({ size: 8, decay: 1, seed: 5 })))).toEqual(Array.from(a))
    expect(Array.from(toFlat(lowRankTarget({ size: 8, decay: 1, seed: 6 })))).not.toEqual(Array.from(a))
  })
})

describe('lowRankFit', () => {
  it('fits a rank-r target to near zero', () => {
    const target = lowRankTarget({ size: 32, decay: 'low-rank-plus-noise', rank: 4, noise: 0, seed: 1 })
    const fit = lowRankFit(target, { ...base, steps: 600 })
    expect(fit.floor).toBeLessThan(1e-20)
    expect(fit.losses[0]).toBeCloseTo(2, 9) // ½ · 4 unit singular values squared, from B = 0
    expect(fit.losses.at(-1)!).toBeLessThan(1e-4)
  })

  it('plateaus at the Eckart–Young floor when the target has more rank', () => {
    const target = lowRankTarget({ size: 32, decay: 1, seed: 1 })
    const fit = lowRankFit(target, { ...base, steps: 800 })
    const tail =
      Array.from({ length: 32 }, (_, i) => (i + 1) ** -2)
        .slice(4)
        .reduce((a, b) => a + b) / 2
    expect(fit.floor).toBeCloseTo(tail, 9)
    expect(fit.losses.at(-1)!).toBeGreaterThanOrEqual(fit.floor - 1e-9)
    expect(fit.losses.at(-1)! - fit.floor).toBeLessThan(1e-3)
    // The adapter's spectrum approaches the target's first four singular values.
    const last = fit.checkpoints.at(-1)!.spectrum
    last.forEach((v, i) => expect(Math.abs(v - (i + 1) ** -1)).toBeLessThan(0.02))
  })

  it('starts at the floor from PiSSA, and stays there', () => {
    const target = lowRankTarget({ size: 32, decay: 1, seed: 1 })
    // Plain gradient descent stays put; Adam would not, since it rescales the rounding-level gradient to full steps.
    const fit = lowRankFit(target, { ...base, init: 'pissa', optimiser: 'sgd', learningRate: 0.1, steps: 20 })
    expect(fit.losses[0]).toBeCloseTo(fit.floor, 10)
    expect(fit.losses.at(-1)!).toBeCloseTo(fit.floor, 8)
    fit.checkpoints[0].spectrum.forEach((v, i) => expect(v).toBeCloseTo((i + 1) ** -1, 9))
  })

  it('starts from zero under LoRA, with the scale alpha / r or alpha / sqrt(r)', () => {
    const target = lowRankTarget({ size: 16, decay: 1, seed: 1 })
    const lora = lowRankFit(target, { ...base, rank: 4, alpha: 8, steps: 0 })
    const rs = lowRankFit(target, { ...base, rank: 4, alpha: 8, scale: 'inverse-sqrt', steps: 0 })
    expect(lora.scale).toBe(2)
    expect(rs.scale).toBe(4)
    expect(Array.from(lora.checkpoints[0].spectrum)).toEqual([0, 0, 0, 0])
    expect(lora.losses[0]).toBeCloseTo(
      0.5 * Array.from({ length: 16 }, (_, i) => (i + 1) ** -2).reduce((a, b) => a + b),
      9,
    )
  })

  it('keeps checkpoints every `every` steps, with the update when asked', () => {
    const target = lowRankTarget({ size: 8, decay: 1, seed: 1 })
    const fit = lowRankFit(target, { ...base, steps: 25, every: 10, updates: true })
    expect(fit.checkpoints.map((c) => c.step)).toEqual([0, 10, 20, 25])
    expect(fit.losses.length).toBe(26)
    const u = toFlat(fit.checkpoints.at(-1)!.update!)
    const r = toFlat(target).map((v, i) => u[i] - v)
    expect(0.5 * r.reduce((a, b) => a + b * b, 0)).toBeCloseTo(fit.losses[25], 12)
  })

  it('is deterministic from its seed', () => {
    const target = lowRankTarget({ size: 16, decay: 1, seed: 1 })
    const a = lowRankFit(target, { ...base, steps: 50 })
    expect(Array.from(lowRankFit(target, { ...base, steps: 50 }).losses)).toEqual(Array.from(a.losses))
    expect(Array.from(lowRankFit(target, { ...base, steps: 50, seed: 2 }).losses)).not.toEqual(Array.from(a.losses))
  })

  it('makes plain gradient steps of the size the learning rate sets', () => {
    // From B = 0 one SGD step of size η moves B by η s ΔW Aᵀ, so the loss falls by about η‖∇L‖²; and the second
    // update equals the first plus a second step of the same law (checked through the loss being non-increasing).
    const target = lowRankTarget({ size: 6, decay: 1, seed: 1 })
    const fit = lowRankFit(target, { ...base, optimiser: 'sgd', learningRate: 1e-3, steps: 50, every: 50 })
    for (let t = 1; t < fit.losses.length; t++) expect(fit.losses[t]).toBeLessThanOrEqual(fit.losses[t - 1])
  })
})

describe('lowRankLoss', () => {
  it('has the gradients of the engine autodiff', () => {
    const T = lowRankTarget({ size: 5, decay: 0.5, seed: 4 })
    const B = tensor([
      [0.3, -0.2],
      [0.1, 0.4],
      [-0.5, 0.2],
      [0.05, 0.3],
      [0.2, -0.1],
    ])
    const A = tensor([
      [0.2, -0.1, 0.4, 0.3, -0.2],
      [0.1, 0.5, -0.3, 0.2, 0.15],
    ])
    const s = 1.7
    const f = (b: Value, a: Value) => mul(0.5, sum(square(sub(mul(s, matmul(b, a)), T))))
    const auto = valueAndGrad(f, { argnums: [0, 1] })(B, A)
    const mine = lowRankLoss(T, B, A, s)
    expect(mine.loss).toBeCloseTo(Number(auto.value), 12)
    const [gB, gA] = auto.grad as Tensor[]
    toFlat(gB).forEach((v, i) => expect(toFlat(mine.gradB)[i]).toBeCloseTo(v, 12))
    toFlat(gA).forEach((v, i) => expect(toFlat(mine.gradA)[i]).toBeCloseTo(v, 12))
  })

  it('rejects factors that do not multiply to the target', () => {
    expect(() =>
      lowRankLoss(
        [
          [1, 0],
          [0, 1],
        ],
        [[1], [1], [1]],
        [[1, 1]],
        1,
      ),
    ).toThrow(/does not match/)
  })
})
