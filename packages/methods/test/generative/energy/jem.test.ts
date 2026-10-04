import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { add, fromData, logsumexp, neg, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { annulus, moons } from 'aifn-methods/data/synthetic'
import {
  classEnergy,
  classifier,
  classifierEnergy,
  jemRun,
  logitShift,
  classifierLogits,
  type JemRun,
} from 'aifn-methods/generative/energy'

const net = classifier(2, 3, { hidden: [8] })
const params = net.layer.init(stream(1))
const x = fromData(Float64Array.of(0.3, -0.2, 1.5, 0.7, -2, 1), [3, 2])
const flat = (t: unknown) => Array.from(toFlat(unwrap(t as Tensor) as Tensor))

describe('a classifier as an energy-based model', () => {
  it('E(x) = −logsumexp f(x), and −f(x)[y] per class', () => {
    const logits = flat(classifierLogits(net, params, x))
    const e = flat(classifierEnergy(net, params, x))
    const e1 = flat(classEnergy(net, params, x, 1))
    for (let i = 0; i < 3; i++) {
      const row = logits.slice(3 * i, 3 * i + 3)
      const m = Math.max(...row)
      expect(e[i]).toBeCloseTo(-(m + Math.log(row.reduce((s, v) => s + Math.exp(v - m), 0))), 12)
      expect(e1[i]).toBeCloseTo(-row[1], 12)
    }
  })

  it('law: a logit shift c(x) leaves p(y | x) unchanged and moves the energy by −c(x)', () => {
    const logits = unwrap(classifierLogits(net, params, x)) as Tensor
    for (const kind of ['radial', 'tilt', 'bump'] as const) {
      const c = logitShift({ kind, amount: 1.7 }, x)
      const shifted = add(logits, fromData(c, [3, 1])) as Tensor
      const p0 = flat(softmax(logits))
      const p1 = flat(softmax(shifted))
      p0.forEach((v, k) => expect(p1[k]).toBeCloseTo(v, 12))
      const e0 = flat(neg(logsumexp(logits, -1)))
      const e1 = flat(neg(logsumexp(shifted, -1)))
      e0.forEach((v, i) => expect(e1[i]).toBeCloseTo(v - c[i], 12))
    }
    expect(Array.from(logitShift({ kind: 'none', amount: 3 }, x))).toEqual([0, 0, 0])
  })

  it('jemRun trains both models from the same start and reports their checkpoints', () => {
    const data = moons(stream(0), { n: 60, spacing: 'random' })
    const test = moons(stream(1), { n: 40, spacing: 'random' })
    const ood = annulus(stream(2), { n: 20 })
    let last: JemRun | undefined
    for (const s of jemRun(data, test, ood, {
      steps: 6,
      hidden: [8],
      checkpoints: 2,
      grid: 6,
      samples: 8,
      sampleSteps: 3,
      langevinSteps: 3,
      bufferSize: 20,
      batchSize: 16,
    }))
      last = s
    expect(last!.finished).toBe(true)
    const [ce, jem] = [last!.crossEntropy, last!.jem]
    expect(ce.checkpoints.map((c) => c.step)).toEqual([0, 3, 6])
    // The same initial parameters: identical checkpoints at step 0.
    expect(Array.from(ce.checkpoints[0].logits)).toEqual(Array.from(jem.checkpoints[0].logits))
    expect(ce.contrastive.every(Number.isNaN)).toBe(true)
    expect(jem.contrastive.every(Number.isFinite)).toBe(true)
    expect(last!.trueLogDensity!.length).toBe(36)
    expect(jem.checkpoints[2].conditional.length).toBe(2)
    for (const c of [...ce.checkpoints, ...jem.checkpoints]) {
      expect(c.ece).toBeGreaterThanOrEqual(0)
      expect(c.oodAuroc).toBeGreaterThanOrEqual(0)
      expect(c.oodAuroc).toBeLessThanOrEqual(1)
    }
  })
})
