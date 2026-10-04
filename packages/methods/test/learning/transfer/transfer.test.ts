import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, sum, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { labelShiftDomains, rotatingTasks, shiftedMoons } from 'aifn-methods/data/synthetic'
import {
  blackBoxShiftEstimate,
  continualRun,
  coralLoss,
  domainAdaptationRun,
  gradientReversal,
  mamlRun,
  mmdSquared,
  priorShiftEm,
} from 'aifn-methods/learning/transfer'

const num = (v: Value) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

describe('alignment penalties', () => {
  const a = fromData(Float64Array.from([0, 0, 1, 0, 0, 1, 1, 1]), [4, 2])
  const b = fromData(Float64Array.from([3, 3, 4, 3, 3, 4, 4, 4]), [4, 2])
  it('MMD² is 0 for identical samples and positive for shifted ones; CORAL ignores a translation', () => {
    expect(num(mmdSquared(a, a))).toBeCloseTo(0, 12)
    expect(num(mmdSquared(a, b))).toBeGreaterThan(0.5)
    expect(num(coralLoss(a, b, { source: 4, target: 4, dimension: 2 }))).toBeCloseTo(0, 12)
  })
  it('the gradient-reversal layer is the identity forward and −λ backward', () => {
    expect(Array.from(toFlat(gradientReversal(a, 0.7) as Tensor))).toEqual(Array.from(toFlat(a)))
    const g = grad((x: Value) => sum(gradientReversal(x, 0.7)))(a) as Tensor
    for (const v of toFlat(g)) expect(v).toBeCloseTo(-0.7, 12)
  })
})

describe('label shift', () => {
  it('BBSE and EM recover the target priors from a fixed source classifier', () => {
    const d = labelShiftDomains(stream(1), { n: 4000, separation: 4 })
    const k = d.classes
    // A nearest-class-mean classifier with softmax posteriors, fitted on the source.
    const xs = toFlat(d.source.x)
    const ys = Int32Array.from(toFlat(d.source.y))
    const means = new Float64Array(2 * k)
    const counts = new Float64Array(k)
    for (let i = 0; i < ys.length; i++) {
      means[2 * ys[i]] += xs[2 * i]
      means[2 * ys[i] + 1] += xs[2 * i + 1]
      counts[ys[i]]++
    }
    for (let c = 0; c < k; c++) for (let j = 0; j < 2; j++) means[2 * c + j] /= counts[c]
    const posterior = (x: ArrayLike<number>) => {
      const n = x.length / 2
      const out = new Float64Array(n * k)
      for (let i = 0; i < n; i++) {
        let z = 0
        for (let c = 0; c < k; c++) {
          const e = Math.exp(-0.5 * ((x[2 * i] - means[2 * c]) ** 2 + (x[2 * i + 1] - means[2 * c + 1]) ** 2))
          out[i * k + c] = e
          z += e
        }
        for (let c = 0; c < k; c++) out[i * k + c] /= z
      }
      return out
    }
    const argmax = (p: Float64Array) =>
      Int32Array.from({ length: p.length / k }, (_, i) => {
        let best = 0
        for (let c = 1; c < k; c++) if (p[i * k + c] > p[i * k + best]) best = c
        return best
      })
    const ps = posterior(xs)
    const pt = posterior(toFlat(d.target.x))
    const bbse = blackBoxShiftEstimate(ys, argmax(ps), argmax(pt), k)
    const em = priorShiftEm(fromData(pt, [pt.length / k, k]), [1 / 3, 1 / 3, 1 / 3])
    const truth = [0.6, 0.3, 0.1]
    truth.forEach((v, c) => {
      expect(bbse.priors[c]).toBeCloseTo(v, 1)
      expect(em.priors[c]).toBeCloseTo(v, 1)
    })
  })
})

describe('adaptation, continual and meta-learning runs', () => {
  it('DANN beats source-only training on rotated moons', () => {
    const pair = shiftedMoons(stream(1), { amount: 0.8 })
    const last = (method: 'source-only' | 'dann') => {
      let r: { checkpoints: { targetAccuracy: number }[] } | undefined
      for (const s of domainAdaptationRun(pair, { method, steps: 1200, checkpoints: 2 })) r = s
      return r!.checkpoints.at(-1)!.targetAccuracy
    }
    expect(last('dann')).toBeGreaterThan(last('source-only') + 0.15)
  })
  it('EWC and replay forget the first task less than naive fine-tuning', () => {
    const tasks = rotatingTasks(stream(2), { tasks: 3 })
    const first = (method: 'naive' | 'ewc' | 'replay') => {
      let r: { accuracy: number[][] } | undefined
      for (const s of continualRun(tasks, { method })) r = s
      return r!.accuracy[0].at(-1)!
    }
    const naive = first('naive')
    expect(first('ewc')).toBeGreaterThan(naive + 0.2)
    expect(first('replay')).toBeGreaterThan(naive + 0.2)
  })
  it('MAML adapts better than the pretrained baseline after a few steps', () => {
    let r: { checkpoints: { mamlCurve: Float64Array; pretrainedCurve: Float64Array }[] } | undefined
    for (const s of mamlRun({ steps: 400, checkpoints: 1, seed: 1 })) r = s
    const c = r!.checkpoints.at(-1)!
    expect(c.mamlCurve[5]).toBeLessThan(c.mamlCurve[0])
    expect(c.mamlCurve[5]).toBeLessThan(c.pretrainedCurve[5])
  })
})
