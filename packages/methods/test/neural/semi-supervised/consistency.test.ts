/**
 * aifn-methods/neural/semi-supervised: `consistencyTraining` and its rules, on the values of the site's self-check
 * (consistency-training.check.ts).
 */
import { describe, expect, it } from 'vitest'
import { moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import {
  annealingThreshold,
  consistencyTraining,
  flexMatchThresholds,
  predictProbabilities,
  rampUpWeight,
  semiSupervisedNetworkAlgorithms,
  semiSupervisedNetworkFunctions,
  sharpen,
  type ConsistencyMethod,
} from 'aifn-methods/neural/semi-supervised'
import { stream } from 'aifn-compute/foundation/random'
import { float64Data, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { expectInfo } from '../../registry'

describe('the rules', () => {
  it('sharpen: [0.6, 0.4] at T = 0.5 is [0.36, 0.16] / 0.52; T = 1 is the identity; T → 0 is one-hot', () => {
    const sh = sharpen([0.6, 0.4], 2, 0.5)
    expect(sh[0]).toBeCloseTo(0.36 / 0.52, 12)
    expect(sh[1]).toBeCloseTo(0.16 / 0.52, 12)
    const id = sharpen([0.2, 0.3, 0.5], 3, 1)
    expect(id[0]).toBeCloseTo(0.2, 12)
    expect(id[2]).toBeCloseTo(0.5, 12)
    expect(sharpen([0.2, 0.3, 0.5], 3, 0.01)[2]).toBeGreaterThan(0.999999)
  })

  it('sharpen refuses a row with no positive entry and a non-positive temperature', () => {
    expect(() => sharpen([0, 0, 0.5, 0.5], 2, 0.5)).toThrow(/row 0/)
    expect(() => sharpen([0.5, 0.5], 2, 0)).toThrow(/temperature/)
  })

  it('training-signal annealing: η = α(1 − 1/K) + 1/K', () => {
    expect(annealingThreshold('linear', 50, 100, 2)).toBeCloseTo(0.75, 12)
    expect(annealingThreshold('log', 0, 100, 4)).toBeCloseTo(0.25, 12)
    expect(annealingThreshold('exp', 100, 100, 10)).toBeCloseTo(1, 12)
    expect(annealingThreshold('log', 20, 100, 2)).toBeCloseTo((1 - Math.exp(-1)) * 0.5 + 0.5, 12)
    expect(annealingThreshold('none', 20, 100, 2)).toBe(1)
  })

  it('FlexMatch: counts [50, 25] of 100 at τ = 0.95 give [0.95, 0.3167], and the warm-up denominator', () => {
    const fm = flexMatchThresholds([50, 25], 100, 0.95)
    expect(fm[0]).toBeCloseTo(0.95, 12)
    expect(fm[1]).toBeCloseTo(0.95 / 3, 12)
    const b0 = 10 / 85
    expect(flexMatchThresholds([10, 5], 100, 0.95)[0]).toBeCloseTo((b0 / (2 - b0)) * 0.95, 12)
  })

  it('ramp-up: λe⁻⁵ at t = 0, λ from R on', () => {
    expect(rampUpWeight(2, 0, 10)).toBeCloseTo(2 * Math.exp(-5), 14)
    expect(rampUpWeight(2, 10, 10)).toBe(2)
  })
})

describe('consistencyTraining', () => {
  // Two moons, 4 labels per class, split as the site's consistency figures split (one global permutation).
  const d = moons(stream('check/moons'), { n: 100, noise: 0.1 })
  const y = Array.from(toFlat(d.y))
  const sp = semiSupervisedSplit(stream('check/split'), y, 4, { x: d.x, draw: 'global', strict: true })
  const problem = {
    labelledX: sp.labelledX!,
    labelledY: sp.labelledY,
    unlabelledX: sp.unlabelledX!,
    classes: 2,
    truth: sp.unlabelledTruth,
  }
  const net = Mlp([2, 8, 2], { activation: 'tanh', init: xavierUniform() })
  const start = { params: net.init(stream('check/init')) }
  const after = (method: ConsistencyMethod, steps: number, extra = {}) =>
    run(consistencyTraining(problem, { model: net, method, ...extra }), start, steps)

  it('the split has 4 labels per class and 92 unlabelled points', () => {
    expect(sp.labelledY.length).toBe(8)
    expect(float64Data(sp.unlabelledX!).length).toBe(92 * 2)
    expect(Array.from(sp.labelledY).filter((c) => c === 0).length).toBe(4)
  })

  it('Π-model with no noise has zero consistency loss', () => {
    expect(Math.abs(after({ kind: 'pi-model' }, 1, { weak: 0 }).unsupervisedLoss)).toBeLessThanOrEqual(1e-12)
  })

  it('mean teacher with decay 0 copies the student', () => {
    const mt = after({ kind: 'mean-teacher', decay: 0 }, 3)
    const pS = predictProbabilities(net, mt.params, problem.unlabelledX)
    const pT = predictProbabilities(net, mt.teacher!, problem.unlabelledX)
    pS.forEach((v, i) => expect(Math.abs(v - pT[i])).toBeLessThanOrEqual(1e-12))
  })

  it('temporal ensembling: no target at the first step, every point at the second', () => {
    const te1 = after({ kind: 'temporal-ensembling' }, 1)
    expect(te1.maskRate).toBe(0)
    expect(Array.from(te1.ensembleCounts).every((c) => c === 1)).toBe(true)
    expect(after({ kind: 'temporal-ensembling', momentum: 0.6 }, 2).maskRate).toBe(1)
  })

  it('FixMatch: τ > 1 masks every point (loss = supervised loss), τ = 0 keeps every point', () => {
    const fx = after({ kind: 'fixmatch', threshold: 1.01 }, 1)
    expect(fx.maskRate).toBe(0)
    expect(Math.abs(fx.loss - fx.supervisedLoss)).toBeLessThanOrEqual(1e-12)
    expect(after({ kind: 'fixmatch', threshold: 0 }, 1).maskRate).toBe(1)
  })

  it('VAT on a linear softmax model: perturbations of length ε parallel to w₁ − w₀', () => {
    const lin = Mlp([2, 2])
    const linStart = { params: lin.init(stream('check/linear')) }
    const vat = run(consistencyTraining(problem, { model: lin, method: { kind: 'vat', radius: 0.3 } }), linStart, 1)
    const W = float64Data((linStart.params[0] as { weight: Tensor }).weight)
    // The 2 × 2 weight may be stored [in, out] or [out, in]: test the class difference under both readings.
    const diffs = [
      [W[1] - W[0], W[3] - W[2]],
      [W[2] - W[0], W[3] - W[1]],
    ]
    const worst = diffs.map((diff) => {
      let w = 1
      for (let i = 0; i < 92; i++) {
        const r = [vat.perturbation[2 * i], vat.perturbation[2 * i + 1]]
        const len = Math.hypot(r[0], r[1])
        expect(Math.abs(len - 0.3)).toBeLessThanOrEqual(1e-9)
        w = Math.min(w, Math.abs(r[0] * diff[0] + r[1] * diff[1]) / (len * Math.hypot(diff[0], diff[1])))
      }
      return w
    })
    expect(Math.max(...worst)).toBeGreaterThan(0.999999)
  })

  it('every method runs minibatch steps with a finite loss and per-point diagnostics', () => {
    // The self-check runs 20 steps each; 5 keep the test fast and exercise every branch, the adaptive rules included.
    const methods: ConsistencyMethod[] = [
      { kind: 'supervised' },
      { kind: 'pi-model' },
      { kind: 'temporal-ensembling' },
      { kind: 'mean-teacher' },
      { kind: 'vat' },
      { kind: 'uda', annealing: 'log' },
      { kind: 'mixmatch' },
      { kind: 'fixmatch' },
      { kind: 'fixmatch', thresholds: 'flexmatch' },
      { kind: 'fixmatch', thresholds: 'freematch', momentum: 0.9 },
      { kind: 'fixmatch', thresholds: 'softmatch', momentum: 0.9 },
    ]
    for (const m of methods) {
      const s = after(m, 5, { batch: 32 })
      expect(Number.isFinite(s.loss), m.kind).toBe(true)
      expect(s.mask.length).toBe(92)
      expect(s.batch.length).toBe(32)
    }
  })

  it('is registered', () => {
    expectInfo(semiSupervisedNetworkAlgorithms, 'algorithm')
    expectInfo(semiSupervisedNetworkFunctions, 'function')
  })
})
