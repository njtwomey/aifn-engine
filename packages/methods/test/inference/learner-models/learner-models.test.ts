/**
 * Learner models (IRT, linear KTM, IRT-ZILM): the likelihood against a hand computation, the reduction laws (zero
 * inflation off is IRT; KTM without context is IRT), gradients against central finite differences, recovery of known
 * parameters on simulated learners, and the equity and score measures by hand.
 */

import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { correlation } from 'aifn-compute/probability/stats'
import { learnerResponses } from 'aifn-methods/data/synthetic'
import {
  abilityEquity,
  fitLearnerModel,
  learnerEquitySweep,
  learnerModelRun,
  learnerObjective,
  predictLearner,
  responseScores,
  structuralZeroPosterior,
  zilmProbability,
} from 'aifn-methods/inference/learner-models'

const num = (v: Value) => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

// Three students, two items; student 1 has condition 0, student 2 condition 1. Item features: [read, digits].
const responses = tensor([
  [1, 0],
  [0, NaN],
  [0, 1],
])
const conditions = tensor([
  [0, 0],
  [1, 0],
  [0, 1],
])
const itemFeatures = tensor([
  [1, 0],
  [0, 1],
])
// x = [θ (3), b (2), log a (2), w₀, W (2 × 2)].
const theta = [0.4, -0.3, 1.1]
const b = [0.2, -0.5]
const logA = [0.1, -0.2]
const w0 = -2
const W = [0.8, -0.1, 0.3, 1.5]

describe('learner-model likelihoods', () => {
  it('IRT-ZILM probability and the structural-zero posterior, by hand', () => {
    expect(zilmProbability(0.5, 1.5, -0.2, 0.3)).toBeCloseTo(0.7 * sigmoid(1.5 * 0.7), 14)
    expect(zilmProbability(0.5, 1.5, -0.2)).toBeCloseTo(sigmoid(1.05), 14)
    // P(zero) = π + (1 − π)(1 − p); the structural share is π / P(zero).
    expect(structuralZeroPosterior(0.2, 0.9)).toBeCloseTo(0.2 / (0.2 + 0.8 * 0.1), 14)
    expect(structuralZeroPosterior(0, 0.4)).toBe(0)
    expect(structuralZeroPosterior(1, 0.4)).toBe(1)
  })

  it('IRT-ZILM log-likelihood equals the hand sum of Eqn (1)', () => {
    const f = learnerObjective(responses, { model: 'zilm', conditions, itemFeatures })
    const x = tensor([...theta, ...b, ...logA, w0, ...W])
    const y = toFlat(responses)
    const zc = toFlat(conditions)
    const xc = toFlat(itemFeatures)
    let want = 0
    for (let s = 0; s < 3; s++)
      for (let i = 0; i < 2; i++) {
        const v = y[s * 2 + i]
        if (!Number.isFinite(v)) continue
        let eta = w0
        for (let k = 0; k < 2; k++) for (let g = 0; g < 2; g++) eta += zc[s * 2 + k] * xc[i * 2 + g] * W[g * 2 + k]
        const pi = sigmoid(eta)
        const p1 = zilmProbability(theta[s], Math.exp(logA[i]), b[i], pi)
        want += Math.log(v === 1 ? p1 : 1 - p1)
      }
    expect(f.n).toBe(5)
    expect(num(f.logLikelihood(x))).toBeCloseTo(want, 12)
  })

  it('reduces to IRT when the zero inflation is off, and KTM without context is IRT', () => {
    const irt = learnerObjective(responses, { model: 'irt' })
    const ktm = learnerObjective(responses, { model: 'ktm', conditions, itemFeatures })
    const zilm = learnerObjective(responses, { model: 'zilm', conditions, itemFeatures })
    const base = [...theta, ...b, ...logA]
    const lIrt = num(irt.logLikelihood(tensor(base)))
    expect(num(ktm.logLikelihood(tensor([...base, 0, 0, 0, 0])))).toBeCloseTo(lIrt, 13)
    // π = σ(−60) ≈ 1e-26: the zero-inflated Bernoulli is the Bernoulli.
    expect(num(zilm.logLikelihood(tensor([...base, -60, 0, 0, 0, 0])))).toBeCloseTo(lIrt, 13)
    // And the same for the fits: zero inflation that the data never asks for stays near zero.
    const d = learnerResponses(stream('zilm/no-inflation'), {
      students: 150,
      items: 20,
      attempts: 20,
      inflation: { dyslexia: 0.001, dyscalculia: 0.001, spd: 0.001 },
      baseRate: 0.001,
      guessing: 0,
    })
    const a = fitLearnerModel(d.responses, { model: 'irt' })
    const z = fitLearnerModel(d.responses, { model: 'zilm', conditions: d.conditions, itemFeatures: d.itemFeatures })
    expect(sigmoid(z.intercept)).toBeLessThan(0.02)
    expect(correlation(Array.from(a.ability), Array.from(z.ability))).toBeGreaterThan(0.995)
  })

  it('has gradients that match central finite differences', () => {
    for (const model of ['irt', 'ktm', 'zilm'] as const) {
      const f = learnerObjective(responses, { model, conditions, itemFeatures })
      const x0 = Float64Array.from({ length: f.dim }, (_, k) => 0.3 * Math.sin(1.3 * k + 0.2))
      const { grad } = valueAndGrad(f.objective)(fromData(x0, [f.dim]))
      const g = toFlat(unwrap(grad as Value) as Tensor)
      const h = 1e-6
      for (let k = 0; k < f.dim; k++) {
        const up = Float64Array.from(x0)
        const down = Float64Array.from(x0)
        up[k] += h
        down[k] -= h
        const fd = (num(f.objective(fromData(up, [f.dim]))) - num(f.objective(fromData(down, [f.dim])))) / (2 * h)
        expect(Math.abs(g[k] - fd)).toBeLessThan(1e-6)
      }
    }
  })
})

describe('recovery on simulated learners', () => {
  const d = learnerResponses(stream('zilm/recovery'), {
    students: 400,
    items: 30,
    attempts: 30,
    guessing: 0,
    prevalence: { dyslexia: 0.2, dyscalculia: 0.15, spd: 0.2 },
  })
  const opts = { conditions: d.conditions, itemFeatures: d.itemFeatures }
  const irt = fitLearnerModel(d.responses, { model: 'irt' })
  const zilm = fitLearnerModel(d.responses, { model: 'zilm', ...opts, init: irt })

  it('IRT-ZILM recovers abilities, difficulties and π, without the bias against conditions', () => {
    expect(zilm.steps).toBeGreaterThan(10)
    expect(correlation(Array.from(zilm.ability), Array.from(d.ability))).toBeGreaterThan(0.93)
    expect(correlation(Array.from(zilm.difficulty), Array.from(d.difficulty))).toBeGreaterThan(0.97)
    const pred = predictLearner(zilm, d.conditions, d.itemFeatures)
    expect(correlation(Array.from(pred.pi), Array.from(d.pi))).toBeGreaterThan(0.9)
    const eZilm = abilityEquity(zilm.ability, d.ability, d.group)
    const eIrt = abilityEquity(irt.ability, d.ability, d.group)
    expect(eIrt.gap).toBeLessThan(-0.08)
    expect(Math.abs(eZilm.gap)).toBeLessThan(Math.abs(eIrt.gap) / 2)
    expect(eZilm.rmse).toBeLessThan(eIrt.rmse)
  })

  it('separates structural zeros from incorrect answers', () => {
    const pred = predictLearner(zilm, d.conditions, d.itemFeatures)
    const y = toFlat(d.responses)
    let hit = 0
    let pairs = 0
    const s: number[] = []
    const o: number[] = []
    y.forEach((v, k) => {
      if (v !== 0) return
      const r = structuralZeroPosterior(pred.pi[k], pred.base[k])
      ;(d.structural[k] ? s : o).push(r)
    })
    // AUROC by counting pairs.
    for (const a of s) for (const c of o) (pairs++, (hit += a > c ? 1 : a === c ? 0.5 : 0))
    expect(hit / pairs).toBeGreaterThan(0.85)
  })
})

describe('the simulator', () => {
  it('draws structural zeros at the rate π and keeps its truth consistent', () => {
    const d = learnerResponses(stream('zilm/sim'), { students: 2000, items: 40, attempts: 20 })
    const y = toFlat(d.responses)
    let n = 0
    let piSum = 0
    let structural = 0
    y.forEach((v, k) => {
      if (!Number.isFinite(v)) {
        expect(d.structural[k]).toBe(0)
        return
      }
      n++
      piSum += d.pi[k]
      structural += d.structural[k]
      if (d.structural[k]) expect(v).toBe(0)
    })
    expect(n).toBe(2000 * 20)
    expect(Math.abs(structural / n - piSum / n)).toBeLessThan(0.006)
    const z = toFlat(d.conditions)
    const prevalence = [0, 1, 2].map((k) => z.filter((_, m) => m % 3 === k).reduce((a, c) => a + c, 0) / 2000)
    expect(Math.abs(prevalence[0] - 0.1)).toBeLessThan(0.02)
    expect(Math.abs(prevalence[1] - 0.06)).toBeLessThan(0.02)
    expect(Math.abs(prevalence[2] - 0.11)).toBeLessThan(0.02)
    // A student without conditions has π = π₀ on every item.
    const none = Array.from(d.group).indexOf(0)
    for (let i = 0; i < 40; i++) expect(d.pi[none * 40 + i]).toBeCloseTo(0.02, 12)
  })

  it('gives π = r_k on a fully unsuitable item for a student with condition k alone', () => {
    const d = learnerResponses(stream('zilm/sim2'), { students: 400, items: 200, inflation: { dyslexia: 0.7 } })
    const p = Array.from(d.group).indexOf(1)
    for (let i = 0; i < 200; i++) {
      const u = d.unsuitability[i * 3]
      const eta = Math.log(0.02 / 0.98) + u * (Math.log(0.7 / 0.3) - Math.log(0.02 / 0.98))
      expect(d.pi[p * 200 + i]).toBeCloseTo(sigmoid(eta), 12)
      expect(u).toBeLessThanOrEqual(1 + 1e-12)
    }
  })
})

describe('equity and scores', () => {
  it('computes group bias, RMSE and the gap by hand', () => {
    const e = abilityEquity([1, 0, -1, 0.5], [0.5, 0, 0, 1], [0, 0, 1, 2])
    expect(e.groups[0].bias).toBeCloseTo(0.25, 14)
    expect(e.groups[0].rmse).toBeCloseTo(Math.sqrt(0.125), 14)
    expect(e.groups[1].bias).toBeCloseTo(-1, 14)
    expect(e.groups[2].bias).toBeCloseTo(-0.5, 14)
    expect(e.gap).toBeCloseTo(-0.75 - 0.25, 14)
  })

  it('scores probabilities on the masked, observed responses', () => {
    const s = responseScores([1, 0, NaN, 1, 0], [0.8, 0.4, 0.5, 0.3, 0.9], [1, 1, 1, 1, 0])
    expect(s.n).toBe(3)
    expect(s.accuracy).toBeCloseTo(2 / 3, 14)
    expect(s.nll).toBeCloseTo(-(Math.log(0.8) + Math.log(0.6) + Math.log(0.3)) / 3, 12)
    expect(s.brier).toBeCloseTo((0.04 + 0.16 + 0.49) / 3, 12)
    // TP 1, FP 0, FN 1: F₁ = 2/3.
    expect(s.f1).toBeCloseTo(2 / 3, 14)
  })
})

describe('streamed runs', () => {
  it('yields after every fit and ends with every model evaluated', () => {
    const d = learnerResponses(stream('zilm/run'), { students: 80, items: 15, attempts: 10 })
    const g = learnerModelRun(d, { seed: 3, maxSteps: 60 })
    const seen: number[] = []
    let r = g.next()
    while (!r.done) {
      seen.push(r.value.results.length)
      r = g.next()
    }
    expect(seen).toEqual([0, 1, 2, 3])
    expect(r.value.finished).toBe(true)
    expect(r.value.results.map((m) => m.model)).toEqual(['irt', 'ktm', 'zilm'])
    const zeros = r.value.results[2].zeros!
    expect(zeros.posterior.every((v) => v >= 0 && v <= 1)).toBe(true)
  })

  it('sweeps datasets', () => {
    const rates = [0.02, 0.5]
    const data = rates.map((rate) =>
      learnerResponses(stream('zilm/sweep'), {
        students: 60,
        items: 12,
        attempts: 8,
        inflation: { dyslexia: rate, dyscalculia: rate, spd: rate },
      }),
    )
    const g = learnerEquitySweep(data, rates, { models: ['irt', 'zilm'], maxSteps: 40 })
    let r = g.next()
    while (!r.done) r = g.next()
    expect(r.value.points.map((p) => p.rate)).toEqual(rates)
    expect(r.value.points[1].structuralShare.with).toBeGreaterThan(r.value.points[0].structuralShare.with)
  })
})
