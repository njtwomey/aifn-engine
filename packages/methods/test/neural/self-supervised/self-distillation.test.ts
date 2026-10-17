/**
 * aifn-methods/neural/self-supervised: `selfDistillationTraining` (no site check; tests written from the code: the
 * teacher is the EMA of the student, the centre the EMA of the teacher's logits, and the collapse diagnostics obey
 * $\expect[\entropy(P_t)] \le \entropy(\expect P_t) \le \log K$).
 */
import { describe, expect, it } from 'vitest'
import { selfDistillationTraining, selfSupervisedAlgorithms } from 'aifn-methods/neural/self-supervised'
import { normal, stream } from 'aifn-compute/foundation/random'
import { float64Data, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { expectInfo } from '../../registry'

const x = normal(stream('dino/x'), 0, 1, { shape: [24, 2] }) as Tensor
const net = Mlp([2, 8, 4], { activation: 'tanh', init: xavierUniform() })
const start = { params: net.init(stream('dino/init')) }
const logitsOf = (params: typeof start.params) => float64Data(net.apply(params, x) as Tensor)

describe('selfDistillationTraining', () => {
  it('a teacher momentum of 0 makes the teacher the student', () => {
    const s = run(selfDistillationTraining(x, { model: net, teacherMomentum: 0 }), start, 2)
    expect(Array.from(logitsOf(s.teacher))).toEqual(Array.from(logitsOf(s.params)))
  })

  it('the teacher starts as the student and the centre at zero; one step moves the centre by (1 − m) of the mean', () => {
    const alg = selfDistillationTraining(x, { model: net, jitter: 0, centreMomentum: 0.75 })
    const s0 = run(alg, start, 0)
    expect(Array.from(s0.centre)).toEqual([0, 0, 0, 0])
    expect(s0.teacher).toBe(s0.params)
    const s1 = run(alg, start, 1)
    // Without jitter both views are x, so the centre is (1 − m) times the column means of the initial logits.
    const z = logitsOf(start.params)
    for (let k = 0; k < 4; k++) {
      let m = 0
      for (let i = 0; i < 24; i++) m += z[i * 4 + k] / 24
      expect(s1.centre[k]).toBeCloseTo(0.25 * m, 12)
    }
  })

  it('the diagnostics are consistent: E[H(P_t)] ≤ H(E[P_t]) ≤ log K, rows sum to 1, KL ≥ 0', () => {
    for (const options of [{}, { centring: false, teacherTemperature: 0.01 }, { teacherTemperature: 0.1 }]) {
      const s = run(selfDistillationTraining(x, { model: net, batch: 16, ...options }), start, 5)
      expect(Number.isFinite(s.loss)).toBe(true)
      expect(s.conditionalEntropy).toBeLessThanOrEqual(s.marginalEntropy + 1e-12)
      expect(s.marginalEntropy).toBeLessThanOrEqual(Math.log(4) + 1e-12)
      expect(s.teacherStudentKl).toBeGreaterThanOrEqual(-1e-12)
      for (let i = 0; i < 24; i++) {
        let row = 0
        for (let k = 0; k < 4; k++) row += s.teacherProbabilities[i * 4 + k]
        expect(row).toBeCloseTo(1, 12)
        expect(s.assignment[i]).toBeGreaterThanOrEqual(0)
      }
    }
  })

  it('is registered', () => {
    expectInfo(selfSupervisedAlgorithms, 'algorithm')
  })
})
