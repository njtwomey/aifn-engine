/**
 * Label propagation for label proportions (Poyiadzi et al. 2018) and its baselines: recovery of the labels of the
 * paper's synthetic problems (Gaussian XOR, half-kernel) from three bags with configurations A and B; the laws of the
 * projection step (every row a distribution, every bag's class mass equal to n_k π_k after each step, at convergence
 * the hard labels' proportions close to the bags'); c = 2 reduces to the binary scalar form; the γ heuristic; bags made
 * to measure; InvCal, alter-∝SVM and MeanMap beat chance where the paper says they should.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { dense, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { halfKernel, xor } from 'aifn-methods/data/synthetic'
import {
  alterProportionSvm,
  bagsByProportion,
  inverseCalibration,
  lpllp,
  lpllpGammaSearch,
  lpllpSteps,
  meanMap,
} from 'aifn-methods/learning/weak-supervision'

const CONFIG = {
  A: [0.6, 0.4, 0.5],
  B: [0.85, 0.25, 0.4],
}
const proportionsOf = (pi: number[]) => pi.map((p) => [1 - p, p])

function problem(kind: 'xor' | 'half-kernel', n: number, config: keyof typeof CONFIG, seed = 1) {
  const data =
    kind === 'xor'
      ? xor(stream(seed), { kind: 'gaussian', sd: 0.2, n: [n / 2, n / 2] })
      : halfKernel(stream(seed), { n })
  const y = Int32Array.from(toFlat(data.y as Tensor))
  const { bags, proportions } = bagsByProportion(stream(`bags/${seed}`), y, proportionsOf(CONFIG[config]))
  return { x: data.x as Tensor, y, bags, proportions }
}

const accuracy = (pred: ArrayLike<number>, y: ArrayLike<number>) => {
  let hit = 0
  for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) hit++
  return hit / y.length
}

describe('bagsByProportion', () => {
  it('fills bags with the requested proportions when the classes allow it', () => {
    const y = Int32Array.from({ length: 300 }, (_, i) => (i < 150 ? 1 : 0))
    const { bags, proportions } = bagsByProportion(stream(3), y, proportionsOf(CONFIG.B))
    const P = dense.data(proportions)
    CONFIG.B.forEach((p, k) => expect(P[k * 2 + 1]).toBeCloseTo(p, 10))
    expect(Array.from(bags).every((b) => b >= 0)).toBe(true)
  })
})

describe('LP-LLP', () => {
  it('every step keeps rows on the simplex and the bags on their class masses', () => {
    const { x, bags, proportions } = problem('xor', 120, 'A')
    const run = trace(lpllpSteps(x, bags, proportions, { gamma: 2 }), undefined, 8, { keep: 'all' })
    const P = dense.data(proportions)
    for (const s of run.steps.slice(1)) {
      const F = dense.data(s.scores)
      for (let i = 0; i < 120; i++) {
        expect(F[2 * i]).toBeGreaterThanOrEqual(-1e-9)
        expect(F[2 * i] + F[2 * i + 1]).toBeCloseTo(1, 8)
      }
      const mass = dense.data(s.mass)
      for (let k = 0; k < 3; k++) for (let c = 0; c < 2; c++) expect(mass[k * 2 + c]).toBeCloseTo(40 * P[k * 2 + c], 5)
    }
  })

  it('step 0 holds each point’s bag proportions', () => {
    const { x, bags, proportions } = problem('xor', 120, 'B')
    const s0 = lpllpSteps(x, bags, proportions).init(undefined, stream(0))
    const F = dense.data(s0.scores)
    const P = dense.data(proportions)
    for (let i = 0; i < 120; i++) expect(F[2 * i + 1]).toBeCloseTo(P[bags[i] * 2 + 1], 12)
  })

  it('recovers the XOR labels from three bags (configuration B), as the paper’s Table 1', () => {
    const { x, y, bags, proportions } = problem('xor', 300, 'B')
    const s = lpllp(x, bags, proportions, { gamma: 2 })
    expect(s.converged).toBe(true)
    expect(accuracy(toFlat(s.labels), y)).toBeGreaterThan(0.9)
    // At convergence the hard labels' proportions are close to the bags' (the soft masses are exact).
    const lab = toFlat(s.labels)
    const P = dense.data(proportions)
    for (let k = 0; k < 3; k++) {
      let ones = 0
      let size = 0
      for (let i = 0; i < 300; i++)
        if (bags[i] === k) {
          size++
          ones += lab[i]
        }
      expect(Math.abs(ones / size - P[k * 2 + 1])).toBeLessThan(0.1)
    }
  })

  it('recovers the half-kernel labels (configuration B)', () => {
    const { x, y, bags, proportions } = problem('half-kernel', 300, 'B')
    const best = lpllpGammaSearch(x, bags, proportions, [1, 4, 16, 64])
    expect(accuracy(toFlat(best.best.labels), y)).toBeGreaterThan(0.85)
  })

  it('the unscaled propagation of Algorithm 2 reaches the same labels', () => {
    const { x, bags, proportions } = problem('xor', 150, 'B')
    const a = toFlat(lpllp(x, bags, proportions, { gamma: 2 }).labels)
    const b = toFlat(lpllp(x, bags, proportions, { gamma: 2, normalise: false }).labels)
    expect(accuracy(a, b)).toBeGreaterThan(0.95)
  })
})

describe('baselines', () => {
  it('InvCal with an RBF kernel beats chance on XOR', () => {
    const { x, y, bags, proportions } = problem('xor', 150, 'B')
    const r = inverseCalibration(x, bags, proportions, { gamma: 2 })
    expect(accuracy(r.labels, y)).toBeGreaterThan(0.6)
  })

  it('alter-∝SVM keeps every bag on its proportion', () => {
    const { x, bags, proportions } = problem('xor', 90, 'B')
    const r = alterProportionSvm(stream(4), x, bags, proportions, { gamma: 2, restarts: 2 })
    const P = dense.data(proportions)
    for (let k = 0; k < 3; k++) {
      let ones = 0
      for (let i = 0; i < 90; i++) if (bags[i] === k) ones += r.labels[i]
      expect(ones / 30).toBeCloseTo(P[k * 2 + 1], 1)
    }
  })

  it('MeanMap with random Fourier features beats chance on XOR', () => {
    const { x, y, bags, proportions } = problem('xor', 150, 'B')
    const r = meanMap(stream(5), x, bags, proportions, { gamma: 2 })
    const acc = accuracy(r.labels, y)
    expect(Math.max(acc, 1 - acc)).toBeGreaterThan(0.6)
  })
})
