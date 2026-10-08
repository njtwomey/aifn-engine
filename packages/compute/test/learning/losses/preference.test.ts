/**
 * Preference losses against their papers' formulas evaluated directly; gradients against finite differences; batches,
 * reductions and the options' edge cases; registration in the `preference` family.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { get, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { dpo, ipo, kto, listLosses, orpo, simpo } from 'aifn-compute/learning/losses'

const sigma = (z: number) => 1 / (1 + Math.exp(-z))
const logSigma = (z: number) => Math.log(sigma(z))

/** Central differences of a scalar function of a few numbers. */
function numericGrad(f: (v: number[]) => number, at: number[], h = 1e-6): number[] {
  return at.map((_, i) => {
    const up = at.slice()
    const down = at.slice()
    up[i] += h
    down[i] -= h
    return (f(up) - f(down)) / (2 * h)
  })
}

const [lw, ll, rw, rl] = [-3.2, -4.1, -3.5, -3.9]
const h = lw - rw - (ll - rl)

describe('dpo', () => {
  it('is −log σ(βh), and with label smoothing ε the conservative mixture', () => {
    expect(dpo(lw, ll, rw, rl, { beta: 0.3 })).toBeCloseTo(-logSigma(0.3 * h), 12)
    expect(dpo(lw, ll, rw, rl, { beta: 0.3, labelSmoothing: 0.1 })).toBeCloseTo(
      -0.9 * logSigma(0.3 * h) - 0.1 * logSigma(-0.3 * h),
      12,
    )
  })
  it('is exact for large margins and has the gradient of its formula', () => {
    expect(dpo(800, 0, 0, 0, { beta: 1 })).toBe(0)
    expect(dpo(-800, 0, 0, 0, { beta: 1 })).toBeCloseTo(800, 9)
    const f = (v: number[]) => -logSigma(0.5 * (v[0] - rw - (v[1] - rl)))
    const g = toFlat(
      grad((lp: Value) => dpo(get(lp, 0), get(lp, 1), rw, rl, { beta: 0.5 }))(tensor([lw, ll])) as Tensor,
    )
    numericGrad(f, [lw, ll]).forEach((v, i) => expect(g[i]).toBeCloseTo(v, 6))
  })
  it('reduces over a batch', () => {
    const W = tensor([-1, -2, -3])
    const L = tensor([-2, -2, -1])
    const each = toFlat(dpo(W, L, 0, 0, { beta: 1, reduction: 'none' }) as never)
    expect(Array.from(each)).toEqual([1, 0, -2].map((m) => -logSigma(m)).map((v) => expect.closeTo(v, 12)))
    expect(dpo(W, L, 0, 0, { beta: 1 })).toBeCloseTo(each.reduce((a, b) => a + b, 0) / 3, 12)
    expect(dpo(W, L, 0, 0, { beta: 1, reduction: 'sum' })).toBeCloseTo(
      each.reduce((a, b) => a + b, 0),
      12,
    )
  })
  it('rejects a non-positive beta and a smoothing of a half or more', () => {
    expect(() => dpo(0, 0, 0, 0, { beta: 0 })).toThrow(/beta/)
    expect(() => dpo(0, 0, 0, 0, { beta: 1, labelSmoothing: 0.5 })).toThrow(/labelSmoothing/)
  })
})

describe('ipo', () => {
  it('is (h − 1/(2τ))² with its minimum at the target margin', () => {
    expect(ipo(lw, ll, rw, rl, { tau: 0.2 })).toBeCloseTo((h - 2.5) ** 2, 12)
    expect(ipo(2.5, 0, 0, 0, { tau: 0.2 })).toBe(0)
  })
})

describe('kto', () => {
  it('is λ_D(1 − σ(β(r − z₀))) for desirable and λ_U(1 − σ(β(z₀ − r))) for undesirable responses', () => {
    const r = lw - rw
    expect(kto(lw, rw, true, 0.1, { beta: 2, lambdaD: 1.5 })).toBeCloseTo(1.5 * (1 - sigma(2 * (r - 0.1))), 12)
    expect(kto(lw, rw, false, 0.1, { beta: 2, lambdaU: 0.7 })).toBeCloseTo(0.7 * (1 - sigma(2 * (0.1 - r))), 12)
  })
  it('takes one label per example', () => {
    const each = toFlat(
      kto(tensor([0.5, 0.5]), 0, [1, 0], 0, { beta: 1, lambdaD: 2, lambdaU: 3, reduction: 'none' }) as never,
    )
    expect(each[0]).toBeCloseTo(2 * (1 - sigma(0.5)), 12)
    expect(each[1]).toBeCloseTo(3 * (1 - sigma(-0.5)), 12)
  })
})

describe('simpo', () => {
  it('is −log σ(β(ℓ̄_w − ℓ̄_l) − γ)', () => {
    expect(simpo(-1, -1.2, { beta: 2, gamma: 0.5 })).toBeCloseTo(-logSigma(2 * 0.2 - 0.5), 12)
    expect(simpo(-1, -1.2, { beta: 2 })).toBeCloseTo(-logSigma(0.4), 12)
  })
})

describe('orpo', () => {
  it('is NLL_w − λ log σ(log odds_w − log odds_l) with odds of the length-averaged likelihood', () => {
    const odds = (a: number) => Math.exp(a) / (1 - Math.exp(a))
    const ratio = Math.log(odds(-0.5)) - Math.log(odds(-1.5))
    expect(orpo(-0.5, -1.5, 0.5, { lambda: 0.1 })).toBeCloseTo(0.5 - 0.1 * logSigma(ratio), 12)
  })
  it('keeps the log-odds accurate for a likelihood near 1 or near 0', () => {
    // log odds(log p) for p = 1 − 1e-12 is about 27.6; for p = e^{−50} it is about −50.
    const near1 = Math.log1p(-1e-12)
    expect(orpo(near1, -50, 0, { lambda: 1 })).toBeCloseTo(-logSigma(Math.log((1 - 1e-12) / 1e-12) + 50), 6)
  })
})

it('registers every preference loss in its family, with its note', () => {
  const keys = listLosses({ family: 'preference' }).map((l) => l.info.key)
  expect(keys).toEqual(['dpo', 'ipo', 'kto', 'simpo', 'orpo'])
  for (const l of listLosses({ family: 'preference' })) {
    expect(l.info.inputs).toBe('log-probabilities')
    expect(l.info.notes).toContain('direct-preference-optimisation')
  }
})
