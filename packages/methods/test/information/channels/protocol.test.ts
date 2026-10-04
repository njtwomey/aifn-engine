import { describe, expect, it } from 'vitest'
import { blahutArimotoCapacity, blahutArimotoRateDistortion } from 'aifn-methods/information/channels'
import { run } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

// Binary symmetric channel with crossover 0.1: C = ln 2 − H(0.1) nats.
const bsc = [
  [0.9, 0.1],
  [0.1, 0.9],
]
const h = (p: number) => -p * Math.log(p) - (1 - p) * Math.log(1 - p)

describe('Blahut–Arimoto', () => {
  it('capacity: reaches ln 2 − H(0.1) from a skewed start and follows the trace protocol', () => {
    const alg = blahutArimotoCapacity(bsc)
    const s = run(alg, { initial: [0.9, 0.1] }, 500)
    expect(s.converged).toBe(true)
    expect(s.lower).toBeLessThanOrEqual(s.upper + 1e-15)
    expect(s.lower).toBeCloseTo(Math.log(2) - h(0.1), 9)
    expectProtocol(alg, { initial: [0.9, 0.1] }, { n: 8, record: { lower: (st) => st.lower } })
  })
  it('rate–distortion: a Bernoulli(½) source under Hamming distortion lies on R(D) = ln 2 − H(D)', () => {
    const alg = blahutArimotoRateDistortion(
      [0.5, 0.5],
      [
        [0, 1],
        [1, 0],
      ],
      2,
    )
    const s = run(alg, undefined, 2000)
    expect(s.converged).toBe(true)
    expect(s.rate).toBeCloseTo(Math.log(2) - h(s.distortion), 8)
    expectProtocol(alg, undefined, { n: 8, record: { rate: (st) => st.rate } })
  })
})
