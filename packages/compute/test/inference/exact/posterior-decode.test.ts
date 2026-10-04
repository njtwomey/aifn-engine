import { describe, expect, it } from 'vitest'
import { chainForwardBackward, chainViterbi, posteriorDecode } from 'aifn-compute/inference/exact'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'

describe('posteriorDecode', () => {
  it('is the argmax of the enumerated marginals, and its expected accuracy is Σ_n max_k P(y_n = k | x)', () => {
    const N = 5
    const K = 3
    for (let seed = 0; seed < 5; seed++) {
      const U = toFlat(uniform(child(stream(seed), 'u'), -2, 2, { shape: [N * K] }))
      const P = toFlat(uniform(child(stream(seed), 'p'), -2, 2, { shape: [K * K] }))
      const marg = new Float64Array(N * K)
      let Z = 0
      for (let code = 0; code < K ** N; code++) {
        const y = Array.from({ length: N }, (_, n) => Math.floor(code / K ** n) % K)
        const s = Math.exp(y.reduce((a, k, n) => a + U[n * K + k] + (n > 0 ? P[y[n - 1] * K + k] : 0), 0))
        Z += s
        y.forEach((k, n) => (marg[n * K + k] += s))
      }
      const fb = chainForwardBackward(fromData(Float64Array.from(U), [N, K]), fromData(Float64Array.from(P), [K, K]))
      const d = posteriorDecode(fb.marginals)
      let expected = 0
      for (let n = 0; n < N; n++) {
        const row = Array.from({ length: K }, (_, k) => marg[n * K + k] / Z)
        const best = row.indexOf(Math.max(...row))
        expect(toFlat(d.path)[n]).toBe(best)
        expect(toFlat(d.confidence)[n]).toBeCloseTo(row[best], 10)
        expected += row[best]
      }
      expect(d.expectedCorrect).toBeCloseTo(expected, 10)
    }
  })

  it('can differ from Viterbi, and can take a transition of zero probability', () => {
    // Three paths of two positions have all the mass: (0, 0) 0.35, (1, 1) 0.33 and (1, 2) 0.32. Viterbi takes (0, 0);
    // the marginals favour label 1 first (0.65) and label 0 second (0.35), and 1 → 0 has probability zero.
    const L = Math.log
    const U = fromData(new Float64Array(6), [2, 3])
    const I = -Infinity
    const T = fromData(Float64Array.from([L(0.35), I, I, I, L(0.33), L(0.32), I, I, I]), [3, 3])
    const v = Array.from(toFlat(chainViterbi(U, T).path))
    const d = posteriorDecode(chainForwardBackward(U, T).marginals)
    expect(v).toEqual([0, 0])
    expect(Array.from(toFlat(d.path))).toEqual([1, 0])
    expect(toFlat(T)[1 * 3 + 0]).toBe(-Infinity)
    expect(d.expectedCorrect).toBeCloseTo(0.65 + 0.35, 10)
  })
})
