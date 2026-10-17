/**
 * aifn-compute/nn/sequence `rgLru` and `rgLruLambda`: the gated-linear-recurrence note's four-token worked example, a
 * sequential loop on random gates, and $\sigma(\Lambda)^c$ returning the decay asked for.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { rgLru, rgLruLambda } from 'aifn-compute/nn/sequence'

const flat = (v: Value) => toFlat(v as Tensor)
const maxError = (got: number[], want: number[]) => Math.max(...got.map((g, i) => Math.abs(g - want[i])))
const sig = (z: number) => 1 / (1 + Math.exp(-z))
const logit = (p: number) => Math.log(p / (1 - p))

describe('RG-LRU', () => {
  it('reproduces the worked example: h = (1.9365, 1.7104, 1.5136, -0.5898)', () => {
    // The gates are set through their biases (W = 0), so the four tokens run as one-token calls carrying the state.
    const Lambda = rgLruLambda(tensor([0.25]))
    const x = [2, 0.5, 0.5, -1]
    const g = [1 - 1e-12, 0.1, 0.1, 1 - 1e-12]
    let h: Tensor = zeros([1])
    const hs: number[] = []
    const as: number[] = []
    for (let t = 0; t < 4; t++) {
      const out = rgLru(
        tensor([[x[t]]]),
        { Wa: zeros([1, 1]), ba: tensor([logit(g[t])]), Wx: zeros([1, 1]), bx: tensor([logit(g[t])]), Lambda },
        { initial: h },
      )
      h = tensor([flat(out.outputs)[0]])
      hs.push(toFlat(h)[0])
      as.push(flat(out.decay)[0])
    }
    expect(maxError(as, [0.25, 0.25 ** 0.1, 0.25 ** 0.1, 0.25])).toBeLessThan(1e-9)
    expect(maxError(hs, [1.93649, 1.71042, 1.51361, -0.58984])).toBeLessThan(2e-5)
  })

  it('equals a sequential loop on random gates (D = 3, T = 12)', () => {
    const T = 12
    const D = 3
    const s = stream(3)
    const x = normals(s, [T, D])
    const Wa = normals(s, [D, D])
    const Wx = normals(s, [D, D])
    const ba = normals(s, [D])
    const bx = normals(s, [D])
    const Lambda = rgLruLambda(tensor([0.9, 0.99, 0.999]))
    const out = rgLru(x, { Wa, ba, Wx, bx, Lambda })
    const [X, A, B, ra, rx, L] = [x, Wa, Wx, ba, bx, Lambda].map((v) => toFlat(v))
    const h = new Array<number>(D).fill(0)
    const want: number[] = []
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        let za = ra[d]
        let zx = rx[d]
        for (let e = 0; e < D; e++) {
          za += X[t * D + e] * A[e * D + d]
          zx += X[t * D + e] * B[e * D + d]
        }
        const at = sig(L[d]) ** (8 * sig(za))
        h[d] = at * h[d] + Math.sqrt(1 - at * at) * sig(zx) * X[t * D + d]
      }
      want.push(...h)
    }
    expect(maxError(flat(out.outputs), want)).toBeLessThan(1e-10)
  })

  it('rgLruLambda: sigmoid(Lambda)^8 returns the requested decays', () => {
    const ac = [0.9, 0.95, 0.999]
    const got = toFlat(rgLruLambda(tensor(ac))).map((l) => sig(l) ** 8)
    expect(maxError(got, ac)).toBeLessThan(1e-12)
  })
})
