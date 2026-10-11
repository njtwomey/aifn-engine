import { describe, expect, it } from 'vitest'
import { partials } from 'aifn-compute/dynamics/fields'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import {
  cos,
  exp,
  mean,
  mul,
  sin,
  slice,
  square,
  sub,
  tensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { Mlp } from 'aifn-compute/nn/layers'

// u(x, t) = e^{−t} sin x, row by row: u_x = e^{−t} cos x, u_xx = −u, u_t = −u, u_xt = −e^{−t} cos x.
const col = (X: Value, k: number) => slice(X, null, [k, k + 1])
const u = (X: Value) => mul(exp(mul(-1, col(X, 1))), sin(col(X, 0)))
const X = tensor([
  [0.3, 0],
  [1.2, 0.5],
  [-2, 1.5],
  [2.5, 0.1],
])
const rows = [
  [0.3, 0],
  [1.2, 0.5],
  [-2, 1.5],
  [2.5, 0.1],
]

describe('partials', () => {
  it('matches the closed-form derivatives of every order', () => {
    const [v, ux, uxx, ut, uxt, utx] = partials(u, X, [[], [0], [0, 0], [1], [0, 1], [1, 0]]).map((d) =>
      Array.from(toFlat(d as Tensor)),
    )
    rows.forEach(([x, t], i) => {
      const e = Math.exp(-t)
      expect(v[i]).toBeCloseTo(e * Math.sin(x), 12)
      expect(ux[i]).toBeCloseTo(e * Math.cos(x), 12)
      expect(uxx[i]).toBeCloseTo(-e * Math.sin(x), 12)
      expect(ut[i]).toBeCloseTo(-e * Math.sin(x), 12)
      expect(uxt[i]).toBeCloseTo(-e * Math.cos(x), 12)
      expect(utx[i]).toBeCloseTo(-e * Math.cos(x), 12)
    })
  })

  it('agrees with finite differences for a network', () => {
    const net = Mlp([2, 6, 6, 1], { activation: 'tanh' })
    const p = net.init(stream(1))
    const f = (x: Value) => net.apply(p, x)
    const [ux, uxx] = partials(f, X, [[0], [0, 0]]).map((d) => Array.from(toFlat(d as Tensor)))
    const h = 1e-4
    const at = (dx: number) => Array.from(toFlat(f(tensor(rows.map(([x, t]) => [x + dx, t]))) as Tensor))
    const [lo, mid, hi] = [at(-h), at(0), at(h)]
    rows.forEach((_, i) => {
      expect(ux[i]).toBeCloseTo((hi[i] - lo[i]) / (2 * h), 6)
      expect(uxx[i]).toBeCloseTo((hi[i] - 2 * mid[i] + lo[i]) / (h * h), 3)
    })
  })

  it('is differentiable in the parameters of the field', () => {
    // A residual loss u_t − u_xx in a scale a and a frequency w of u = a sin(w x) e^{−t}.
    const loss = (theta: Value) => {
      const [a, w] = [slice(theta, 0), slice(theta, 1)]
      const g = (Y: Value) => mul(mul(a, exp(mul(-1, col(Y, 1)))), sin(mul(w, col(Y, 0))))
      const [ut, uxx] = partials(g, X, [[1], [0, 0]])
      return mean(square(sub(ut, uxx)))
    }
    const report = gradCheck(loss, tensor([0.7, 1.3]))
    expect(report.ok).toBe(true)
  })

  it('refuses points that are not a matrix and columns out of range', () => {
    expect(() => partials(u, tensor([1, 2]), [[0]])).toThrow(ShapeError)
    expect(() => partials(u, X, [[2]])).toThrow(DomainError)
    expect(() => partials((Y) => cos(Y), X, [[-1]])).toThrow(DomainError)
  })
})
