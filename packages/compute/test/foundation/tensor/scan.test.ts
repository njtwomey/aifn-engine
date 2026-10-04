/**
 * The associative scan against sequential references: cumulative sums (`cumsum`), products and maxima, a
 * non-commutative operation (2 × 2 matrix products), tuples (the linear recurrence), axes and reverse; its gradient
 * against the sequential scan's; and the Hillis–Steele and Blelloch step-through scans against it.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import {
  add,
  associativeScan,
  blellochScanSteps,
  cumsum,
  hillisSteeleScanSteps,
  matmul,
  maximum,
  mul,
  slice,
  stack,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'

const flat = (v: Value) => toFlat(unwrap(v) as Tensor)
const close = (a: number[], b: number[], tol = 1e-12) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol * Math.max(1, Math.abs(b[i]))))
}
const seq = <R>(n: number, f: (i: number) => R): R[] => Array.from({ length: n }, (_, i) => f(i))

/** The sequential inclusive scan along axis 0, as the reference. */
function sequential(op: (a: Value, b: Value) => Value, x: Value, n: number): Value {
  const out: Value[] = [slice(x, 0)]
  for (let k = 1; k < n; k++) out.push(op(out[k - 1], slice(x, k)))
  return stack(out, 0)
}

describe('associativeScan', () => {
  it('equals cumsum for every length from 1 to 17', () => {
    for (let n = 1; n <= 17; n++) {
      const x = tensor(seq(n, (i) => Math.sin(3 * i) + 0.1 * i))
      close(flat(associativeScan(add, x)), flat(cumsum(x)))
    }
  })

  it('scans products and maxima along a chosen axis, and in reverse', () => {
    const x = tensor(seq(3, (r) => seq(7, (c) => 1 + 0.1 * Math.cos(r + 2 * c))))
    close(flat(associativeScan(mul, x, { axis: 1 })), flat(transposed(sequential(mul, tensor(transposeRows(x)), 7))))
    close(flat(associativeScan(maximum, x, { axis: -1, reverse: true })), flat(cumsumLikeReverse(x)))
  })

  it('keeps the operand order for a non-commutative operation (matrix products)', () => {
    const n = 11
    const ms = tensor(
      seq(n, (k) => [
        [Math.cos(k), Math.sin(0.3 * k)],
        [0.2 * k - 1, 1 + 0.05 * k],
      ]),
    )
    close(flat(associativeScan(matmul, ms)), flat(sequential(matmul, ms, n)), 1e-10)
    const back = associativeScan(matmul, ms, { reverse: true })
    // Reverse: y_k = M_k M_{k+1} … M_{n−1}.
    let acc: Value = slice(ms, n - 1)
    const expected: Value[] = [acc]
    for (let k = n - 2; k >= 0; k--) expected.unshift((acc = matmul(slice(ms, k), acc)))
    close(flat(back), flat(stack(expected, 0)), 1e-10)
  })

  it('scans tuples: the linear recurrence h_t = a_t h_{t−1} + b_t', () => {
    const n = 13
    const a = tensor(seq(n, (t) => seq(3, (j) => 0.9 * Math.cos(t + j))))
    const b = tensor(seq(n, (t) => seq(3, (j) => Math.sin(2 * t - j))))
    const op = ([a1, b1]: readonly Value[], [a2, b2]: readonly Value[]) => [mul(a1, a2), add(mul(a2, b1), b2)]
    const [, h] = associativeScan(op, [a, b])
    const ar = flat(a)
    const br = flat(b)
    const ref: number[] = []
    const state = [0, 0, 0]
    for (let t = 0; t < n; t++)
      for (let j = 0; j < 3; j++) ref.push((state[j] = ar[3 * t + j] * state[j] + br[3 * t + j]))
    close(flat(h), ref, 1e-12)
  })

  it('differentiates like the sequential scan', () => {
    const n = 9
    const op = ([a1, b1]: readonly Value[], [a2, b2]: readonly Value[]) => [mul(a1, a2), add(mul(a2, b1), b2)]
    const w = tensor(seq(n, (t) => Math.cos(t)))
    const viaScan = (a: Value, b: Value) => sum(mul(w, associativeScan(op, [a, b])[1]))
    const viaLoop = (a: Value, b: Value) => {
      const hs: Value[] = []
      let h: Value = 0
      for (let t = 0; t < n; t++) hs.push((h = add(mul(slice(a, t), h), slice(b, t))))
      return sum(mul(w, stack(hs, 0)))
    }
    const a = tensor(seq(n, (t) => 0.8 * Math.sin(t + 1)))
    const b = tensor(seq(n, (t) => Math.cos(2 * t)))
    for (const argnums of [0, 1] as const) {
      const g1 = grad(viaScan, { argnums })(a, b)
      const g2 = grad(viaLoop, { argnums })(a, b)
      close(flat(g1 as Value), flat(g2 as Value), 1e-12)
    }
  })
})

describe('step-through scans', () => {
  it('Hillis–Steele ends on the associative scan in ⌈log₂ n⌉ rounds', () => {
    for (const n of [1, 2, 5, 8, 13]) {
      const x = tensor(seq(n, (i) => [i + 1, Math.cos(i)]))
      const s = run(hillisSteeleScanSteps(add, x), undefined, 100)
      close(flat(s.values), flat(associativeScan(add, x)))
      expect(s.t).toBe(Math.ceil(Math.log2(n)))
    }
  })

  it('Blelloch gives the exclusive and inclusive scans with about 2P applications', () => {
    for (const n of [1, 3, 8, 11]) {
      const x = tensor(seq(n, (i) => 0.5 + Math.sin(i)))
      const s = run(blellochScanSteps(add, x, 0), undefined, 100)
      close(flat(s.inclusive!), flat(cumsum(x)))
      close(flat(s.exclusive!), [0, ...flat(cumsum(x)).slice(0, n - 1)])
    }
    const ms = tensor(
      seq(6, (k) => [
        [1, 0.1 * k],
        [0.2 * k, 1],
      ]),
    )
    const eye = tensor([
      [1, 0],
      [0, 1],
    ])
    const s = run(blellochScanSteps(matmul, ms, eye), undefined, 100)
    close(flat(s.inclusive!), flat(associativeScan(matmul, ms)), 1e-12)
  })
})

// Helpers for the axis and reverse cases.
function transposeRows(x: Tensor): number[][] {
  const [r, c] = x.shape
  const v = toFlat(x)
  return seq(c, (j) => seq(r, (i) => v[i * c + j]))
}
function transposed(v: Value): Tensor {
  return tensor(transposeRows(unwrap(v) as Tensor))
}
function cumsumLikeReverse(x: Tensor): Tensor {
  const [r, c] = x.shape
  const v = toFlat(x)
  const out = new Array<number>(r * c)
  for (let i = 0; i < r; i++) {
    let m = -Infinity
    for (let j = c - 1; j >= 0; j--) out[i * c + j] = m = Math.max(m, v[i * c + j])
  }
  return tensor(seq(r, (i) => out.slice(i * c, (i + 1) * c)))
}
