/**
 * Dtypes: the promotion table, bool masks, and complex128 storage, arithmetic, reductions, refusals and derivatives
 * (the ℝ² convention: a complex input's gradient is ∂L/∂x + i ∂L/∂y).
 */
import { describe, expect, it } from 'vitest'
import { grad, jvp, vmap } from 'aifn-compute/foundation/autodiff'
import { DTypeError } from 'aifn-compute/foundation/errors'
import {
  abs,
  add,
  allclose,
  angle,
  argmax,
  astype,
  complex,
  complexAbs,
  complexItem,
  conj,
  cumsum,
  div,
  equalTo,
  exp,
  expj,
  fromData,
  greater,
  imagPart,
  item,
  log,
  matmul,
  max,
  mean,
  mul,
  promoteTypes,
  realPart,
  resultType,
  sin,
  sqrt,
  square,
  sum,
  tensor,
  toArray,
  toComplexFlat,
  toFlat,
  where,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

const z = (pairs: [number, number][]) => tensor(pairs.map(([re, im]) => ({ re, im })))
const pairs = (t: Value) => toComplexFlat(t as Tensor).map(({ re, im }) => [re, im])

describe('promotion', () => {
  it('one table: the larger wins, except int32 with float32, which is float64 (as NumPy); symmetric', () => {
    const order = ['bool', 'int32', 'float32', 'float64', 'complex128'] as const
    order.forEach((a, i) =>
      order.forEach((b, j) => {
        const want =
          (a === 'int32' && b === 'float32') || (a === 'float32' && b === 'int32') ? 'float64' : order[Math.max(i, j)]
        expect(promoteTypes(a, b), `${a} with ${b}`).toBe(want)
        expect(promoteTypes(b, a)).toBe(promoteTypes(a, b))
      }),
    )
  })
  it('result rules', () => {
    expect(resultType('same', 'bool')).toBe('int32')
    expect(resultType('float', 'int32')).toBe('float64')
    expect(resultType('float', 'complex128')).toBe('complex128')
    expect(resultType('real', 'complex128')).toBe('float64')
    expect(resultType('realFloat', 'float32')).toBe('float32')
    expect(resultType('realFloat', 'int32')).toBe('float64')
    expect(resultType('index', 'float64')).toBe('int32')
    expect(resultType('complex', 'bool')).toBe('complex128')
  })
  it('elementwise results follow the table; numbers are weak', () => {
    const i = tensor([1, 2], [2], 'int32')
    const f = astype(tensor([1, 2]), 'float32')
    expect((add(i, i) as Tensor).dtype).toBe('int32')
    expect((div(i, i) as Tensor).dtype).toBe('float64')
    expect((add(f, 1.5) as Tensor).dtype).toBe('float32')
    expect((add(i, f) as Tensor).dtype).toBe('float64')
    expect((add(i, 2) as Tensor).dtype).toBe('int32')
    expect((add(i, 2.5) as Tensor).dtype).toBe('float64')
    expect((add(f, tensor([1, 2])) as Tensor).dtype).toBe('float64')
  })
})

describe('bool', () => {
  it('comparisons give bool masks stored as bytes; arithmetic on bools gives int32', () => {
    const m = greater(tensor([1, -1, 2]), 0) as Tensor
    expect(m.dtype).toBe('bool')
    expect(m.data).toBeInstanceOf(Uint8Array)
    expect((add(m, m) as Tensor).dtype).toBe('int32')
    expect(toFlat(add(m, m))).toEqual([2, 0, 2])
    expect(sum(m)).toBe(2)
    expect(toFlat(where(m, tensor([1, 2, 3]), -1))).toEqual([1, -1, 3])
    expect(astype(tensor([0, 2, -1]), 'bool').data).toEqual(Uint8Array.of(0, 1, 1))
  })
})

describe('complex128 storage and conversion', () => {
  it('constructors and converters', () => {
    const a = z([
      [1, 2],
      [3, -4],
    ])
    expect(a.dtype).toBe('complex128')
    expect(a.shape).toEqual([2])
    expect(toFlat(a)).toEqual([1, 2, 3, -4])
    expect(toArray(a)).toEqual([
      [1, 2],
      [3, -4],
    ])
    expect(complexItem(fromData(Float64Array.of(5, 6), [], 'complex128'))).toEqual({ re: 5, im: 6 })
    expect(() => item(fromData(Float64Array.of(5, 6), [], 'complex128'))).toThrow(DTypeError)
    expect((zeros([2], 'complex128') as Tensor).data.length).toBe(4)
    expect(pairs(complex(tensor([1, 2]), tensor([0.5, -1])))).toEqual([
      [1, 0.5],
      [2, -1],
    ])
  })
  it('realPart and imagPart are zero-copy views; astype to a real dtype refuses', () => {
    const a = z([
      [1, 2],
      [3, -4],
    ])
    const re = realPart(a) as Tensor
    expect(re.data).toBe(a.data)
    expect(toFlat(re)).toEqual([1, 3])
    expect(toFlat(imagPart(a))).toEqual([2, -4])
    expect(() => astype(a, 'float64')).toThrow(DTypeError)
  })
})

describe('complex arithmetic against the formulas', () => {
  const a = z([
    [1, 2],
    [-0.5, 0.25],
    [3, -1],
  ])
  const b = z([
    [0.3, -1],
    [2, 2],
    [-1, 0.5],
  ])
  const ref = (f: (p: [number, number], q: [number, number]) => [number, number]) =>
    pairs(a).map((p, k) => f(p as [number, number], pairs(b)[k] as [number, number]))
  const close = (got: number[][], want: number[][]) =>
    got.forEach((g, k) => g.forEach((v, j) => expect(v).toBeCloseTo(want[k][j], 12)))
  it('mul, div, exp, log, sqrt, abs, angle, conj, expj', () => {
    close(
      pairs(mul(a, b)),
      ref(([x, y], [u, v]) => [x * u - y * v, x * v + y * u]),
    )
    close(
      pairs(div(a, b)),
      ref(([x, y], [u, v]) => {
        const d = u * u + v * v
        return [(x * u + y * v) / d, (y * u - x * v) / d]
      }),
    )
    close(
      pairs(exp(a)),
      pairs(a).map(([x, y]) => [Math.exp(x) * Math.cos(y), Math.exp(x) * Math.sin(y)]),
    )
    close(
      pairs(log(a)),
      pairs(a).map(([x, y]) => [Math.log(Math.hypot(x, y)), Math.atan2(y, x)]),
    )
    close(pairs(square(sqrt(a))), pairs(a))
    expect(toFlat(abs(a))).toEqual(pairs(a).map(([x, y]) => Math.hypot(x, y)))
    expect(toFlat(complexAbs(a))).toEqual(toFlat(abs(a)))
    toFlat(angle(a)).forEach((v, k) => expect(v).toBeCloseTo(Math.atan2(pairs(a)[k][1], pairs(a)[k][0]), 14))
    expect(pairs(conj(a))).toEqual(pairs(a).map(([x, y]) => [x, -y]))
    close(pairs(expj(tensor([0, Math.PI / 2]))), [
      [1, 0],
      [0, 1],
    ])
    // The principal branch: √(−1 ± 0i) = ±i.
    close(
      pairs(
        sqrt(
          z([
            [-1, 0],
            [-1, -0],
          ]),
        ),
      ),
      [
        [0, 1],
        [0, -1],
      ],
    )
  })
  it('reductions and matmul', () => {
    expect(pairs(sum(a))).toEqual([[3.5, 1.25]])
    close(pairs(mean(a)), [[3.5 / 3, 1.25 / 3]])
    close(pairs(cumsum(a)), [
      [1, 2],
      [0.5, 2.25],
      [3.5, 1.25],
    ])
    // A complex matmul is four real ones.
    const A = complex(
      tensor([
        [1, 2],
        [0, -1],
      ]),
      tensor([
        [0.5, 0],
        [1, 1],
      ]),
    )
    const B = complex(
      tensor([
        [2, 0],
        [1, 1],
      ]),
      tensor([
        [0, -1],
        [0.5, 2],
      ]),
    )
    const [ar, ai, br, bi] = [realPart(A), imagPart(A), realPart(B), imagPart(B)]
    const want = complex(add(matmul(ar, br), mul(-1, matmul(ai, bi))), add(matmul(ar, bi), matmul(ai, br)))
    expect(allclose(matmul(A, B) as Tensor, want as Tensor)).toBe(true)
  })
  it('equality compares pairs; orderings, max, argmax and real-only functions refuse', () => {
    expect(
      toFlat(
        equalTo(
          a,
          z([
            [1, 2],
            [0, 0],
            [3, -1],
          ]),
        ),
      ),
    ).toEqual([1, 0, 1])
    expect(() => greater(a, 0)).toThrow(DTypeError)
    expect(() => max(a)).toThrow(DTypeError)
    expect(() => argmax(a)).toThrow(DTypeError)
    expect(() => sin(a)).toThrow(DTypeError)
  })
})

describe('complex derivatives (the ℝ² convention)', () => {
  it('∇|z|² = 2z, also per example under vmap', () => {
    const f = (w: Value) => sum(square(abs(w)))
    const a = z([
      [1, 2],
      [-3, 0.5],
    ])
    expect(pairs(grad(f)(a) as Value)).toEqual([
      [2, 4],
      [-6, 1],
    ])
    expect(pairs(vmap(grad(f))(a) as Value)).toEqual([
      [2, 4],
      [-6, 1],
    ])
  })
  it('the gradient of a real loss of a complex input matches differences in re and im', () => {
    const f = (w: Value) => sum(realPart(mul(exp(w), conj(w))))
    const a = z([
      [0.3, -0.7],
      [1.1, 0.4],
    ])
    const g = toFlat(grad(f)(a) as Tensor)
    const base = toFlat(a)
    const h = 1e-6
    base.forEach((_, k) => {
      const at = (d: number) => {
        const e = base.slice()
        e[k] += d
        return f(fromData(Float64Array.from(e), [2], 'complex128')) as number
      }
      expect(g[k]).toBeCloseTo((at(h) - at(-h)) / (2 * h), 7)
    })
  })
  it('holomorphic jvp: d/dt exp(z + t v) = exp(z) v', () => {
    const a = z([[0.2, 1]])
    const v = z([[0.5, -0.3]])
    const t = jvp((w: Value) => exp(w), a, v).tangent
    expect(allclose(t as Tensor, mul(exp(a), v) as Tensor)).toBe(true)
  })
  it('grad refuses a complex output', () => {
    expect(() => grad((w: Value) => sum(w))(z([[1, 1]]))).toThrow(DTypeError)
  })
})
