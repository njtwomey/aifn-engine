import { describe, expect, it } from 'vitest'
import {
  abs,
  add,
  broadcastTo,
  clip,
  concat,
  cos,
  defineOp,
  definePrimitive,
  diag,
  diagonal,
  div,
  dot,
  einsum,
  exp,
  expandDims,
  expm1,
  flatten,
  get,
  greater,
  log,
  log1p,
  logsumexp,
  map,
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  mul,
  neg,
  norm,
  outer,
  permute,
  pow,
  prod,
  reshape,
  set,
  sign,
  sin,
  slice,
  sqrt,
  square,
  squeeze,
  stack,
  std,
  sub,
  sum,
  sumLike,
  sumTo,
  tanh,
  tensor,
  transpose,
  unwrap,
  variance,
  where,
  elementwise,
  linearCombination,
  scalar,
  toFlat,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { grad, hessian, jvp, vmap } from 'aifn-compute/foundation/autodiff'
import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import { checkGradient } from './check-gradient'

/** Deterministic test data of a shape, in (lo, hi), with no ties. */
function data(shape: number[], lo = -1.5, hi = 1.5, seed = 1): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  return tensor(
    Array.from(
      { length: n },
      (_, k) => lo + (((hi - lo) * (((Math.sin(12.9898 * (k + seed)) * 43758.5453) % 1) + 1)) % 1),
    ),
    shape,
  )
}

const X = data([3, 4])
const P = data([3, 4], 0.2, 2)

describe('unary vjps match finite differences', () => {
  const cases: [string, (x: Value) => Value, Raw][] = [
    ['neg', neg, X],
    ['abs', abs, X],
    ['square', square, X],
    ['exp', exp, X],
    ['expm1', expm1, X],
    ['log', log, P],
    ['log1p', log1p, P],
    ['sqrt', sqrt, P],
    ['sin', sin, X],
    ['cos', cos, X],
    ['tanh', tanh, X],
    ['exp of a number', exp, 0.7],
  ]
  it.each(cases)('%s', (_name, f, x) => {
    checkGradient(f, [x])
  })
  it('sign has a zero cotangent', () => {
    const g = checkGradient((x) => sign(x), [X])
    expect(g[0].every((v) => v === 0)).toBe(true)
  })
})

describe('binary vjps match finite differences, with broadcasting', () => {
  const a = data([3, 1], -1, 1, 2)
  const b = data([1, 4], 0.5, 1.5, 3)
  const cases: [string, (x: Value, y: Value) => Value][] = [
    ['add', add],
    ['sub', sub],
    ['mul', mul],
    ['div', div],
    ['maximum', maximum],
    ['minimum', minimum],
  ]
  it.each(cases)('%s', (_name, f) => {
    checkGradient(f, [a, b])
    checkGradient(f, [X, 0.3])
    checkGradient(f, [0.3, X])
    checkGradient(f, [0.4, 0.9])
  })
  it('pow in base and exponent', () => {
    checkGradient(pow, [data([3, 1], 0.3, 2, 4), b])
    checkGradient((x) => pow(x, 3), [X])
  })
  it('where routes cotangents by the condition', () => {
    checkGradient((x, y) => where(greater(X, 0), x, y), [X, data([4], -1, 1, 5)])
  })
  it('clip', () => checkGradient((x) => clip(x, -0.5, 0.8), [X]))
})

describe('structural vjps match finite differences', () => {
  const T = data([2, 3, 4])
  const cases: [string, (x: Value) => Value][] = [
    ['reshape', (x) => reshape(x, [4, -1])],
    ['flatten', (x) => flatten(x)],
    ['permute', (x) => permute(x, [2, 0, 1])],
    ['transpose then reshape (a copy)', (x) => reshape(transpose(x), [6, 4])],
    ['broadcastTo', (x) => broadcastTo(slice(x, null, [0, 1]), [5, 2, 3, 4])],
    ['sumTo', (x) => sumTo(x, [3, 1])],
    ['squeeze and expandDims', (x) => squeeze(expandDims(x, 1), 1)],
    ['slice with steps', (x) => slice(x, [null, null, -1], [0, 3, 2], [1])],
    ['integer slice', (x) => slice(x, 1, -1)],
    ['get', (x) => get(x, 1, 2, -1)],
    ['set', (x) => set(x, [0, 1, 2], mul(get(x, 1, 1, 1), 3))],
    ['concat', (x) => concat([x, slice(x, null, [1])], 1)],
    ['stack', (x) => stack([slice(x, 0), slice(x, 1)], 2)],
  ]
  it.each(cases)('%s', (_name, f) => {
    checkGradient(f, [T])
  })
  it('diagonal and diag', () => {
    checkGradient((x) => diagonal(x), [X])
    checkGradient((x) => diagonal(transpose(x)), [X])
    checkGradient((x) => diag(x), [data([3])])
  })
})

describe('reduction vjps match finite differences', () => {
  const T = data([2, 3, 4])
  const axes = [undefined, 0, 1, -1, [0, 2]] as const
  it.each(axes)('axis %s', (axis) => {
    const ax = axis as number | number[] | undefined
    checkGradient((x) => sum(x, ax), [T])
    checkGradient((x) => mean(x, ax, true), [T])
    checkGradient((x) => max(x, ax), [T])
    checkGradient((x) => min(x, ax), [T])
    checkGradient((x) => prod(x, ax), [T])
    checkGradient((x) => logsumexp(x, ax), [T])
    checkGradient((x) => variance(x, ax, false, 1), [T])
    checkGradient((x) => std(x, ax), [T])
  })
  it('norms of each order', () => {
    for (const ord of [1, 2, 3, Infinity, -Infinity]) checkGradient((x) => norm(x, 1, false, ord), [T])
  })
  it('prod with zeros gives the product of the others', () => {
    const z = tensor([
      [2, 0, 3],
      [0, 0, 5],
      [1, 2, 4],
    ])
    const [g] = checkGradient((x) => prod(x, 1), [z])
    const w = [0.3 + 0.7 * Math.sin(0.4), 0.3 + 0.7 * Math.sin(2.1), 0.3 + 0.7 * Math.sin(3.8)]
    expect(g[1]).toBeCloseTo(6 * w[0], 12)
    expect(g[3]).toBe(0)
  })
  it('prod has exact second derivatives with two or more zeros: ∂²y/∂xᵢ∂xⱼ is the product of the rest', () => {
    // H[i][j] = Π over k ∉ {i, j} of xₖ for i ≠ j, 0 on the diagonal.
    const exact = (v: number[]) =>
      v.map((_, i) => v.map((_, j) => (i === j ? 0 : v.reduce((p, x, k) => (k === i || k === j ? p : p * x), 1))))
    for (const v of [
      [0, 0, 2, 3],
      [0, 2, 0, 3],
      [0, 0, 0, 2],
      [0, 2, 3, 5],
      [1, 2, 3, 4],
    ]) {
      const H = hessian((x: Value) => prod(x))(tensor(v)) as Tensor
      expect(Array.from(H.data)).toEqual(exact(v).flat())
    }
    // Along an axis: row [0, 0, 5] has H[0][1] = 5.
    const rows = hessian((x: Value) => sum(prod(x, 1)))(
      tensor([
        [0, 0, 5],
        [1, 2, 4],
      ]),
    ) as Tensor
    expect(rows.data[1]).toBe(5)
    expect(rows.data[3 * 6 + 4]).toBe(4)
  })
  it('max splits the cotangent equally among ties', () => {
    const g = grad((x: Value) => max(x))(tensor([1, 3, 3, 2]))
    expect(Array.from((unwrap(g) as Tensor).data)).toEqual([0, 0.5, 0.5, 0])
  })
})

describe('product vjps match finite differences', () => {
  it('matmul: matrices, batches with broadcasting, and vectors', () => {
    checkGradient(matmul, [data([2, 3]), data([3, 4], -1, 1, 7)])
    checkGradient(matmul, [data([2, 1, 3, 4]), data([5, 4, 2], -1, 1, 8)])
    checkGradient(matmul, [data([3]), data([3, 4], -1, 1, 9)])
    checkGradient(matmul, [data([4, 2, 3]), data([3], -1, 1, 10)])
    checkGradient(matmul, [data([3]), data([3], -1, 1, 11)])
  })
  it('dot and outer', () => {
    checkGradient(dot, [data([4]), data([4], -1, 1, 12)])
    checkGradient(outer, [data([3]), data([4], -1, 1, 13)])
  })
  it.each([
    [
      'ij,jk->ik',
      [
        [2, 3],
        [3, 4],
      ],
    ],
    ['ij->ji', [[2, 3]]],
    ['i,i->', [[3], [3]]],
    ['ij->i', [[2, 3]]],
    [
      'bij,bjk->bik',
      [
        [2, 3, 4],
        [2, 4, 2],
      ],
    ],
    [
      'ij,jk,kl->il',
      [
        [2, 3],
        [3, 4],
        [4, 2],
      ],
    ],
    ['ij,k->ijk', [[2, 3], [2]]],
    ['ijk->', [[2, 3, 2]]],
  ] as [string, number[][]][])('einsum %s', (spec, shapes) => {
    checkGradient(
      (...xs) => einsum(spec, ...xs),
      shapes.map((s, i) => data(s, -1, 1, 20 + i)),
    )
  })
  it('einsum differentiates an operand with a repeated label (the cotangent lands on the diagonal)', () => {
    checkGradient((x) => einsum('ii->', x), [data([3, 3])])
    checkGradient((x) => einsum('ii->i', x), [data([3, 3])])
    checkGradient((x, y) => einsum('ij,jj->i', x, y), [data([2, 3]), data([3, 3], -1, 1, 5)])
  })
})

describe('higher derivatives', () => {
  it('the second derivative of exp(sin x) is traced through the vjps', () => {
    const x0 = 0.7
    const f = (x: Value) => exp(sin(x))
    expect(unwrap(grad(f)(x0))).toBeCloseTo(Math.cos(x0) * Math.exp(Math.sin(x0)), 14)
    const exact = Math.exp(Math.sin(x0)) * (Math.cos(x0) ** 2 - Math.sin(x0))
    expect(unwrap(grad((x: Value) => grad(f)(x) as Value)(x0))).toBeCloseTo(exact, 14)
  })
  it('the Hessian of a matrix expression (logsumexp of Ax) row by row', () => {
    const A = data([3, 2], -1, 1, 30)
    const x0 = tensor([0.2, -0.4])
    const g = (x: Value) => grad((y: Value) => logsumexp(matmul(A, y)))(x) as Value
    // H = Aᵀ (diag(p) − p pᵀ) A with p = softmax(Ax).
    const z = Array.from((matmul(A, x0) as Tensor).data)
    const m = Math.max(...z)
    const e = z.map((v) => Math.exp(v - m))
    const p = e.map((v) => v / e.reduce((s, u) => s + u, 0))
    const a = (i: number, j: number) => A.data[i * 2 + j]
    for (let r = 0; r < 2; r++) {
      const row = grad((x: Value) => get(g(x), r))(x0)
      for (let c = 0; c < 2; c++) {
        let h = 0
        for (let i = 0; i < 3; i++)
          for (let j = 0; j < 3; j++) h += a(i, r) * ((i === j ? p[i] : 0) - p[i] * p[j]) * a(j, c)
        expect((unwrap(row) as Tensor).data[c]).toBeCloseTo(h, 12)
      }
    }
  })
})

describe('elementwise and defineOp (local primitives)', () => {
  const sigmoid = elementwise({
    id: 'sigmoid',
    f: (v) => 1 / (1 + Math.exp(-v)),
    derivative: [(_v, y) => mul(y, sub(1, y))],
  })
  const softplus = elementwise({
    id: 'softplus',
    f: (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))),
    derivative: [(x) => sigmoid(x)],
  })
  const hypot = elementwise({
    id: 'hypot',
    f: (a: number, b: number) => Math.hypot(a, b),
    derivative: [(a, _b, y) => div(a, y), (_a, b, y) => div(b, y)],
  })
  it('return numbers for numbers and tensors for tensors, broadcasting', () => {
    expect(softplus(0)).toBeCloseTo(Math.LN2, 15)
    expect(typeof softplus(0)).toBe('number')
    expect((softplus(tensor([0, 1])) as Tensor).shape).toEqual([2])
    expect(hypot(3, 4)).toBe(5)
    expect(Array.from((hypot(tensor([3, 5]), tensor([[4], [12]])) as Tensor).data)).toEqual([
      5,
      Math.hypot(5, 4),
      Math.hypot(3, 12),
      13,
    ])
  })
  it('have vjps matching finite differences', () => {
    checkGradient(softplus, [X])
    checkGradient(hypot, [data([3, 1], 0.5, 2, 3), data([1, 4], 0.5, 2, 4)])
  })
  it('the derivative is a primitive, so every order is differentiable', () => {
    const s = 1 / (1 + Math.exp(-0.3))
    const d2 = grad((x: Value) => grad(softplus)(x) as Value)(0.3)
    expect(unwrap(d2)).toBeCloseTo(s * (1 - s), 14)
    // ∂²/∂a² hypot(a, b) = b²/y³ and ∂²/∂a∂b = −ab/y³.
    const [gaa, gab] = grad((a: Value, b: Value) => grad(hypot)(a, b) as Value, { argnums: [0, 1] })(3, 4)
    expect(unwrap(gaa)).toBeCloseTo(16 / 125, 14)
    expect(unwrap(gab)).toBeCloseTo(-12 / 125, 14)
    expect(unwrap(jvp((x: Value) => softplus(x), 0.3, 1).tangent as Value)).toBeCloseTo(s, 14)
  })
  it('a null derivative throws when that argument is differentiated, and is ignored when it is constant', () => {
    const scaled = elementwise({ id: 'scaled', f: (a, k) => a * k, derivative: [(_a, k) => k, null] })
    checkGradient((a) => scaled(a, 2), [X])
    expect(() => checkGradient((k) => scaled(X, k), [0.5])).toThrow(NotDifferentiableError)
  })
  it('a ternary elementwise primitive broadcasts and differentiates in every argument', () => {
    const fma = elementwise({
      id: 'fma',
      f: (a, b, c) => a * b + c * c,
      derivative: [(_a, b) => b, (a) => a, (_a, _b, c) => mul(2, c)],
    })
    expect(fma(2, 3, 1)).toBe(7)
    const y = fma(tensor([1, 2]), tensor([[1], [10]]), 1) as Tensor
    expect(y.shape).toEqual([2, 2])
    expect(Array.from(y.data)).toEqual([2, 3, 11, 21])
    expect((fma(tensor([1, 2], [2], 'int32'), 3, 0) as Tensor).dtype).toBe('float64')
    checkGradient(fma, [data([3, 1], -1, 1, 2), data([1, 4], -1, 1, 5), data([4], -1, 1, 6)])
    checkGradient((a, b, c) => fma(a, b, c), [0.3, data([2, 2], -1, 1, 7), 1.5])
    expect(unwrap(grad((c: Value) => grad((z: Value) => fma(1, 2, z))(c) as Value)(0.7))).toBe(2)
  })
  it('an op without a derivative is an error to differentiate through', () => {
    expect(() => checkGradient((x) => map(x, Math.floor), [X])).toThrow(NotDifferentiableError)
  })
  it('defineOp and definePrimitive: general primitives with parameters', () => {
    const scaleRows = defineOp<number>(
      'scaleRows',
      ([m], k) => mul(m as Tensor, k) as Raw,
      (g, _inputs, _y, k) => [mul(g, k)],
    )
    expect(Array.from((scaleRows([tensor([1, 2])], 3) as Tensor).data)).toEqual([3, 6])
    checkGradient((x) => scaleRows([x], 2.5), [X])
    // A linear primitive needs only its transpose: vjp and jvp are derived.
    const twice = definePrimitive<undefined>({
      id: 'twice',
      arity: 1,
      impl: ([x]) => mul(x, 2) as Raw,
      linear: 'linear',
      transpose: (ct) => mul(ct, 2),
    })
    checkGradient((x) => twice([x], undefined), [X])
    expect(unwrap(jvp((x: Value) => twice([x], undefined), 1.5, 1).tangent as Value)).toBe(2)
  })
  it('sumLike reduces a broadcast cotangent to the input shape and kind', () => {
    const g = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(sumLike(g, 0)).toBe(21)
    expect(Array.from((sumLike(g, tensor([0, 0, 0])) as Tensor).data)).toEqual([5, 7, 9])
    expect(Array.from((sumLike(g, tensor([[0], [0]])) as Tensor).data)).toEqual([6, 15])
    expect(sumLike(g, g)).toBe(g)
    expect((sumLike(2, tensor([0, 0])) as Tensor).shape).toEqual([2])
  })
})

describe('review regressions (2026-10-01)', () => {
  it('pow has finite derivatives at a = 0 with b = 0 (polynomial features at x = 0)', () => {
    // d/dx Σₖ xᵏ for k = 0, 1, 2 at x = 0 is 0 + 1 + 0; b·aᵇ⁻¹ alone gives 0·∞ = NaN for the constant term.
    expect(grad((x: Value) => sum(pow(x, tensor([0, 1, 2]))))(0)).toBe(1)
    expect(grad((x: Value) => pow(x, 0))(0)).toBe(0)
    expect(hessian((x: Value) => pow(x, 0))(0)).toBe(0)
    // d/db 0ᵇ = 0 for b > 0, and its second derivative is 0 too (y·log a guarded inside the unused branch).
    expect(grad((b: Value) => pow(0, b))(2)).toBe(0)
    expect(hessian((b: Value) => pow(0, b))(2)).toBe(0)
    // Away from the guards the rules are unchanged, including the mixed partial at b = 0: ∂²(aᵇ)/∂a∂b = 1/a there.
    const mixed = grad((b: Value) => grad((a: Value) => pow(a, b))(2) as Value)(0) as number
    expect(mixed).toBeCloseTo(0.5, 12)
    expect(grad((x: Value) => pow(x, 3))(-2)).toBeCloseTo(12, 12)
  })

  it('the gradient of a sum has the kind of its input (a rank-0 tensor gives a rank-0 tensor)', () => {
    const g = grad((x: Value) => sum(x))(scalar(2)) as Tensor
    expect(typeof g).toBe('object')
    expect(g.shape).toEqual([])
    expect(grad((x: Value) => sum(x))(2)).toBe(1)
  })

  it('linearCombination batches a number example against a tensor one, as its forward rule broadcasts', () => {
    const c = tensor([1, 2, 3])
    const batched = vmap((a: Value) => linearCombination([a, c], [1, 1]))(tensor([10, 20])) as Tensor
    expect(batched.shape).toEqual([2, 3])
    expect(toFlat(batched)).toEqual([11, 12, 13, 21, 22, 23])
    const both = vmap((a: Value, b: Value) => linearCombination([a, b], [1, 2]))(
      tensor([1, 2]),
      tensor([
        [1, 2],
        [3, 4],
      ]),
    )
    expect(toFlat(both as Tensor)).toEqual([3, 5, 8, 10])
  })
})

describe('subgradients at the edges (review foundation 13, 14)', () => {
  const finite = (g: Value) => toFlat(g as Tensor) as ArrayLike<number>
  it('logsumexp: a fully −∞ group has gradient 0, a +∞ group splits it over its +∞ entries', () => {
    const masked = tensor([
      [-Infinity, -Infinity, -Infinity],
      [0, Math.log(3), -Infinity],
    ])
    const g = finite(grad((x: Value) => sum(logsumexp(x, 1)))(masked))
    expect(Array.from(g)).toEqual([0, 0, 0, 0.25, 0.75, 0])
    const h = finite(grad((x: Value) => sum(logsumexp(x, 1)))(tensor([[Infinity, 1, Infinity]])))
    expect(Array.from(h)).toEqual([0.5, 0, 0.5])
    // Second derivatives through the masked row stay finite.
    const hh = hessian((x: Value) => logsumexp(x))(tensor([-Infinity, -Infinity])) as Tensor
    expect(Array.from(toFlat(hh)).every(Number.isFinite)).toBe(true)
  })
  it('norm: the gradient at 0 is 0 for every order p > 1, and unchanged away from 0', () => {
    for (const ord of [1.5, 3, 4]) {
      const g = finite(grad((x: Value) => norm(x, null, false, ord))(tensor([0, 0, 0])))
      expect(Array.from(g)).toEqual([0, 0, 0])
      const x = tensor([1, -2, 0.5])
      const at = finite(grad((v: Value) => norm(v, null, false, ord))(x))
      const n = (1 + 2 ** ord + 0.5 ** ord) ** (1 / ord)
      const want = [1, -2, 0.5].map((v) => (Math.sign(v) * Math.abs(v) ** (ord - 1)) / n ** (ord - 1))
      for (let i = 0; i < 3; i++) expect(at[i]).toBeCloseTo(want[i], 12)
    }
  })
})
