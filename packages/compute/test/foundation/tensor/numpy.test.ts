import { describe, expect, it } from 'vitest'
import {
  add,
  arange,
  argmax,
  argmin,
  concat,
  div,
  einsum,
  greater,
  linspace,
  logsumexp,
  matmul,
  max,
  maximum,
  mean,
  min,
  mul,
  norm,
  outer,
  pow,
  prod,
  reshape,
  shapeOf,
  slice,
  stack,
  std,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  transpose,
  variance,
  where,
  type Axes,
  type NestedArray,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type N = NestedArray
const F = fixture<{
  broadcasting: Record<string, N>
  reductions: Record<string, N | Record<string, N>>
  matmul: Record<string, N>
  einsum: Record<string, N> & { cases: { spec: string; operands: string[]; result: N }[] }
  views: Record<string, N>
  linspace: number[]
  linspaceOpen: number[]
  arange: number[]
}>('foundation/tensor')

/** Compare a value with a numpy result: same shape, elements within `tol` (relative to max(1, |expected|)). */
function expectClose(actual: Value, expected: N, tol = 1e-12) {
  const shape = shapeOf(expected)
  if (typeof actual === 'number') {
    expect(shape).toEqual([])
    expect(Math.abs(actual - (expected as number))).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected as number)))
    return
  }
  const t = actual as Tensor
  expect(t.shape).toEqual(shape)
  const got = toFlat(t)
  const want = typeof expected === 'number' ? [expected] : ((expected as N[]).flat(Infinity as 1) as number[])
  got.forEach((v, k) =>
    expect(Math.abs(v - want[k]), `element ${k}`).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(want[k]))),
  )
}

describe('broadcasting against numpy', () => {
  const B = F.broadcasting
  const a = tensor(B.a)
  const b = tensor(B.b)
  it.each([
    ['add', add],
    ['sub', sub],
    ['mul', mul],
    ['div', div],
    ['maximum', maximum],
  ] as const)('%s broadcasts (3,1,4) with (2,1) to (3,2,4)', (name, op) => {
    expectClose(op(a, b), B[name])
  })
  it('pow', () => expectClose(pow(tensor(B.positive), b), B.pow))
  it('where with a broadcast condition', () => expectClose(where(greater(a, 0), a, b), B.where))
  it('row plus column', () => expectClose(add(reshape(arange(3), [3, 1]), reshape(arange(4), [1, 4])), B.rowPlusColumn))
})

describe('reductions against numpy', () => {
  const R = F.reductions
  const x = tensor(R.x as N)
  const axes: [string, Axes | undefined][] = [
    ['all', undefined],
    ['0', 0],
    ['1', 1],
    ['-1', -1],
    ['0,2', [0, 2]],
  ]
  for (const [name, axis] of axes) {
    const want = R[name] as Record<string, N>
    it(`axis ${name}`, () => {
      expectClose(sum(x, axis), want.sum)
      expectClose(mean(x, axis), want.mean)
      expectClose(max(x, axis), want.max)
      expectClose(min(x, axis), want.min)
      expectClose(prod(x, axis), want.prod)
      expectClose(variance(x, axis), want.variance)
      expectClose(variance(x, axis, false, 1), want.varianceDdof1)
      expectClose(std(x, axis), want.std)
      expectClose(logsumexp(x, axis), want.logsumexp)
      expectClose(sum(x, axis, true), want.sumKeep)
    })
  }
  it('argmax and argmin', () => {
    expect(toArray(argmax(x, 1))).toEqual(R.argmax1)
    expect(toArray(argmin(x, 2))).toEqual(R.argmin2)
    expect(argmax(x)).toBe(R.argmaxAll)
  })
  it('vector norms along an axis and over everything', () => {
    expectClose(norm(x, 1, false, 1), R.norm1Axis1 as N)
    expectClose(norm(x, 1), R.norm2Axis1 as N)
    expectClose(norm(x, 1, false, Infinity), R.normInfAxis1 as N)
    expectClose(norm(x, 1, false, 3), R.norm3Axis1 as N)
    expectClose(norm(x), R.normAll as N)
  })
})

describe('matmul against numpy', () => {
  const M = F.matmul
  const t = (k: string) => tensor(M[k])
  it('matrix × matrix', () => expectClose(matmul(t('m1'), t('m2')), M.m1m2))
  it('batched with broadcast batch axes: (2,1,3,4) × (5,4,2) → (2,5,3,2)', () =>
    expectClose(matmul(t('ba'), t('bb')), M.babb))
  it('vector × matrix and matrix × vector', () => {
    expectClose(matmul(t('v3'), t('m2')), M.v3m2)
    expectClose(matmul(t('m2'), t('v4')), M.m2v4)
  })
  it('vector × vector gives a rank-0 tensor', () => expectClose(matmul(t('v3'), t('v3')), M.v3v3))
  it('stack × vector', () => expectClose(matmul(t('s3'), t('v3')), M.s3v3))
  it('matmul of transposed views', () =>
    expectClose(matmul(transpose(t('m2')), transpose(t('m1'))), toArray(transpose(tensor(M.m1m2)))))
  it('outer', () => expectClose(outer(t('v3'), t('v4')), M.outer))
})

describe('einsum against numpy', () => {
  const E = F.einsum
  for (const c of E.cases) {
    it(c.spec, () => expectClose(einsum(c.spec, ...c.operands.map((k) => tensor(E[k]))), c.result, 1e-12))
  }
  it('works on strided views', () => {
    const e2 = tensor(E.e2)
    expectClose(einsum('ji,jk->ik', transpose(e2), tensor(E.e5)), toArray(matmul(e2, tensor(E.e5))))
  })
})

describe('views against numpy', () => {
  const V = F.views
  const grid = tensor(V.grid)
  const t = transpose(grid, [2, 0, 1])
  it('basic slicing', () => {
    expectClose(slice(grid, 1), V.row1)
    expectClose(slice(grid, null, [null, null, 2]), V.evenColumns)
    expectClose(slice(grid, [null, null, -1], [1, 3], -1), V.mixed)
    expectClose(slice(grid, null, null, [null, null, -2]), V.reversedLast)
    expectClose(slice(grid, [-2], [-3, -1], [1, 4, 2]), V.negativeRange)
    // An empty selection keeps its other axes (JSON cannot carry the shape of an empty array).
    expect(V.empty).toEqual([])
    expect(slice(grid, [2, 1]).shape).toEqual([0, 4, 5])
  })
  it('views share data', () => {
    const s = slice(grid, 1) as Tensor
    expect(s.data).toBe(grid.data)
    expect(t.data).toBe(grid.data)
  })
  it('transpose, slices of transposes, and reshape of a non-contiguous view (a copy)', () => {
    expectClose(t, V.transposed)
    expectClose(slice(t, [1, 4], null, [null, null, 3]), V.transposedSlice)
    const r = reshape(t, [5, 12])
    expectClose(r, V.transposedReshape)
    expect(r.data).not.toBe(grid.data)
    expect(reshape(grid, [12, 5]).data).toBe(grid.data)
  })
  it('concat and stack', () => {
    expectClose(concat([grid, slice(grid, null, [null, 2])], 1), V.concat1)
    expectClose(stack([slice(grid, 0), slice(grid, 1)], 2), V.stack2)
  })
})

describe('ranges against numpy', () => {
  it('linspace and arange', () => {
    expectClose(linspace(-1, 2, 7), F.linspace)
    expectClose(linspace(-1, 2, 7, false), F.linspaceOpen)
    expectClose(arange(2, 3, 0.25), F.arange)
  })
})
