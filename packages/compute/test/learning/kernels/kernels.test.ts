/**
 * aifn-compute/learning/kernels: closed-form values (Rasmussen and Williams, 2006, ch. 4), Gram matrices and diagonals, and
 * hyperparameters as pytrees: `logParams`/`kernelFromLog` round trips, `ravel` of a combined kernel's tree, and
 * gradients of a Gram-matrix objective with respect to the whole tree against central differences.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { ravel, treeLeaves } from 'aifn-compute/foundation/pytree'
import { mul, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  constant,
  gram,
  kernelDiagonal,
  kernelFromLog,
  kernelProfile,
  linearKernel,
  logParams,
  matern12,
  matern32,
  matern52,
  periodic,
  polynomial,
  productKernel,
  rationalQuadratic,
  rbf,
  sumKernel,
  white,
  type Kernel,
  type KernelParams,
} from 'aifn-compute/learning/kernels'

const x = tensor([[0], [0.5], [2]])
const y = tensor([[1], [-1]])
const r = (a: number, b: number) => Math.abs(a - b)
const pairs = (f: (a: number, b: number) => number) => toFlat(x).flatMap((a) => Array.from(toFlat(y), (b) => f(a, b)))
const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-12) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++)
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i])))
}

describe('closed forms', () => {
  const l = 0.7
  const v = 1.3
  it('stationary kernels', () => {
    close(
      toFlat(gram(rbf({ lengthscale: l, variance: v }), x, y)),
      pairs((a, b) => v * Math.exp(-(r(a, b) ** 2) / (2 * l * l))),
    )
    close(
      toFlat(gram(matern12({ lengthscale: l, variance: v }), x, y)),
      pairs((a, b) => v * Math.exp(-r(a, b) / l)),
    )
    close(
      toFlat(gram(matern32({ lengthscale: l, variance: v }), x, y)),
      pairs((a, b) => {
        const s = (Math.sqrt(3) * r(a, b)) / l
        return v * (1 + s) * Math.exp(-s)
      }),
    )
    close(
      toFlat(gram(matern52({ lengthscale: l, variance: v }), x, y)),
      pairs((a, b) => {
        const s = (Math.sqrt(5) * r(a, b)) / l
        return v * (1 + s + (s * s) / 3) * Math.exp(-s)
      }),
    )
    close(
      toFlat(gram(rationalQuadratic({ lengthscale: l, variance: v, alpha: 2 }), x, y)),
      pairs((a, b) => v * (1 + r(a, b) ** 2 / (2 * 2 * l * l)) ** -2),
    )
    close(
      toFlat(gram(periodic({ lengthscale: l, variance: v, period: 1.5 }), x, y)),
      pairs((a, b) => v * Math.exp((-2 * Math.sin((Math.PI * r(a, b)) / 1.5) ** 2) / (l * l))),
    )
  })

  it('dot-product kernels and white noise', () => {
    close(
      toFlat(gram(linearKernel({ variance: v, bias: 0.4 }), x, y)),
      pairs((a, b) => 0.4 + v * a * b),
    )
    close(
      toFlat(gram(polynomial(3, { variance: v, bias: 0.4 }), x, y)),
      pairs((a, b) => (0.4 + v * a * b) ** 3),
    )
    expect(() => polynomial(1.5)).toThrow()
    // White noise appears only when the inputs are the same set (y omitted).
    close(toFlat(gram(white({ variance: 0.3 }), x)), [0.3, 0, 0, 0, 0.3, 0, 0, 0, 0.3])
    close(toFlat(gram(white({ variance: 0.3 }), x, x)), Array(9).fill(0))
    close(toFlat(gram(constant({ variance: 2 }), x, y)), Array(6).fill(2))
  })

  it('sums and products combine elementwise; diagonals match the Gram diagonal', () => {
    const a = rbf({ lengthscale: l })
    const b = periodic({ period: 2 })
    const s = sumKernel(a, b, white({ variance: 0.1 }))
    const p = productKernel(a, b)
    const ga = toFlat(gram(a, x))
    const gb = toFlat(gram(b, x))
    close(
      toFlat(gram(s, x)),
      ga.map((e, i) => e + gb[i] + (i % 4 === 0 ? 0.1 : 0)),
    )
    close(
      toFlat(gram(p, x)),
      ga.map((e, i) => e * gb[i]),
    )
    expect(s.name).toBe('sum(rbf, periodic, white)')
    expect(s.stationary).toBe(true)
    expect(sumKernel(a, linearKernel()).stationary).toBe(false)
    expect(() => sumKernel()).toThrow()
    for (const k of [a, b, s, p, linearKernel({ bias: 1 }), polynomial(2)] as Kernel[]) {
      const g = toFlat(gram(k, x))
      close(toFlat(kernelDiagonal(k, x) as Tensor), [g[0], g[4], g[8]])
      expect(k.kind).toBe('kernel')
    }
  })

  it('kernelProfile is k(τ, 0)', () => {
    const k = matern32({ lengthscale: 2 })
    const lags = tensor([0, 1, 3])
    close(toFlat(kernelProfile(k, lags)), toFlat(gram(k, tensor([[0], [1], [3]]), tensor([[0]]))))
  })
})

describe('hyperparameters as pytrees', () => {
  const k = sumKernel(
    productKernel(rbf({ lengthscale: 0.8, variance: 1.5 }), periodic({ lengthscale: 1.2, period: 2 })),
    white({ variance: 0.2 }),
  )

  it('logParams and kernelFromLog round-trip the tree', () => {
    const theta = logParams(k)
    expect(treeLeaves(theta).map((leaf) => leaf.path)).toEqual(treeLeaves(k.params).map((leaf) => leaf.path))
    const back = kernelFromLog(k, theta)
    close(
      treeLeaves(back.params).map((leaf) => leaf.value as number),
      treeLeaves(k.params).map((leaf) => leaf.value as number),
    )
    close(toFlat(gram(back, x)), toFlat(gram(k, x)))
    expect(back.name).toBe(k.name)
  })

  it('ravel flattens the tree to one vector and unravels it', () => {
    const { vector, unravel } = ravel(logParams(k))
    // rbf (lengthscale, variance) + periodic (lengthscale, variance, period) + white (variance).
    expect(vector.length).toBe(6)
    const moved = unravel(Array.from(vector, (v) => v + 0.1))
    const k2 = kernelFromLog(k, moved)
    close(
      treeLeaves(k2.params).map((leaf) => leaf.value as number),
      treeLeaves(k.params).map((leaf) => (leaf.value as number) * Math.exp(0.1)),
    )
    expect(() => unravel([1, 2])).toThrow()
    expect(() => k.withParams({ terms: [] })).toThrow(/term/)
  })

  it('gradients through the whole tree match central differences', () => {
    const w = tensor([
      [1, -0.5, 0.2],
      [0.3, 2, -1],
      [0.1, 0.4, 1],
    ])
    const objective = (theta: KernelParams): Value => sum(mul(gram(kernelFromLog(k, theta), x), w))
    const theta = logParams(k)
    const g = ravel(grad(objective)(theta)).vector
    const { vector, unravel } = ravel(theta)
    const h = 1e-6
    const f = (v: Float64Array) => objective(unravel(v)) as number
    vector.forEach((_, i) => {
      const up = Float64Array.from(vector)
      const down = Float64Array.from(vector)
      up[i] += h
      down[i] -= h
      expect(g[i]).toBeCloseTo((f(up) - f(down)) / (2 * h), 6)
    })
  })

  it('the default linear kernel is homogeneous: no bias leaf, every log parameter finite', () => {
    const k = linearKernel({ variance: 2 })
    expect(Object.keys(k.params)).toEqual(['variance'])
    const theta = logParams(k)
    for (const leaf of treeLeaves(theta)) expect(Number.isFinite(leaf.value as number)).toBe(true)
    // σ² xᵀx′ with no offset; the round trip through log space keeps the kernel.
    const x = tensor([
      [1, 2],
      [-1, 0.5],
    ])
    const expected = [10, 0, 0, 2.5]
    close(toFlat(gram(k, x) as Tensor), expected, 1e-14)
    close(toFlat(gram(kernelFromLog(k, theta), x) as Tensor), expected, 1e-14)
    expect(linearKernel({ bias: 0 }).params).toEqual({ variance: 1 })
    // A positive bias is a hyperparameter and has a log.
    const withBias = logParams(linearKernel({ bias: 0.5 })) as { bias: number }
    expect(withBias.bias).toBeCloseTo(Math.log(0.5), 15)
  })
})
