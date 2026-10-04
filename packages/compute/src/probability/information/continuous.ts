/**
 * Mutual information of continuous variables: the Gaussian closed form, and the Kraskov–Stögbauer–Grassberger
 * k-nearest-neighbour estimator from samples.
 */

import { logDet } from 'aifn-compute/numerics/linalg'
import { digamma } from 'aifn-compute/numerics/special'
import type { Index, MatrixLike, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import {
  add,
  dense,
  fromData,
  isTensor,
  matmul,
  mul,
  shapeOfValue,
  sub,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Selection matrix picking the coordinates `indices` of a d-vector. */
function selector(indices: readonly Index[], d: Size): Tensor {
  const out = new Float64Array(indices.length * d)
  indices.forEach((j, i) => {
    if (!Number.isInteger(j) || j < 0 || j >= d) throw new DomainError('selector', `index ${j} is outside 0 … ${d - 1}`)
    out[i * d + j] = 1
  })
  return fromData(out, [indices.length, d])
}

/**
 * I(X; Y) of jointly Gaussian X = z[x] and Y = z[y] with covariance Σ of z (d × d): ½ log(|Σ_XX| |Σ_YY| / |Σ_(X,Y)|),
 * in nats (Cover and Thomas, 2006, §8.5). For scalars with correlation ρ this is −½ log(1 − ρ²). Differentiable in Σ.
 */
export function gaussianMutualInformation(
  covariance: Value,
  x: readonly Index[],
  y: readonly Index[],
  { base }: { base?: number } = {},
): Value {
  const d = shapeOfValue(covariance)[0]
  const block = (idx: readonly Index[]) => {
    const S = selector(idx, d)
    return matmul(matmul(S, covariance), transpose(S))
  }
  const nats = mul(0.5, sub(add(logDet(block(x)), logDet(block(y))), logDet(block([...x, ...y]))))
  return base === undefined ? nats : mul(nats, 1 / Math.log(base))
}

/** Samples as rows of a flat buffer: [n] (d = 1) or [n, d], read through tensor's dense kernels. */
function rows(x: VectorLike | MatrixLike, where: string): { n: Size; d: Size; values: Float64Array } {
  const first = isTensor(x) ? null : (x as ArrayLike<unknown>)[0]
  if (isTensor(x) ? x.shape.length === 2 : typeof first === 'object' && first !== null) {
    const { data, m, n } = dense.toMatrixF64(x as MatrixLike, where)
    return { n: m, d: n, values: data }
  }
  const values = dense.toF64(x as VectorLike, where)
  return { n: values.length, d: 1, values }
}

/**
 * The Kraskov–Stögbauer–Grassberger estimate of I(X; Y) from n paired samples (their estimator 1): with εᵢ the
 * max-norm distance from sample i to its k-th nearest neighbour in the joint space, and n_x(i), n_y(i) the numbers of
 * other samples strictly within εᵢ in each marginal space,
 * Î = ψ(k) + ψ(n) − ⟨ψ(n_x + 1) + ψ(n_y + 1)⟩ (Kraskov, Stögbauer and Grassberger, 2004, "Estimating mutual
 * information", Phys. Rev. E 69, eq. 8). In nats; O(n²) time. The estimate can be slightly negative for independent
 * variables. `x` and `y` are [n] or [n, d] (rows are samples).
 */
export function ksgMutualInformation(
  x: VectorLike | MatrixLike,
  y: VectorLike | MatrixLike,
  { k = 3 }: { k?: Size } = {},
): Scalar {
  const a = rows(x, 'ksgMutualInformation')
  const b = rows(y, 'ksgMutualInformation')
  if (a.n !== b.n)
    throw new ShapeError('ksgMutualInformation', 'ksgMutualInformation: x and y need the same number of samples')
  const n = a.n
  if (!(Number.isInteger(k) && k >= 1 && k < n))
    throw new DomainError('ksgMutualInformation', `ksgMutualInformation: need 1 ≤ k < n, got ${k}`)
  const dist = (s: typeof a, i: number, j: number) => {
    let m = 0
    for (let c = 0; c < s.d; c++) m = Math.max(m, Math.abs(s.values[i * s.d + c] - s.values[j * s.d + c]))
    return m
  }
  const joint = new Float64Array(n - 1)
  let total = 0
  for (let i = 0; i < n; i++) {
    let q = 0
    for (let j = 0; j < n; j++) if (j !== i) joint[q++] = Math.max(dist(a, i, j), dist(b, i, j))
    joint.sort()
    const eps = joint[k - 1]
    let nx = 0
    let ny = 0
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      if (dist(a, i, j) < eps) nx++
      if (dist(b, i, j) < eps) ny++
    }
    total += (digamma(nx + 1) as number) + (digamma(ny + 1) as number)
  }
  return (digamma(k) as number) + (digamma(n) as number) - total / n
}
