/**
 * Mutual information of continuous variables: the Gaussian closed form from a covariance matrix, and the
 * Kraskov–Stögbauer–Grassberger $k$-nearest-neighbour estimator from samples. The closed form is a composition of
 * primitives (differentiable in the covariance); the estimator works on plain numbers.
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

/**
 * The selection matrix $\Smat$ that picks the coordinates `indices` of a vector of length $d$: $\Smat\zvec$ is those
 * coordinates, in order. Throws a `DomainError` for an index that is not an integer in $0, \dots, d - 1$.
 *
 * @param indices The coordinates to pick, in the order wanted.
 * @param d The length $d$ of the vector.
 * @returns $\Smat$, of shape $[k, d]$ for $k$ indices, with a single 1 in each row.
 */
function selector(indices: readonly Index[], d: Size): Tensor {
  const out = new Float64Array(indices.length * d)
  indices.forEach((j, i) => {
    if (!Number.isInteger(j) || j < 0 || j >= d) throw new DomainError('selector', `index ${j} is outside 0 … ${d - 1}`)
    out[i * d + j] = 1
  })
  return fromData(out, [indices.length, d])
}

/**
 * The mutual information $I(X; Y)$ of jointly Gaussian blocks $X$ and $Y$ of a vector $\zvec$ with covariance
 * $\Sigmamat$ ($d \times d$):
 * $\tfrac{1}{2} \log(\lvert \Sigmamat_{XX} \rvert \lvert \Sigmamat_{YY} \rvert / \lvert \Sigmamat_{(X,Y)} \rvert)$,
 * in nats unless `base` is given (Cover and Thomas, 2006, §8.5). For scalars with correlation $\rho$ this is
 * $-\tfrac{1}{2} \log(1 - \rho^2)$. Differentiable in $\Sigmamat$. The blocks should not share coordinates (a shared
 * one makes $\Sigmamat_{(X,Y)}$ singular).
 *
 * @param covariance The covariance $\Sigmamat$ of $\zvec$, a $d \times d$ matrix.
 * @param x The coordinates of $\zvec$ that make up $X$.
 * @param y The coordinates of $\zvec$ that make up $Y$.
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits). Not checked.
 * @returns The mutual information, a number (a traced value under `grad`).
 *
 * @example Correlation 0.6, and one variable against a pair
 * const cov = tensor([[1, 0.6], [0.6, 1]])
 * print('I (nats):', gaussianMutualInformation(cov, [0], [1]))
 * print('-0.5 log(1 - 0.36):', -0.5 * Math.log(1 - 0.36))
 * const three = tensor([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
 * print('I(z0; (z1, z2)) (bits):', gaussianMutualInformation(three, [0], [1, 2], { base: 2 }))
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

/**
 * Samples as rows of a flat buffer, read through tensor's dense kernels: a matrix (a rank-2 tensor or an array of
 * rows) has a sample per row, and anything else is a vector of scalar samples ($d = 1$).
 *
 * @param x The samples: $n$ values, or an $n \times d$ matrix.
 * @param where The caller's name, for error messages.
 * @returns `n` samples of dimension `d`, with sample $i$ at entries `i * d` to `i * d + d - 1` of `values`.
 */
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
 * The Kraskov–Stögbauer–Grassberger estimate of $I(X; Y)$ from $n$ paired samples (their estimator 1): with
 * $\varepsilon_i$ the max-norm distance from sample $i$ to its $k$-th nearest neighbour in the joint space, and
 * $n_x(i)$, $n_y(i)$ the numbers of other samples strictly within $\varepsilon_i$ in each marginal space,
 * $\hat I = \psi(k) + \psi(n) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle$ ($\psi$ the digamma function, the
 * angle brackets the mean over samples; Kraskov, Stögbauer and Grassberger, 2004, "Estimating mutual information",
 * Phys. Rev. E 69, eq. 8). In nats; $O(n^2 \log n)$ time. The estimate can be slightly negative for independent
 * variables. Not differentiable. Throws a `ShapeError` when `x` and `y` have different numbers of samples, and a
 * `DomainError` unless $1 \le k < n$.
 *
 * @param x The samples of $X$: $n$ values, or an $n \times d_x$ matrix with one sample per row.
 * @param y The paired samples of $Y$: $n$ values, or an $n \times d_y$ matrix.
 * @param options The estimator's setting.
 * @param options.k The number of neighbours $k$: small $k$ has less bias, large $k$ less variance.
 * @returns The estimate, in nats.
 *
 * @example Correlated Gaussian samples, against the closed form
 * const s = stream(1)
 * const x = normal(s, 0, 1, { shape: [300] })
 * const y = add(x, normal(s, 0, 1, { shape: [300] }))
 * print('KSG estimate:', ksgMutualInformation(x, y))
 * print('exact, rho^2 = 1/2:', -0.5 * Math.log(0.5))
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
