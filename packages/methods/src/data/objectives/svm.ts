/**
 * A tuning surface for an RBF support vector machine: the cross-validated error over $\log_{10} C$ and
 * $\log_{10}\gamma$, tabulated once by `scripts/svm-surface.ts` (`svm-table.ts`) and interpolated, so that a
 * hyperparameter search over it costs nothing. Evaluation noise can be added to stand for the variance of a validation
 * estimate, which is what makes tuning by Bayesian optimisation noisy.
 *
 * The table holds the mean error of `supportVectorMachine` with the kernel $\exp(-\gamma\lVert \xvec - \xvec'
 * \rVert^2)$ on Fisher's iris (versicolor against virginica), over ten repeats of stratified 5-fold cross-validation,
 * on a 21 by 21 grid with $\log_{10} C \in [-2, 3]$ and $\log_{10}\gamma \in [-3, 2]$. Between the nodes the surface is
 * bilinear: on a cell with corners $f_{00}, f_{10}, f_{01}, f_{11}$ and local coordinates $u, v \in [0, 1]$,
 *
 * $$ f = (1 - u)(1 - v) f_{00} + u(1 - v) f_{10} + (1 - u) v f_{01} + u v f_{11}, $$
 *
 * so its minimum over the box is at a node.
 */

import type { ObjectiveInfo } from 'aifn-compute/foundation/contracts'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { real, space } from 'aifn-compute/foundation/space'
import { dense, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { TestFunction } from './surfaces'
import { SVM_TABLE } from './svm-table'

const { mat, toF64, vec } = dense

/** A tabulated tuning surface: a `TestFunction` whose value may be noisy, with the table it interpolates. */
export type TuningSurface = TestFunction & {
  /**
   * The surface at $\xvec = (\log_{10} C, \log_{10}\gamma)$ plus, when `noise` is positive, a Gaussian error of that
   * standard deviation drawn from `stream` (or, left out, the stream the surface was made with).
   */
  value: (x: Vector, stream?: Stream) => number
  /** The surface without noise: the interpolated mean validation error. */
  mean: (x: Vector) => number
  /** The standard deviation of the evaluation noise (0 for a deterministic surface). */
  noise: number
  /** The grid's values of $\log_{10} C$ (the table's rows). */
  logC: Vector
  /** The grid's values of $\log_{10}\gamma$ (the table's columns). */
  logGamma: Vector
  /** The tabulated mean validation error, one row per value of $\log_{10} C$. */
  error: Matrix
  /** The standard deviation of the validation error over the folds at each node, laid out as `error`. */
  spread: Matrix
}

/**
 * The cell of a coordinate on an equally spaced axis: its index and the local coordinate in $[0, 1]$. A coordinate
 * outside the axis is moved to its nearest end.
 *
 * @param axis The axis's values, increasing and equally spaced.
 * @param x The coordinate.
 * @returns `i`, the cell's lower node (at most the second-to-last), and `u`, the position in the cell.
 */
function cell(axis: readonly number[], x: number): { i: number; u: number } {
  const n = axis.length
  const h = axis[1] - axis[0]
  const t = (Math.min(Math.max(x, axis[0]), axis[n - 1]) - axis[0]) / h
  const i = Math.min(Math.floor(t), n - 2)
  return { i, u: t - i }
}

/**
 * The SVM tuning surface: the cross-validated error of an RBF support vector machine on iris (versicolor against
 * virginica) as a function of $\xvec = (\log_{10} C, \log_{10}\gamma)$, interpolated bilinearly from a 21 by 21 table
 * on $[-2, 3] \times [-3, 2]$, optionally with Gaussian evaluation noise. A diagonal valley of good settings (larger
 * $C$ pairs with smaller $\gamma$) runs between a plateau of underfitting and a cliff of overfitting at large
 * $\gamma$. A point outside the box is evaluated at the nearest point of the box. The gradient and Hessian
 * (`objective`, `hessian`) are those of the noise-free interpolant within a cell; the minimisers are the nodes of
 * smallest error.
 * Throws `DomainError` for negative noise, and, at evaluation, for positive noise with no stream.
 *
 * @param options The noise and its stream.
 * @param options.noise The standard deviation of the evaluation noise (default 0); the table's `spread` gives a
 *   realistic scale, the spread of a single fold's error.
 * @param options.stream The stream the noise is drawn from when `value` is not given one.
 * @returns The surface, as a `TuningSurface`.
 *
 * @example The best node, a point between nodes, and a noisy evaluation
 * const f = svmTuningSurface()
 * print('lowest error', f.minimumValue, 'at log10 C, log10 gamma =', f.minima[0])
 * print('error at (0.1, -1.1):', f.value([0.1, -1.1]))
 * const noisy = svmTuningSurface({ noise: 0.01, stream: stream(0) })
 * print('noisy evaluations there:', [1, 2, 3].map(() => noisy.value([0.1, -1.1]).toFixed(4)))
 */
export function svmTuningSurface({ noise = 0, stream }: { noise?: number; stream?: Stream } = {}): TuningSurface {
  if (!(noise >= 0)) throw new DomainError('svmTuningSurface', 'svmTuningSurface: noise must be ≥ 0')
  const { logC, logGamma, error, sd } = SVM_TABLE
  const nC = logC.length
  const nG = logGamma.length
  const hC = logC[1] - logC[0]
  const hG = logGamma[1] - logGamma[0]
  const read = (x: Vector) => {
    const a = toF64(x, 'svmTuningSurface')
    if (a.length !== 2)
      throw new ShapeError('svmTuningSurface', `svmTuningSurface: expected a point of length 2, got ${a.length}`)
    const { i, u } = cell(logC, a[0])
    const { i: j, u: v } = cell(logGamma, a[1])
    const [f00, f10, f01, f11] = [error[i][j], error[i + 1][j], error[i][j + 1], error[i + 1][j + 1]]
    return { u, v, f00, f10, f01, f11 }
  }
  const mean = (x: Vector) => {
    const { u, v, f00, f10, f01, f11 } = read(x)
    return (1 - u) * (1 - v) * f00 + u * (1 - v) * f10 + (1 - u) * v * f01 + u * v * f11
  }
  const grad = (x: Vector) => {
    const { u, v, f00, f10, f01, f11 } = read(x)
    return vec(
      Float64Array.of(((1 - v) * (f10 - f00) + v * (f11 - f01)) / hC, ((1 - u) * (f01 - f00) + u * (f11 - f10)) / hG),
    )
  }
  let lowest = Infinity
  for (const row of error) for (const e of row) lowest = Math.min(lowest, e)
  const minima: number[][] = []
  for (let i = 0; i < nC; i++)
    for (let j = 0; j < nG; j++) if (error[i][j] === lowest) minima.push([logC[i], logGamma[j]])
  const centre = [(logC[0] + logC[nC - 1]) / 2, (logGamma[0] + logGamma[nG - 1]) / 2]
  return {
    name: 'svm-tuning-surface',
    dimension: 2,
    value: (x: Vector, s?: Stream) => {
      const f = mean(x)
      if (noise === 0) return f
      const from = s ?? stream
      if (!from)
        throw new DomainError('svmTuningSurface', 'svmTuningSurface: a noisy surface needs a stream to draw from')
      return f + noise * normal(from)
    },
    mean,
    objective: (x) => ({ value: mean(x), grad: grad(x) }),
    hessian: (x) => {
      const { f00, f10, f01, f11 } = read(x)
      const cross = (f11 - f10 - f01 + f00) / (hC * hG)
      return mat(Float64Array.of(0, cross, cross, 0), 2, 2)
    },
    minima: minima.map((m) => vec(Float64Array.from(m))),
    minimumValue: lowest,
    start: vec(Float64Array.from(centre)),
    lo: vec(Float64Array.of(logC[0], logGamma[0])),
    hi: vec(Float64Array.of(logC[nC - 1], logGamma[nG - 1])),
    noise,
    logC: vec(Float64Array.from(logC)),
    logGamma: vec(Float64Array.from(logGamma)),
    error: mat(Float64Array.from(error.flat()), nC, nG),
    spread: mat(Float64Array.from(sd.flat()), nC, nG),
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<ObjectiveInfo>('objective', 'data/objectives')(
  {
    key: 'svmTuningSurface',
    name: 'SVM tuning surface',
    summary: 'The cross-validated error of an RBF SVM over log C and log γ, tabulated once, with optional noise.',
    params: space({ noise: real(0, 0.05, { default: 0 }) }),
    dim: 2,
    truth: true,
  },
  svmTuningSurface,
)
