/**
 * Scalarisation of several objectives into one: the augmented Chebyshev function of ParEGO (Knowles, 2006), and the
 * lattice of weight vectors ParEGO draws from.
 *
 * With objectives normalised to $[0, 1]$ and weights $\lambda_j \ge 0$, the augmented Chebyshev scalarisation is
 *
 * $$ f_{\lambdavec}(\yvec) = \max_{j} \lambda_j y_j + \rho \sum_{j} \lambda_j y_j. $$
 *
 * Unlike a weighted sum, the maximum term reaches every point of the Pareto front, including those in a non-convex
 * part of it, as the weights vary; the small sum term (Knowles uses $\rho = 0.05$) breaks ties in favour of points
 * that are not weakly dominated. ParEGO draws a weight vector at random at each iteration and fits one surrogate to
 * the scalarised values, so a single-objective method such as expected improvement explores the whole front.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { rowsOf, signs, type Directions } from './pareto'

const { mat, vec } = dense

/** Options of `chebyshevScalarisation`. */
export interface ChebyshevOptions {
  /** The weight $\rho \ge 0$ of the sum term (default 0.05, ParEGO's value). */
  rho?: number
  /**
   * The ideal point $\zvec^*$, one value per objective (the best of each), subtracted before weighting. Left out, the
   * values are used as they are, which assumes they are already normalised.
   */
  ideal?: readonly number[]
  /**
   * The nadir point (the worst value of each objective). With `ideal`, each objective is normalised to
   * $(y_j - z^*_j) / (z^{\mathrm{nad}}_j - z^*_j)$, the $[0, 1]$ scale ParEGO uses.
   */
  nadir?: readonly number[]
  /** Minimise every objective (default true), maximise every one (false), or one flag per objective. */
  minimise?: Directions
}

/**
 * The augmented Chebyshev scalarisation of ParEGO (Knowles, 2006, equation 2): $\max_j \lambda_j y_j + \rho \sum_j
 * \lambda_j y_j$ for each row, after subtracting the ideal point and dividing by the range to the nadir point when
 * they are given. Smaller is better; a maximised objective is negated first (and its ideal and nadir points with it),
 * so the result is always minimised. Throws `ShapeError` when the weights, the ideal or the nadir point differ in
 * length from the rows, and `DomainError` for a negative or NaN weight, $\rho < 0$, a nadir point without an ideal one
 * or a range of zero.
 *
 * @param y The objective values, $n \times M$, one row per point.
 * @param weights The weights $\lambda_j \ge 0$, $M$ values (ParEGO's sum to 1; see `simplexLattice`).
 * @param options $\rho$, the normalisation and the directions, as `ChebyshevOptions`.
 * @param options.rho The weight $\rho \ge 0$ of the sum term (default 0.05).
 * @param options.ideal The ideal point, subtracted before weighting; left out, the values are used as given.
 * @param options.nadir The nadir point; with `ideal`, each objective is divided by its range between the two.
 * @param options.minimise The direction of each objective (default: minimise all).
 * @returns The $n$ scalarised values, a vector.
 *
 * @example A non-convex front: the middle point wins under Chebyshev weights, never under a weighted sum
 * const front = [[0, 1], [0.6, 0.6], [1, 0]]
 * print('Chebyshev, weights (0.5, 0.5):', chebyshevScalarisation(front, [0.5, 0.5]))
 * print('weighted sum:', front.map(([a, b]) => 0.5 * a + 0.5 * b))
 *
 * @example Normalising by the ideal and nadir points
 * const y = [[10, 300], [20, 100], [40, 50]]
 * const s = chebyshevScalarisation(y, [0.3, 0.7], { ideal: [10, 50], nadir: [40, 300] })
 * print('scalarised:', s)
 */
export function chebyshevScalarisation(
  y: MatrixLike,
  weights: readonly number[],
  { rho = 0.05, ideal, nadir, minimise = true }: ChebyshevOptions = {},
): Tensor {
  const where = 'chebyshevScalarisation'
  const M = weights.length
  if (M === 0) throw new ShapeError(where, `${where}: no weights`)
  if (!weights.every((w) => w >= 0)) throw new DomainError(where, `${where}: weights must be non-negative`)
  if (!(rho >= 0)) throw new DomainError(where, `${where}: rho must be ≥ 0`)
  const sign = signs(minimise, M, where)
  const point = (v: readonly number[] | undefined, name: string) => {
    if (v === undefined) return null
    if (v.length !== M) throw new ShapeError(where, `${where}: ${name} has ${v.length} values for ${M} objectives`)
    return Float64Array.from(v, (x, j) => x * sign[j])
  }
  const z = point(ideal, 'the ideal point')
  const nad = point(nadir, 'the nadir point')
  if (nad && !z) throw new DomainError(where, `${where}: a nadir point needs an ideal point`)
  if (z && nad && !nad.every((v, j) => v !== z[j]))
    throw new DomainError(where, `${where}: the ideal and nadir points must differ in every objective`)
  return vec(
    Float64Array.from(rowsOf(y, sign, where), (row) => {
      let largest = -Infinity
      let sum = 0
      for (let j = 0; j < M; j++) {
        let v = row[j]
        if (z) v -= z[j]
        if (z && nad) v /= nad[j] - z[j]
        largest = Math.max(largest, weights[j] * v)
        sum += weights[j] * v
      }
      return largest + rho * sum
    }),
  )
}

/**
 * The simplex lattice of weight vectors: every $\lambdavec$ with $\lambda_j = l_j / s$ for non-negative integers
 * $l_j$ summing to $s$, $\binom{s + M - 1}{M - 1}$ vectors in all. ParEGO draws its weights uniformly from this set
 * with $s = 10$: 11 vectors for two objectives and 66 for three (Knowles, 2006). Throws `DomainError` unless both
 * counts are positive integers.
 *
 * @param objectives The number of objectives $M$.
 * @param divisions The number of divisions $s$ of each axis (default 10).
 * @returns The weights, one vector per row, in lexicographic order of $(l_1, \dots, l_M)$.
 *
 * @example ParEGO's weights for two objectives, and how many there are for three
 * print('M = 2:', simplexLattice(2))
 * print('M = 3:', simplexLattice(3).shape[0], 'vectors')
 */
export function simplexLattice(objectives: number, divisions = 10): Tensor {
  const where = 'simplexLattice'
  if (!(Number.isInteger(objectives) && objectives >= 1))
    throw new DomainError(where, `${where}: the number of objectives must be a positive integer`)
  if (!(Number.isInteger(divisions) && divisions >= 1))
    throw new DomainError(where, `${where}: the number of divisions must be a positive integer`)
  const rows: number[][] = []
  const walk = (prefix: number[], left: number) => {
    if (prefix.length === objectives - 1) {
      rows.push([...prefix, left].map((l) => l / divisions))
      return
    }
    for (let l = 0; l <= left; l++) walk([...prefix, l], left - l)
  }
  walk([], divisions)
  return mat(Float64Array.from(rows.flat()), rows.length, objectives)
}
