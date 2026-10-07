/**
 * Dictionary learning: a dictionary $\Dmat$ ($m \times k$, unit-norm atoms) and sparse codes $\Xmat$ ($k \times n$)
 * such that $\Ymat \approx \Dmat\Xmat$ for training signals $\Ymat$ ($m \times n$, one per column), by minimising
 * $\frac{1}{2}\lVert \Ymat - \Dmat\Xmat \rVert_F^2$ subject to at most $s$ non-zeros in each column of $\Xmat$.
 *
 * Each step alternates the two stages of Aharon, Elad and Bruckstein (2006): sparse coding of every signal with
 * $\Dmat$ fixed (`orthogonalMatchingPursuit` to $s$ atoms), then an update of the dictionary with the support of
 * $\Xmat$ fixed, by one of two rules:
 *
 * - `mod`: the method of optimal directions (Engan, Aase and Husøy, 1999), the least-squares dictionary for the codes,
 *   $\Dmat = \Ymat\Xmat^\top(\Xmat\Xmat^\top)^{-1}$ (by `lstsq`), with its atoms then scaled to unit norm and the rows
 *   of $\Xmat$ scaled to match.
 * - `ksvd`: K-SVD (Aharon, Elad and Bruckstein, 2006), one atom at a time: with $\omega_j$ the signals that use atom
 *   $j$ and $\Emat_j$ the residual on them without that atom, the atom and its coefficients become the best rank-one
 *   approximation of $\Emat_j$, $\dvec_j = \uvec_1$ and $\xvec^j_{\omega_j} = \sigma_1 \vvec_1^\top$ (by `svd`). The
 *   coefficients change but their support does not, so the codes stay $s$-sparse.
 *
 * Greedy coding can fit a signal worse than its previous code did, so a signal keeps its previous code when that fits
 * better; with this, neither update lets the objective increase (up to rounding). An atom no signal uses is replaced by the
 * worst-represented training signal, scaled to unit norm, as Aharon et al. do. The default initial dictionary is $k$
 * distinct training signals drawn at random, scaled to unit norm. The objective is not convex in $\Dmat$ and $\Xmat$
 * together, so the result is a local minimum that depends on the start.
 */

import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { normal, permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { lstsq, svd } from 'aifn-compute/numerics/linalg'
import { atomNorms, readDictionary } from './atoms'
import { sparseCode } from './code'

type F64 = dense.F64

/** The dictionary update rule: the method of optimal directions, or K-SVD. */
export type DictionaryUpdate = 'mod' | 'ksvd'

/** Options of `dictionaryLearningSteps` and `dictionaryLearning`. */
export type DictionaryLearningOptions = {
  /** The number of atoms $k$ to learn; more than $m$ makes the dictionary overcomplete. */
  atoms: Size
  /** The number of non-zeros $s$ in each code, from 1 to $\min(m, k)$. */
  sparsity: Size
  /** The dictionary update (default `'ksvd'`). */
  update?: DictionaryUpdate
  /** The initial dictionary, $m \times k$ (its atoms are scaled to unit norm); default $k$ random training signals. */
  init?: MatrixLike
  /** Stop when the objective changes by less than this fraction of itself in one step. Default $10^{-6}$. */
  tolerance?: number
}

/** The state of `dictionaryLearningSteps`. */
export interface DictionaryLearningState extends Status {
  /** Steps taken: rounds of coding and dictionary update. */
  t: Size
  /** The dictionary $\Dmat$, $m \times k$, unit-norm atoms as columns. */
  D: Tensor
  /** The codes $\Xmat$, $k \times n$, at most $s$ non-zeros per column. */
  X: Tensor
  /** $\frac{1}{2}\lVert \Ymat - \Dmat\Xmat \rVert_F^2$. */
  objective: number
  /** Atoms replaced on this step because no signal used them. */
  replaced: Size
  /** Set when the objective's relative change in a step falls below the tolerance, or the fit is exact. */
  converged: boolean
}

/**
 * The residual $\Ymat - \Dmat\Xmat$.
 *
 * @param Y The signals, $m \times n$ row-major.
 * @param D The dictionary, $m \times k$ row-major.
 * @param X The codes, $k \times n$ row-major.
 * @param m The signal length.
 * @param k The number of atoms.
 * @param n The number of signals.
 * @returns The residual, $m \times n$ row-major.
 */
function residualOf(Y: F64, D: F64, X: F64, m: Size, k: Size, n: Size): F64 {
  return dense.sub(Y, dense.matMul(D, X, m, k, n))
}

/**
 * Each signal's better code: the fresh one from orthogonal matching pursuit, unless the previous one (also at most $s$
 * non-zeros) fits the signal strictly better over the same dictionary. Greedy coding alone can lose ground; keeping
 * the better code makes the coding stage, and so every step, non-increasing in the objective.
 *
 * @param Y The signals, $m \times n$ row-major.
 * @param D The dictionary both codes are for, $m \times k$ row-major.
 * @param fresh The new codes, $k \times n$ row-major; returned, with the columns that lose overwritten.
 * @param previous The previous codes, $k \times n$ row-major.
 * @param m The signal length.
 * @param k The number of atoms.
 * @param n The number of signals.
 * @returns The codes, $k \times n$ row-major.
 */
function keepBetter(Y: F64, D: F64, fresh: F64, previous: F64, m: Size, k: Size, n: Size): F64 {
  const columnErrors = (X: F64) => {
    const R = residualOf(Y, D, X, m, k, n)
    const e = new Float64Array(n)
    for (let r = 0; r < m; r++) for (let i = 0; i < n; i++) e[i] += R[r * n + i] ** 2
    return e
  }
  const now = columnErrors(fresh)
  const before = columnErrors(previous)
  for (let i = 0; i < n; i++) if (before[i] < now[i]) for (let j = 0; j < k; j++) fresh[j * n + i] = previous[j * n + i]
  return fresh
}

/**
 * Replace every atom no signal uses with the signal (scaled to unit norm) whose residual is largest, a different
 * signal for each, and zero its row of codes. `D` and `X` are modified in place.
 *
 * @param Y The signals, $m \times n$ row-major.
 * @param D The dictionary, $m \times k$ row-major; modified.
 * @param X The codes, $k \times n$ row-major; modified.
 * @param m The signal length.
 * @param k The number of atoms.
 * @param n The number of signals.
 * @returns How many atoms were replaced.
 */
function replaceUnused(Y: F64, D: F64, X: F64, m: Size, k: Size, n: Size): Size {
  const unused: number[] = []
  for (let j = 0; j < k; j++) {
    let used = false
    for (let i = 0; i < n && !used; i++) used = X[j * n + i] !== 0
    if (!used) unused.push(j)
  }
  if (!unused.length) return 0
  const R = residualOf(Y, D, X, m, k, n)
  const error = Array.from({ length: n }, (_, i) => {
    let e = 0
    for (let r = 0; r < m; r++) e += R[r * n + i] ** 2
    return e
  })
  const worst = error.map((e, i) => [e, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1])
  unused.forEach((j, at) => {
    const i = worst[at % n][1]
    let norm = 0
    for (let r = 0; r < m; r++) norm += Y[r * n + i] ** 2
    norm = Math.sqrt(norm)
    for (let r = 0; r < m; r++) D[r * k + j] = norm > 0 ? Y[r * n + i] / norm : r === j % m ? 1 : 0
    for (let c = 0; c < n; c++) X[j * n + c] = 0
  })
  return unused.length
}

/**
 * The method of optimal directions: $\Dmat = \Ymat\Xmat^\top(\Xmat\Xmat^\top)^{-1}$ by least squares, its atoms scaled
 * to unit norm and the rows of $\Xmat$ scaled by the same factors, so $\Dmat\Xmat$ is unchanged by the scaling.
 *
 * @param Y The signals, $m \times n$ row-major.
 * @param X The codes, $k \times n$ row-major; its rows are rescaled in place.
 * @param m The signal length.
 * @param k The number of atoms.
 * @param n The number of signals.
 * @returns The new dictionary, $m \times k$ row-major.
 */
function modUpdate(Y: F64, X: F64, m: Size, k: Size, n: Size): F64 {
  // Xᵀ Dᵀ = Yᵀ in the least-squares sense; the minimum-norm solution gives an unused atom a zero column.
  const Dt = dense.data(lstsq(dense.mat(dense.transpose(X, k, n), n, k), dense.mat(dense.transpose(Y, m, n), n, m)).x)
  const D = dense.transpose(Dt, k, m)
  const norms = atomNorms({ data: D, m, k })
  for (let j = 0; j < k; j++) {
    if (norms[j] === 0) continue
    for (let r = 0; r < m; r++) D[r * k + j] /= norms[j]
    for (let i = 0; i < n; i++) X[j * n + i] *= norms[j]
  }
  return D
}

/**
 * K-SVD's dictionary update: each used atom in turn, with its coefficients, becomes the best rank-one approximation of
 * the residual it is meant to explain. `D` and `X` are modified in place.
 *
 * @param Y The signals, $m \times n$ row-major.
 * @param D The dictionary, $m \times k$ row-major; modified.
 * @param X The codes, $k \times n$ row-major; the non-zeros of each row are modified, the zeros kept.
 * @param m The signal length.
 * @param k The number of atoms.
 * @param n The number of signals.
 */
function ksvdUpdate(Y: F64, D: F64, X: F64, m: Size, k: Size, n: Size): void {
  for (let j = 0; j < k; j++) {
    const omega: number[] = []
    for (let i = 0; i < n; i++) if (X[j * n + i] !== 0) omega.push(i)
    if (!omega.length) continue
    // E = Y_ω − D X_ω + d_j x^j_ω: the residual on the signals using atom j, with atom j's part added back.
    const w = omega.length
    const E = new Float64Array(m * w)
    for (let r = 0; r < m; r++)
      omega.forEach((i, c) => {
        let v = Y[r * n + i]
        for (let a = 0; a < k; a++) if (a !== j) v -= D[r * k + a] * X[a * n + i]
        E[r * w + c] = v
      })
    const { U, S, V } = svd(dense.mat(E, m, w))
    const sigma = S.data[0]
    if (!(sigma > 0)) continue
    const kU = U.shape[1]
    const kV = V.shape[1]
    for (let r = 0; r < m; r++) D[r * k + j] = U.data[r * kU]
    omega.forEach((i, c) => (X[j * n + i] = sigma * V.data[c * kV]))
  }
}

/**
 * Dictionary learning by alternating sparse coding (orthogonal matching pursuit) and a dictionary update (K-SVD or
 * the method of optimal directions), one round per step. The initial state codes the signals over the initial
 * dictionary; `init` takes a stream for the random initial atoms (unused when `options.init` gives them).
 *
 * @param Y The training signals $\Ymat$, $m \times n$, one signal per column.
 * @param options The number of atoms, the sparsity, the update rule, the initial dictionary and the tolerance.
 * @returns A step-through algorithm whose states hold $\Dmat$, $\Xmat$ and the objective.
 *
 * @example K-SVD against the method of optimal directions
 * // Eighteen signals in R^3, each a combination of two of four atoms. Neither run reaches zero from its random start:
 * // the objective is not convex, and each settles in a local minimum.
 * const atoms = toArray(normaliseAtoms([[1, 0, 1, 1], [0, 1, 1, -1], [0, 0, 1, 1]]))
 * const pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
 * const weights = [[1, 0.5], [-0.7, 1], [0.3, -1.2]]
 * const Y = atoms.map((row) => pairs.flatMap(([a, b]) => weights.map(([p, q]) => p * row[a] + q * row[b])))
 * for (const update of ['ksvd', 'mod']) {
 *   const alg = dictionaryLearningSteps(Y, { atoms: 4, sparsity: 2, update })
 *   print(update, 'objective after 0, 1, 5 and 30 steps:', [0, 1, 5, 30].map((t) => run(alg, undefined, t).objective))
 * }
 */
export function dictionaryLearningSteps(
  Y: MatrixLike,
  options: DictionaryLearningOptions,
): Algorithm<void, DictionaryLearningState> {
  const { atoms: k, sparsity, update = 'ksvd', tolerance = 1e-6 } = options
  const y = dense.toMatrixF64(Y, 'dictionaryLearning')
  const { m, n } = y
  const signals = Float64Array.from(y.data)
  if (!(Number.isInteger(k) && k >= 1))
    throw new DomainError('dictionaryLearning', `dictionaryLearning: atoms must be a positive integer, got ${k}`)
  if (!(Number.isInteger(sparsity) && sparsity >= 1 && sparsity <= Math.min(m, k)))
    throw new DomainError(
      'dictionaryLearning',
      `dictionaryLearning: sparsity must be an integer from 1 to ${Math.min(m, k)}, got ${sparsity}`,
    )
  // An objective this small relative to the signals' energy is rounding error: the fit is exact.
  const exact = 1e-20 * 0.5 * dense.dot(signals, signals)
  const objectiveOf = (D: F64, X: F64) => {
    const R = residualOf(signals, D, X, m, k, n)
    return 0.5 * dense.dot(R, R)
  }
  if (n === 0) throw new DomainError('dictionaryLearning', 'dictionaryLearning: there are no training signals')
  const Ymat = dense.mat(signals, m, n)
  const code = (D: F64) =>
    Float64Array.from(dense.data(sparseCode(dense.mat(D, m, k), Ymat, { method: 'omp', sparsity }).X))
  return {
    name: update === 'ksvd' ? 'ksvd' : 'mod',
    init: (_start, s: Stream) => {
      let D: F64
      if (options.init) {
        const d = readDictionary(options.init, 'dictionaryLearning: init')
        if (d.m !== m || d.k !== k)
          throw new ShapeError(
            'dictionaryLearning',
            `dictionaryLearning: init must be ${m} × ${k}, got ${d.m} × ${d.k}`,
          )
        D = d.data
      } else {
        // k distinct training signals; past n of them, Gaussian atoms.
        D = new Float64Array(m * k)
        const order = dense.data(permutation(s, n))
        const extra = dense.data(normal(s, 0, 1, { shape: [m * k] }))
        for (let j = 0; j < k; j++)
          for (let r = 0; r < m; r++) D[r * k + j] = j < n ? signals[r * n + order[j]] : extra[r * k + j]
      }
      const norms = atomNorms({ data: D, m, k })
      for (let r = 0; r < m; r++) for (let j = 0; j < k; j++) if (norms[j] > 0) D[r * k + j] /= norms[j]
      const X = code(D)
      return {
        t: 0,
        D: dense.mat(D, m, k),
        X: dense.mat(X, k, n),
        objective: objectiveOf(D, X),
        replaced: 0,
        converged: false,
      }
    },
    step: (state) => {
      let D = Float64Array.from(dense.data(state.D))
      const X = keepBetter(signals, D, code(D), dense.data(state.X), m, k, n)
      if (update === 'ksvd') ksvdUpdate(signals, D, X, m, k, n)
      else D = modUpdate(signals, X, m, k, n)
      const replaced = replaceUnused(signals, D, X, m, k, n)
      const objective = objectiveOf(D, X)
      const change = Math.abs(state.objective - objective) / Math.max(state.objective, Number.MIN_VALUE)
      return {
        t: state.t + 1,
        D: dense.mat(D, m, k),
        X: dense.mat(X, k, n),
        objective,
        replaced,
        converged: replaced === 0 && (change < tolerance || objective <= exact),
        diverged: !Number.isFinite(objective),
      }
    },
  }
}

/**
 * Dictionary learning run to the tolerance or for at most `maxSteps` rounds (default 50): the one-call form of
 * `dictionaryLearningSteps`, as scikit-learn's `DictionaryLearning` with OMP coding (which takes signals and atoms as
 * rows instead).
 *
 * @param Y The training signals $\Ymat$, $m \times n$, one signal per column.
 * @param options The options of `dictionaryLearningSteps`, the step budget and the random stream.
 * @param options.maxSteps The most rounds of coding and update. Default 50.
 * @param options.stream The stream for the random initial dictionary.
 * @returns The dictionary $\Dmat$, the codes $\Xmat$, the objective and the rounds taken.
 *
 * @example Recover the atoms that generated the data
 * // Signals in R^3, each one of the four atoms of D0 times a coefficient. The learnt atoms match D0's up to order
 * // and sign, so every row and column of D0^T D holds one entry of magnitude 1.
 * const D0 = normaliseAtoms([[1, 0, 1, 1], [0, 1, 1, -1], [0, 0, 1, 1]])
 * const coefs = [1, -2, 0.5, 3, -1.5]
 * const Y = toArray(D0).map((row) => row.flatMap((v) => coefs.map((c) => c * v)))
 * const { D, objective, steps } = dictionaryLearning(Y, { atoms: 4, sparsity: 1 })
 * print('objective =', objective, 'after', steps, 'rounds')
 * print('D0^T D =', matmul(transpose(D0), D))
 */
export function dictionaryLearning(
  Y: MatrixLike,
  options: DictionaryLearningOptions & { maxSteps?: Size; stream?: Stream },
): { D: Tensor; X: Tensor; objective: number; steps: Size } {
  const s = run(dictionaryLearningSteps(Y, options), undefined, options.maxSteps ?? 50, { stream: options.stream })
  return { D: s.D, X: s.X, objective: s.objective, steps: s.t }
}
