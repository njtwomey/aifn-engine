/**
 * `aifn-compute/numerics/linalg`: dense linear algebra on `aifn-compute/foundation/tensor` matrices.
 *
 * - Factorisations that report failure instead of returning NaN: `cholesky` (with jitter; factored once),
 *   `luFactor` (packed, for repeated solves) and `lu` ($\Lmat$, $\Umat$, $\Pmat$ unpacked), `qr`, `eigh`
 *   (descending eigenvalues, vectors as columns), `svd` (thin, one-sided Jacobi), and `eig` for the general
 *   (non-symmetric) real eigenproblem (complex128 `values` and `vectors`, `converged`; its Francis QR is also what
 *   `aifn-compute/numerics/polynomial`'s `roots` uses).
 * - Solves and functions: `solveTriangular`, `choleskySolve`, `luSolve`, `solve`, `inverse`, `det`, `logDet`,
 *   `choleskyLogDet`, `pinv`, `lstsq`, `kron`, `matrixTrace`, `normFrobenius`, `conditionNumber`, `expm` (Padé,
 *   scaling and squaring), `symmetricInverseSqrt` ($\Smat^{-1/2}$ of a symmetric positive-definite matrix, by
 *   `eigh`). Solvers throw `LinAlgError` for a singular system.
 * - Matrix equations: `lyapunov` (continuous Lyapunov or discrete Stein); the algebraic Riccati equations as traceable
 *   algorithms, `kleinmanIteration` and `riccatiMatrixSign` (CARE), `riccatiRecursion` and `riccatiDoubling` (DARE);
 *   the Toeplitz Yule–Walker system by `levinsonDurbin`.
 * - Distances between point sets: `pairwiseDistances` (Euclidean, squared, Manhattan, Chebyshev, Minkowski, cosine)
 *   and `squaredDistances`.
 * - For inner loops on row-major `Float64Array`s: `solveDense`, and `factorDense` + `solveFactored` for repeated
 *   solves (all report `singular` instead of throwing).
 * - Large symmetric eigenproblems, matrix-free: `eigsh` (thick-restart Lanczos, top-k / bottom-k / largest-magnitude)
 *   on a `LinearOperator` (a matrix or a function $\vvec \mapsto \Amat\vvec$; `operatorOf`), shared with the
 *   iterative solvers.
 * - Differentiable in reverse and forward mode, and batched by `vmap` (primitives with vjp and jvp rules, or
 *   compositions of them): `cholesky`'s $\Lmat$, `solveTriangular`, `choleskySolve`, `choleskyLogDet`, `luSolve` (in
 *   $\Amat$ through the factor, and in $\Bmat$), `solve`, `inverse`, `det`, `logDet`, `lu`'s $\Lmat$ and $\Umat$,
 *   `eigh`, `svd`, `qr` (reduced), `expm` (the Fréchet derivative), `kron`, `matrixTrace`, `normFrobenius`.
 *   Derivatives of `solve`, `det` and `logDet` reuse the one LU factor. Under `vmap` each factorisation runs once over
 *   the whole batch: a batched kernel loops over the contiguous matrices in one call (`kernelBatch` in `rules.ts`),
 *   and a solve with an unbatched matrix folds the batch into the right-hand side's columns. Differentiating a flagged
 *   result throws: a failed Cholesky, a singular LU or determinant, a non-converged or degenerate `eigh`/`svd` (unless
 *   the function is invariant), a rank-deficient `qr`. The general `eig` has no derivative (its values may be complex)
 *   and refuses traced input. Rule sources are cited at each primitive; the shared rule helpers are in `rules.ts`.
 * - Step-through forms (traceable algorithms) of what the notes walk through: `gramSchmidtSteps` (classical or
 *   modified; `gramSchmidt` runs it), `householderSteps` (the step `qr` itself runs), `jacobiSteps` and
 *   `gaussSeidelSteps` (with SOR; `solveStationary` runs them) and `powerIterationSteps` (and inverse iteration).
 * - Closed forms on $2 \times 2$ tuples: `det2`, `apply2`, `inv2`, `eigh2`, `eig2`, `cholesky2`, `svd2`.
 */

export { LinAlgError } from './dense'
export { solveTriangular, type TriangularOptions } from './triangular'
export { cholesky, choleskyLogDet, choleskySolve, type Cholesky, type CholeskyOptions } from './cholesky'
export { det, inverse, logDet, lu, luFactor, luSolve, signDet, solve, type LU, type LuFactor } from './lu'
export { householderSteps, qr, type HouseholderState, type QR } from './qr'
export { eigh, type Eigh } from './eigh'
export { symmetricInverseSqrt, type InverseSqrtOptions } from './inverseSqrt'
export { conditionNumber, lstsq, pinv, svd, type LeastSquares, type SVD } from './svd'
export { kron, matrixTrace, normFrobenius } from './products'
export { eig, type Eigen } from './eig'
export { expm, type MatrixExponential } from './expm'
export { pairwiseDistances, rowDistance, squaredDistances, squaredRowDistance, type PairwiseMetric } from './distances'
export { factorDense, solveDense, solveFactored, type DenseFactor, type DenseSolution } from './solveDense'
export { apply2, cholesky2, det2, eig2, eigh2, inv2, svd2, type Eig2, type Mat2, type Vec2 } from './small'
export { lyapunov, type LyapunovSolution } from './lyapunov'
export {
  kleinmanIteration,
  riccatiDoubling,
  riccatiMatrixSign,
  riccatiRecursion,
  type DoublingState,
  type RiccatiFailure,
  type RiccatiOptions,
  type RiccatiProblem,
  type RiccatiState,
  type SignState,
} from './riccati'
export { levinsonDurbin, type LevinsonDurbin } from './levinson'
export { eigsh, operatorOf, type EigshOptions, type EigshResult, type EigshWhich, type LinearOperator } from './lanczos'
export {
  gramSchmidt,
  gramSchmidtSteps,
  type GramSchmidt,
  type GramSchmidtOptions,
  type GramSchmidtState,
  type GramSchmidtVariant,
} from './orthogonalise'
export {
  gaussSeidelSteps,
  jacobiSteps,
  powerIterationSteps,
  solveStationary,
  type PowerIterationState,
  type StationaryOptions,
  type StationarySolution,
  type StationaryState,
} from './iterative'
export { linalgAlgorithms, linalgFunctions } from './registry'
