/**
 * The algorithms of `aifn-compute/numerics/linalg`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cholesky from './cholesky'
import * as distances from './distances'
import * as eig from './eig'
import * as inverseSqrt from './inverseSqrt'
import * as iterative from './iterative'
import * as lanczos from './lanczos'
import * as levinson from './levinson'
import * as lu from './lu'
import * as lyapunov from './lyapunov'
import * as orthogonalise from './orthogonalise'
import * as products from './products'
import * as qr from './qr'
import * as riccati from './riccati'
import * as svd from './svd'

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/linalg')

algorithm(
  {
    key: 'kleinmanIteration',
    name: 'Kleinman iteration',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.kleinmanIteration,
)
algorithm(
  {
    key: 'riccatiMatrixSign',
    name: 'Matrix sign function',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.riccatiMatrixSign,
)
algorithm(
  {
    key: 'riccatiRecursion',
    name: 'Riccati recursion',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.riccatiRecursion,
)
algorithm(
  {
    key: 'riccatiDoubling',
    name: 'Structured doubling',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.riccatiDoubling,
)

// ── Factorisations and iterations a note walks through ───────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'gramSchmidtSteps',
    name: 'Gram–Schmidt',
    summary: 'Orthonormalise the columns one at a time, classical or modified; after n steps A = QR.',
    problem: 'system',
    state: { iterate: 'Q', objective: 'orthogonalityError', flags: [] },
    notes: ['gram-schmidt', 'qr-decomposition', 'orthogonal-projection'],
    cite: ['golub2013', 'bjorck1967'],
  },
  orthogonalise.gramSchmidtSteps,
)
algorithm(
  {
    key: 'householderSteps',
    name: 'Householder QR',
    summary: 'One reflector per column maps it onto a multiple of e₁ below the diagonal; QR = A throughout.',
    problem: 'system',
    state: { iterate: 'R', objective: 'residual', flags: [] },
    notes: ['qr-decomposition'],
    cite: ['golub2013'],
  },
  qr.householderSteps,
)
algorithm(
  {
    key: 'jacobiSteps',
    name: 'Jacobi iteration',
    summary: 'x ← D⁻¹(b − (L + U)x), every component from the previous iterate.',
    problem: 'system',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'diverged'] },
    notes: ['iterative-linear-solvers'],
    cite: ['saad2003'],
  },
  iterative.jacobiSteps,
)
algorithm(
  {
    key: 'gaussSeidelSteps',
    name: 'Gauss–Seidel and SOR',
    summary: 'Sweeps that use each updated component at once, with optional over-relaxation ω.',
    problem: 'system',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'diverged'] },
    notes: ['iterative-linear-solvers'],
    cite: ['saad2003'],
  },
  iterative.gaussSeidelSteps,
)
algorithm(
  {
    key: 'powerIterationSteps',
    name: 'Power iteration',
    summary: 'v ← Av/‖Av‖ with the Rayleigh quotient as the eigenvalue estimate; inverse iteration with a shift.',
    problem: 'system',
    state: { iterate: 'vector', objective: 'residualNorm', flags: ['converged', 'diverged'] },
    notes: ['eigendecomposition', 'spectral-theorem'],
    cite: ['golub2013'],
  },
  iterative.powerIterationSteps,
)

// ── Functions (the factorisations themselves are primitives, listed in the primitive table) ──────────────────────────

const fn = definer<FunctionInfo>('function', 'numerics/linalg')

fn(
  { key: 'gramSchmidt', name: 'Gram–Schmidt QR', role: 'solver', notes: ['gram-schmidt', 'qr-decomposition'] },
  orthogonalise.gramSchmidt,
)
fn(
  {
    key: 'solveStationary',
    name: 'Solve by Jacobi or Gauss–Seidel',
    role: 'solver',
    notes: ['iterative-linear-solvers'],
  },
  iterative.solveStationary,
)
fn(
  {
    key: 'choleskySolve',
    name: 'Solve by Cholesky',
    role: 'solver',
    notes: ['cholesky-decomposition', 'positive-definite-matrices'],
  },
  cholesky.choleskySolve,
)
fn(
  {
    key: 'choleskyLogDet',
    name: 'Log-determinant by Cholesky',
    role: 'property',
    notes: ['cholesky-decomposition', 'determinant'],
  },
  cholesky.choleskyLogDet,
)
fn({ key: 'luFactor', name: 'LU factor (packed)', role: 'solver', notes: ['lower-upper-decomposition'] }, lu.luFactor)
fn(
  {
    key: 'solve',
    name: 'Solve a linear system',
    role: 'solver',
    notes: ['lower-upper-decomposition', 'matrix-inverse'],
  },
  lu.solve,
)
fn({ key: 'inverse', name: 'Matrix inverse', role: 'solver', notes: ['matrix-inverse'] }, lu.inverse)
fn({ key: 'signDet', name: 'Sign of the determinant', role: 'property', notes: ['determinant'] }, lu.signDet)
fn(
  {
    key: 'lstsq',
    name: 'Least squares',
    role: 'solver',
    notes: ['pseudoinverse', 'qr-decomposition', 'singular-value-decomposition'],
  },
  svd.lstsq,
)
fn(
  {
    key: 'pinv',
    name: 'Moore–Penrose pseudoinverse',
    role: 'solver',
    notes: ['pseudoinverse', 'singular-value-decomposition'],
  },
  svd.pinv,
)
fn(
  {
    key: 'conditionNumber',
    name: 'Condition number',
    role: 'property',
    notes: ['condition-number', 'numerical-stability-and-conditioning'],
  },
  svd.conditionNumber,
)
fn(
  {
    key: 'symmetricInverseSqrt',
    name: 'Symmetric inverse square root',
    summary: 'S^{−1/2} = V Λ^{−1/2} Vᵀ of a symmetric positive-definite matrix, by its eigendecomposition.',
    role: 'transform',
    notes: ['eigendecomposition'],
  },
  inverseSqrt.symmetricInverseSqrt,
)
fn({ key: 'kron', name: 'Kronecker product', role: 'construction' }, products.kron)
fn({ key: 'matrixTrace', name: 'Trace', role: 'property' }, products.matrixTrace)
fn({ key: 'normFrobenius', name: 'Frobenius norm', role: 'property', notes: ['matrix-norms'] }, products.normFrobenius)
fn(
  {
    key: 'eig',
    name: 'General eigendecomposition',
    summary: 'Eigenvalues and vectors of a real non-symmetric matrix by Francis QR.',
    role: 'solver',
    notes: ['eigendecomposition'],
    cite: ['golub2013'],
  },
  eig.eig,
)
fn(
  { key: 'pairwiseDistances', name: 'Pairwise distances', role: 'construction', notes: ['vector-norms'] },
  distances.pairwiseDistances,
)
fn(
  { key: 'squaredDistances', name: 'Squared Euclidean distances', role: 'construction', notes: ['vector-norms'] },
  distances.squaredDistances,
)
fn(
  {
    key: 'lyapunov',
    name: 'Lyapunov equation',
    summary: 'AX + XAᵀ + Q = 0 (continuous) or AXAᵀ − X + Q = 0 (discrete).',
    role: 'solver',
    notes: ['lyapunov-stability', 'controllability-and-observability'],
  },
  lyapunov.lyapunov,
)
fn(
  {
    key: 'levinsonDurbin',
    name: 'Levinson–Durbin recursion',
    summary: 'The Toeplitz Yule–Walker system solved order by order, with the reflection coefficients.',
    role: 'solver',
    notes: ['linear-prediction', 'autocorrelation-and-partial-autocorrelation', 'autoregressive-model'],
  },
  levinson.levinsonDurbin,
)
fn(
  {
    key: 'eigsh',
    name: 'Lanczos eigensolver',
    summary: 'A few extreme eigenpairs of a large symmetric operator by thick-restart Lanczos.',
    role: 'solver',
    notes: ['eigendecomposition', 'spectral-theorem'],
  },
  lanczos.eigsh,
)

/** The functions of the module (factorisations are primitives), keyed by name. */
export const linalgFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    orthogonalise,
    iterative,
    cholesky,
    lu,
    svd,
    products,
    eig,
    distances,
    lyapunov,
    levinson,
    lanczos,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

/** Every algorithm of the module, keyed by factory name. */
export const linalgAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', riccati, orthogonalise, qr, iterative) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
