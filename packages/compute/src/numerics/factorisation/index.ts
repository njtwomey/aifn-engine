/**
 * `aifn-compute/numerics/factorisation`: matrix factorisations and projections under constraints. Non-negative matrix
 * factorisation X ≈ WH by Lee–Seung multiplicative updates (squared error or generalised KL) or HALS, stepped
 * (`nmfSteps`) or run (`nmf`); random projections (Gaussian, sparse) with the Johnson–Lindenstrauss dimension and a
 * distortion measure; canonical correlation analysis, classical or regularised.
 */

export { nmf, nmfSteps, type NmfLoss, type NmfOptions, type NmfSolver, type NmfState } from './nmf'
export {
  distanceDistortion,
  johnsonLindenstraussDimension,
  johnsonLindenstraussEpsilon,
  randomProjection,
  randomProjectionMatrix,
  type ProjectionKind,
  type RandomProjectionOptions,
} from './projection'
export { canonicalCorrelation, type Cca, type CcaOptions } from './cca'
export { factorisationAlgorithms, factorisationFunctions } from './registry'
