/**
 * `aifn-compute/numerics/polynomial`: polynomials in descending powers (numpy's and scipy.signal's convention), real or
 * complex128: `polyval` (Horner), `polyDerivative`, `polyMul` (convolution), `polyDivide` (deconvolution), `roots`
 * (companion-matrix eigenvalues; `polynomialRoots` flags a QR failure instead of throwing) and `companionMatrix`, `polyFromRoots`, the partial-fraction expansions `residue`
 * (in s) and `residuez` (in z⁻¹), and `complexVector` for `ComplexLike` inputs.
 */

export {
  companionMatrix,
  complexVector,
  polyDerivative,
  polyDivide,
  polyFromRoots,
  polyMul,
  polynomialRoots,
  polyval,
  residue,
  residuez,
  roots,
  type ComplexLike,
  type PartialFractions,
  type PolynomialRoots,
  type ResidueOptions,
} from './polynomial'
export { polynomialFunctions } from './registry'
