/**
 * `aifn-compute/numerics/interpolate`: splines and interpolation.
 *
 * ```ts
 * evaluatePiecewise(cubicSpline(x, y, { bc: 'natural' }), grid)
 * bsplineBasis(grid, uniformKnots(0, 1, 10), 3) // [m, 13]
 * pspline(x, y, { lambda: 'gcv' }).evaluate(grid)
 * ```
 *
 * - Piecewise polynomials (scipy's `PPoly` layout): `piecewisePolynomial`, `evaluatePiecewise` (with derivatives),
 *   `integratePiecewise`.
 * - Interpolants: `linearInterpolant`, `cubicSpline` (not-a-knot, natural, clamped, periodic, given end slopes or
 *   curvatures), `naturalCubicSpline`, `hermiteSpline`, `pchip`, `akima` (and `makima`); polynomial interpolation
 *   `interpolatingPolynomial` (Newton and barycentric forms), `chebyshevNodes`, `lebesgueFunction`.
 * - Smoothing: `smoothingSpline` (Reinsch), `pspline` (λ given or by GCV) and `psplineGcvPath`, `leastSquaresSpline`,
 *   `thinPlateSpline`, `thinPlateRegressionBasis` (Wood, 2003).
 * - B-splines: `bsplineBasis` (Cox–de Boor, derivatives, extrapolation), `bsplineCount`, `bspline`, `uniformKnots`,
 *   `clampedKnots`, `cyclicBsplineBasis`, `tensorProductBasis`.
 * - Penalties: `differenceMatrix`, `differencePenalty`, `cyclicDifferenceMatrix`, `cyclicDifferencePenalty`,
 *   `derivativePenalty` (∫f⁽ᵐ⁾², exact), `tensorProductPenalties`.
 */

export {
  akima,
  cubicSpline,
  evaluatePiecewise,
  hermiteSpline,
  integratePiecewise,
  linearInterpolant,
  naturalCubicSpline,
  pchip,
  piecewisePolynomial,
  smoothingSpline,
  type EndCondition,
  type PiecewisePolynomial,
  type SmoothingSpline,
} from './piecewise'
export { chebyshevNodes, interpolatingPolynomial, lebesgueFunction, type InterpolatingPolynomial } from './polynomial'
export {
  bspline,
  bsplineBasis,
  bsplineCount,
  clampedKnots,
  cyclicBsplineBasis,
  cyclicDifferenceMatrix,
  cyclicDifferencePenalty,
  derivativePenalty,
  differenceMatrix,
  differencePenalty,
  leastSquaresSpline,
  pspline,
  psplineGcvPath,
  tensorProductBasis,
  tensorProductPenalties,
  uniformKnots,
  type BSpline,
  type Extrapolation,
  type PSplineFit,
  type PSplineOptions,
} from './bspline'
export {
  thinPlateRegressionBasis,
  thinPlateSpline,
  type ThinPlateRegressionBasis,
  type ThinPlateSpline,
} from './thinplate'
export { interpolateFunctions } from './registry'
