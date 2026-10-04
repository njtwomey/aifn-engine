/**
 * The functions of `aifn-compute/numerics/interpolate`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bspline from './bspline'
import * as piecewise from './piecewise'
import * as polynomial from './polynomial'
import * as thinplate from './thinplate'

const fn = definer<FunctionInfo>('function', 'numerics/interpolate')
const BS = ['b-splines', 'splines-and-interpolation']
const PS = ['p-splines', 'smoothing-and-penalised-splines']
const PW = ['piecewise-polynomials-and-continuity', 'splines-and-interpolation']

fn(
  {
    key: 'interpolatingPolynomial',
    name: 'Interpolating polynomial',
    role: 'fit',
    notes: ['polynomial-interpolation-and-runge'],
    cite: ['trefethen2019'],
  },
  polynomial.interpolatingPolynomial,
)
fn(
  {
    key: 'chebyshevNodes',
    name: 'Chebyshev nodes',
    role: 'construction',
    notes: ['polynomial-interpolation-and-runge'],
    cite: ['trefethen2019'],
  },
  polynomial.chebyshevNodes,
)
fn(
  {
    key: 'lebesgueFunction',
    name: 'Lebesgue function',
    role: 'property',
    notes: ['polynomial-interpolation-and-runge'],
    cite: ['trefethen2019'],
  },
  polynomial.lebesgueFunction,
)
fn(
  { key: 'bsplineCount', name: 'Number of B-spline basis functions', role: 'property', notes: BS },
  bspline.bsplineCount,
)
fn(
  {
    key: 'bsplineBasis',
    name: 'B-spline basis (Cox–de Boor)',
    role: 'construction',
    notes: [...BS, 'basis-expansions'],
    cite: ['deboor1978'],
  },
  bspline.bsplineBasis,
)
fn({ key: 'uniformKnots', name: 'Uniform knots', role: 'construction', notes: BS }, bspline.uniformKnots)
fn({ key: 'clampedKnots', name: 'Clamped knots', role: 'construction', notes: BS }, bspline.clampedKnots)
fn(
  {
    key: 'bspline',
    name: 'B-spline curve',
    role: 'construction',
    notes: [...BS, 'splines-for-curves-and-surfaces'],
    cite: ['deboor1978'],
  },
  bspline.bspline,
)
fn(
  { key: 'penalisedLeastSquares', name: 'Penalised least squares', role: 'solver', notes: PS },
  bspline.penalisedLeastSquares,
)
fn(
  { key: 'leastSquaresSpline', name: 'Least-squares spline', role: 'fit', notes: ['basis-expansions', ...BS] },
  bspline.leastSquaresSpline,
)
fn(
  { key: 'differenceMatrix', name: 'Difference matrix', role: 'construction', notes: PS, cite: ['eilers1996'] },
  bspline.differenceMatrix,
)
fn(
  {
    key: 'differencePenalty',
    name: 'Difference penalty',
    tex: 'D_d^\\top D_d',
    role: 'construction',
    notes: PS,
    cite: ['eilers1996'],
  },
  bspline.differencePenalty,
)
fn(
  {
    key: 'cyclicDifferenceMatrix',
    name: 'Cyclic difference matrix',
    role: 'construction',
    notes: ['cyclic-factor-and-by-terms', ...PS],
  },
  bspline.cyclicDifferenceMatrix,
)
fn(
  {
    key: 'cyclicDifferencePenalty',
    name: 'Cyclic difference penalty',
    role: 'construction',
    notes: ['cyclic-factor-and-by-terms', ...PS],
  },
  bspline.cyclicDifferencePenalty,
)
fn(
  {
    key: 'derivativePenalty',
    name: 'Derivative penalty',
    tex: '\\int f^{(m)}(x)^2 dx',
    role: 'construction',
    notes: ['smoothing-splines', 'smoothing-and-penalised-splines', 'curvature-and-fairness'],
  },
  bspline.derivativePenalty,
)
fn(
  {
    key: 'cyclicBsplineBasis',
    name: 'Cyclic B-spline basis',
    role: 'construction',
    notes: ['cyclic-factor-and-by-terms', ...BS],
  },
  bspline.cyclicBsplineBasis,
)
fn(
  {
    key: 'tensorProductBasis',
    name: 'Tensor-product basis',
    role: 'construction',
    notes: ['splines-for-curves-and-surfaces', 'splines-in-machine-learning'],
    cite: ['wood2017'],
  },
  bspline.tensorProductBasis,
)
fn(
  {
    key: 'tensorProductPenalties',
    name: 'Tensor-product penalties',
    role: 'construction',
    notes: ['splines-in-machine-learning'],
    cite: ['wood2017'],
  },
  bspline.tensorProductPenalties,
)
fn(
  {
    key: 'psplineGcvPath',
    name: 'P-spline GCV path',
    summary: 'Generalised cross-validation over a grid of smoothing parameters.',
    role: 'estimator',
    notes: PS,
    cite: ['craven1979', 'eilers1996'],
  },
  bspline.psplineGcvPath,
)
fn(
  {
    key: 'pspline',
    name: 'P-spline',
    summary: 'A B-spline basis with a difference penalty on its coefficients.',
    role: 'fit',
    notes: PS,
    cite: ['eilers1996'],
  },
  bspline.pspline,
)
fn(
  { key: 'piecewisePolynomial', name: 'Piecewise polynomial', role: 'construction', notes: PW },
  piecewise.piecewisePolynomial,
)
fn(
  { key: 'evaluatePiecewise', name: 'Evaluate a piecewise polynomial', role: 'transform', notes: PW },
  piecewise.evaluatePiecewise,
)
fn(
  { key: 'integratePiecewise', name: 'Integrate a piecewise polynomial', role: 'transform', notes: PW },
  piecewise.integratePiecewise,
)
fn({ key: 'linearInterpolant', name: 'Linear interpolant', role: 'fit', notes: PW }, piecewise.linearInterpolant)
fn(
  {
    key: 'hermiteSpline',
    name: 'Cubic Hermite spline',
    role: 'fit',
    notes: ['hermite-and-catmull-rom-splines', ...PW],
  },
  piecewise.hermiteSpline,
)
fn(
  {
    key: 'cubicSpline',
    name: 'Cubic spline',
    role: 'fit',
    notes: ['cubic-spline-interpolation', ...PW],
    cite: ['deboor1978'],
  },
  piecewise.cubicSpline,
)
fn(
  {
    key: 'naturalCubicSpline',
    name: 'Natural cubic spline',
    role: 'fit',
    notes: ['cubic-spline-interpolation', 'smoothing-splines'],
  },
  piecewise.naturalCubicSpline,
)
fn(
  {
    key: 'pchip',
    name: 'PCHIP (monotone cubic)',
    role: 'fit',
    notes: ['monotone-and-shape-preserving-interpolation'],
    cite: ['fritsch1980', 'fritsch1984'],
  },
  piecewise.pchip,
)
fn(
  {
    key: 'akima',
    name: 'Akima spline',
    role: 'fit',
    notes: ['monotone-and-shape-preserving-interpolation'],
    cite: ['akima1970'],
  },
  piecewise.akima,
)
fn(
  {
    key: 'smoothingSpline',
    name: 'Smoothing spline',
    role: 'fit',
    notes: ['smoothing-splines', 'smoothing-and-penalised-splines'],
    cite: ['reinsch1967', 'green1994'],
  },
  piecewise.smoothingSpline,
)
fn(
  {
    key: 'thinPlateSpline',
    name: 'Thin-plate spline',
    role: 'fit',
    notes: ['thin-plate-regression-splines'],
    cite: ['duchon1977'],
  },
  thinplate.thinPlateSpline,
)
fn(
  {
    key: 'thinPlateRegressionBasis',
    name: 'Thin-plate regression basis',
    role: 'construction',
    notes: ['thin-plate-regression-splines'],
    cite: ['wood2003'],
  },
  thinplate.thinPlateRegressionBasis,
)

/** The functions of the module, keyed by name. */
export const interpolateFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', polynomial, bspline, piecewise, thinplate) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
