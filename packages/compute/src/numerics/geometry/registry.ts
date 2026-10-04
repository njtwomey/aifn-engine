/**
 * The functions of `aifn-compute/numerics/geometry`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as contours from './contours'
import * as ellipse from './ellipse'
import * as grids from './grids'
import * as planar from './planar'
import * as projective from './projective'
import * as simplex from './simplex'

const fn = definer<FunctionInfo>('function', 'numerics/geometry')

fn(
  {
    key: 'covarianceEllipse',
    name: 'Covariance ellipse',
    role: 'construction',
    notes: ['multivariate-normal-distribution', 'eigendecomposition'],
  },
  ellipse.covarianceEllipse,
)
fn(
  {
    key: 'precisionEllipse',
    name: 'Precision ellipse',
    role: 'construction',
    notes: ['multivariate-normal-distribution'],
  },
  ellipse.precisionEllipse,
)
fn(
  {
    key: 'massToRadius',
    name: 'Mahalanobis radius holding a probability mass',
    role: 'transform',
    notes: ['multivariate-normal-distribution', 'chi-squared-distribution'],
  },
  ellipse.massToRadius,
)
fn(
  {
    key: 'convexHull',
    name: 'Convex hull',
    role: 'construction',
    notes: ['convex-sets-and-functions', 'receiver-operating-characteristic-convex-hull'],
  },
  planar.convexHull,
)
fn({ key: 'polygonArea', name: 'Polygon area (shoelace)', role: 'property' }, planar.polygonArea)
fn({ key: 'polygonCentroid', name: 'Polygon centroid', role: 'property' }, planar.polygonCentroid)
fn({ key: 'pointInPolygon', name: 'Point in polygon', role: 'property' }, planar.pointInPolygon)
fn({ key: 'contourSegments', name: 'Contour segments (marching squares)', role: 'transform' }, contours.contourSegments)
fn({ key: 'contourLines', name: 'Contour lines', role: 'transform' }, contours.contourLines)
fn({ key: 'contourLevels', name: 'Contour levels', role: 'construction' }, contours.contourLevels)
fn(
  { key: 'simplexVertices', name: 'Simplex vertices', role: 'construction', notes: ['sampling-the-simplex'] },
  simplex.simplexVertices,
)
fn(
  {
    key: 'barycentricToCartesian',
    name: 'Barycentric to Cartesian',
    role: 'transform',
    notes: ['sampling-the-simplex', 'dirichlet-distribution'],
  },
  simplex.barycentricToCartesian,
)
fn(
  {
    key: 'cartesianToBarycentric',
    name: 'Cartesian to barycentric',
    role: 'transform',
    notes: ['sampling-the-simplex'],
  },
  simplex.cartesianToBarycentric,
)
fn(
  { key: 'simplexGrid', name: 'Grid on the simplex', role: 'construction', notes: ['dirichlet-distribution'] },
  simplex.simplexGrid,
)
fn({ key: 'grid2d', name: 'Two-dimensional grid', role: 'construction' }, grids.grid2d)
fn({ key: 'evaluateGrid', name: 'Evaluate on a grid', role: 'transform' }, grids.evaluateGrid)

// ── Projective geometry ──────────────────────────────────────────────────────────────────────────────────────────────

const H = ['homography']
const EPI = ['epipolar-geometry-and-stereo']
const CAM = ['pinhole-camera-model']
fn(
  {
    key: 'normalisePoints',
    name: 'Hartley normalisation',
    role: 'transform',
    notes: [...H, ...EPI],
    cite: ['hartley1997'],
  },
  projective.normalisePoints,
)
fn(
  {
    key: 'homography',
    name: 'Homography (normalised DLT)',
    summary: 'The 3 × 3 projective map between two planes from four or more correspondences.',
    role: 'fit',
    notes: H,
    cite: ['hartley2004'],
  },
  projective.homography,
)
fn({ key: 'applyHomography', name: 'Apply a homography', role: 'transform', notes: H }, projective.applyHomography)
fn({ key: 'transferError', name: 'Homography transfer error', role: 'property', notes: H }, projective.transferError)
fn(
  {
    key: 'fundamentalMatrix',
    name: 'Fundamental matrix (normalised eight-point)',
    tex: 'x_2^\top F x_1 = 0',
    role: 'fit',
    notes: EPI,
    cite: ['hartley1997', 'longuethiggins1981'],
  },
  projective.fundamentalMatrix,
)
fn({ key: 'epipolarLines', name: 'Epipolar lines', role: 'transform', notes: EPI }, projective.epipolarLines)
fn(
  { key: 'sampsonDistance', name: 'Sampson distance', role: 'property', notes: EPI, cite: ['hartley2004'] },
  projective.sampsonDistance,
)
fn({ key: 'cameraMatrix', name: 'Camera matrix K[R | t]', role: 'construction', notes: CAM }, projective.cameraMatrix)
fn({ key: 'projectPoints', name: 'Project scene points', role: 'transform', notes: CAM }, projective.projectPoints)
fn(
  {
    key: 'triangulate',
    name: 'Linear triangulation',
    role: 'solver',
    notes: [...EPI, 'structure-from-motion'],
    cite: ['hartley1997b'],
  },
  projective.triangulate,
)
fn(
  { key: 'rotationMatrix', name: 'Rotation matrix (Rodrigues)', role: 'construction', notes: CAM },
  projective.rotationMatrix,
)

/** The functions of the module, keyed by name. */
export const geometryFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', projective, ellipse, planar, contours, simplex, grids) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
