/**
 * `aifn-compute/dynamics/fields`: vector and scalar fields and the geometry of their flows: gradient, divergence, curl and
 * Jacobians; flow maps, trajectories, streamlines and density transport; fixed points, their classification,
 * linearisation and invariant manifolds; scalar fields sampled on a grid. Drawing helpers (direction fields,
 * contours, nullclines) live in the lab.
 */

export {
  curl,
  divergence,
  gradientAt,
  gradientField,
  hamiltonianField,
  jacobianAt,
  type ScalarField,
  type VectorField,
} from './calculus'
export {
  autonomous,
  flowMap,
  pushForwardDensity,
  pushForwardDensityFrames,
  streamline,
  streamlines,
  trajectory,
  transportDensity,
  transportDensityFrames,
  type Box,
  type FlowOptions,
  type StreamlineOptions,
  type TransportFrameOptions,
  type TransportOptions,
} from './flow'
export {
  classifyLinear,
  fixedPoints,
  invariantManifolds,
  linearise,
  lyapunovDerivative,
  type Classification,
  type FixedPoint,
  type FixedPointKind,
  type FixedPointOptions,
  type Manifolds,
} from './fixed'
export { gridAxes, sampleScalar, type Grid2 } from './grid'
export { fieldsFunctions } from './registry'
