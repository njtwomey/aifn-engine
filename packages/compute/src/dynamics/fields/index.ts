/**
 * `aifn-compute/dynamics/fields`: vector and scalar fields on $\reals^n$ and the geometry of their flows
 * $\xvec' = \fvec(\xvec)$.
 *
 * - Calculus by autodiff: `jacobianAt`, `divergence`, `curl` (in two and three dimensions) and `gradientAt` at a
 *   point; the fields built from a scalar, `gradientField` ($\mp\nabla V$) and `hamiltonianField` (the symplectic
 *   field of $H(\qvec, \pvec)$).
 * - Flows by fixed-step RK4: `flowMap` (where a point goes in time $t$), `trajectory` (every step of the way),
 *   `streamline` and `streamlines` (curves through seeds, stopping at a box's edge or a fixed point); `autonomous`
 *   adapts a field to the solvers of `aifn-compute/dynamics/ode`.
 * - Transport by the method of characteristics: `transportDensity` and `transportDensityFrames` evaluate a density
 *   carried by the flow (the continuity equation) at given points; `pushForwardDensity` and
 *   `pushForwardDensityFrames` move samples instead, the latter with the log-volume change along each.
 * - Fixed points: `fixedPoints` finds them in a box by Newton from a grid of seeds; `linearise` and `classifyLinear`
 *   name and judge them from the Jacobian's eigenvalues (node, spiral, saddle, centre, ...); `invariantManifolds`
 *   traces the stable and unstable manifolds; `lyapunovDerivative` gives $\dot{V} = \nabla V \cdot \fvec$.
 * - Grids: `gridAxes` and `sampleScalar` sample a scalar field on a rectangular grid of the plane.
 *
 * Fields are functions of a rank-1 tensor written with the tensor primitives, so that every derivative is taken by
 * `aifn-compute/foundation/autodiff`; closed forms can be given where they save time (the divergence of the transport
 * functions). Drawing helpers (direction fields, contours, nullclines) live in the lab.
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
