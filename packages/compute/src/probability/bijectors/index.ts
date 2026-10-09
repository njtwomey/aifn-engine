/**
 * `aifn-compute/probability/bijectors`: invertible maps of the real line with their log-Jacobians, for transformed
 * distributions, constrained parameters and normalising flows.
 *
 * - Intervals and supports: `interval`, the constants `REALS`, `POSITIVE` and `UNIT_INTERVAL`, `intervalInside`,
 *   `supportInterval` and `intervalSupport` (between an `Interval` and a distribution's `Support`),
 *   `supportInteriorPoint` and `formatInterval`.
 * - Scalar bijectors, each with its inverse, $\log \lvert f'(x) \rvert$, direction, domain and codomain:
 *   `expBijector`, `logBijector`, `sigmoidBijector`, `tanhBijector`, `softplusBijector`, `normalCdfBijector`, and the
 *   factories `affineBijector`, `powerBijector` and `chainBijectors` (a composition, checked link by link).
 * - Vector bijectors along the last axis: `orderedBijector` onto increasing vectors (ordinal thresholds), and
 *   `affineCouplingBijector`, the affine (RealNVP) and additive (NICE) coupling layers of normalising flows.
 * - Many-to-one maps given by monotone branches: `squareMap`, `asManyToOne`; `imageOf` and `branchImages` give the
 *   image of an interval, which is how `Transformed` and `Pushforward` find their supports.
 * - `transformLogDensity`: a log-density reparameterised through a bijector or a vector change of variables
 *   (non-centred parameterisations), with the log-Jacobian added.
 * - `bijectorRegistry` lists the bijectors and factories with their metadata; `bijectorFunctions` the module's
 *   functions.
 *
 * Every map is a composition of tensor primitives, so it is differentiable and accepts numbers, tensors and traced
 * values. Bad parameters and intervals outside a map's domain throw a `DomainError`.
 */

export {
  affineBijector,
  affineCouplingBijector,
  asManyToOne,
  branchImages,
  chainBijectors,
  expBijector,
  formatInterval,
  imageOf,
  interval,
  intervalInside,
  intervalSupport,
  logBijector,
  normalCdfBijector,
  orderedBijector,
  POSITIVE,
  powerBijector,
  REALS,
  sigmoidBijector,
  softplusBijector,
  squareMap,
  supportInteriorPoint,
  supportInterval,
  tanhBijector,
  UNIT_INTERVAL,
  type Bijector,
  type Branch,
  type CouplingParameters,
  type Interval,
  type ManyToOneMap,
  type OrderedOptions,
} from './maps'
export { transformLogDensity, type Reparameterisation, type TransformedLogDensity } from './reparameterise'
export { bijectorFunctions, bijectorRegistry } from './registry'
