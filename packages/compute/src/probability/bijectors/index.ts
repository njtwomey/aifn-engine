/**
 * `aifn-compute/probability/bijectors`: bijectors and supports: intervals (`interval`, `REALS`, `POSITIVE`, `UNIT`),
 * invertible maps with log-Jacobians (exp, log, softplus, sigmoid, affine, power, the normal CDF, chains) and
 * many-to-one maps with their branches (`bijectorRegistry` lists the bijectors and factories), the ordered bijector onto increasing vectors (ordinal thresholds), the affine and additive coupling bijectors of normalising flows (RealNVP, NICE), for transformed distributions and constrained parameters, and `transformLogDensity`, a log-density reparameterised
 * through a bijector or a vector change of variables (non-centred parameterisations).
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
