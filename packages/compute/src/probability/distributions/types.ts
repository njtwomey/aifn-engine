/**
 * The distribution protocol: plain objects whose parameters are public fields and whose derived quantities are
 * methods. See `README.md` (Distributions) and `docs/aifn-plan.md` §5.2.
 *
 * The types are defined once, in `aifn-compute/foundation/contracts` (`distribution.ts`), and re-exported here under this module's
 * names, with `LogDensity`, the protocol of an unnormalised log-density that samplers and variational inference consume.
 */

export type {
  AnyMultivariate,
  AnyUnivariate,
  Distribution,
  EventKind,
  ExponentialFamily,
  LogDensity,
  Kind,
  Multivariate,
  SampleKind,
  SampleOptions,
  Support,
  TypedMultivariate,
  TypedUnivariate,
  Univariate,
} from 'aifn-compute/foundation/contracts'
