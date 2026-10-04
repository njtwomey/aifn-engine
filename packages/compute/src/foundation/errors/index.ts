/**
 * `aifn-compute/foundation/errors`: the error hierarchy of aifn (design K §3.3). `AifnError` and its subclasses `ShapeError`,
 * `DTypeError`, `NotDifferentiableError`, `NumericalError` (with a `NumericalKind`) and `DomainError`; every error
 * names the operation that raised it. The error kinds as plain data are types in `aifn-compute/foundation/contracts`.
 */

export {
  AifnError,
  DomainError,
  DTypeError,
  NotDifferentiableError,
  NumericalError,
  ShapeError,
  type NumericalKind,
} from './errors'
