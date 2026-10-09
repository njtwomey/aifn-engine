/**
 * `aifn-compute/foundation/errors`: the error hierarchy of aifn (design K §3.3), one class per kind of failure.
 *
 * - The base class `AifnError`, which every deliberate error extends; its `op` names the operation that raised it.
 * - Bad input: `ShapeError` (shapes that do not fit, with the offending `shapes`), `DTypeError` (a dtype not
 *   accepted) and `DomainError` (parameters outside their domain).
 * - Failed computation: `NumericalError`, whose `kind` (a `NumericalKind`) says why, and `NotDifferentiableError`, a
 *   derivative requested through an operation without a rule.
 *
 * Messages name the operation first. The error kinds as plain data are types in `aifn-compute/foundation/contracts`.
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
