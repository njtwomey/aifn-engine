/**
 * The error kinds aifn raises, as data (design K §3.3). The classes (`AifnError` and its subclasses) are defined by
 * `aifn-compute/foundation/errors`; these unions let a caller, a test or a worker message name an error without
 * importing a class.
 */

/** Why a numerical computation failed. */
export type NumericalErrorKind = 'singular' | 'not-finite' | 'not-converged' | 'not-positive-definite' | 'degenerate'

/**
 * The error classes: shapes that do not align, an unsupported dtype, differentiating through an operation without a
 * derivative, a numerical failure, or a parameter outside its domain (sampler and distribution parameters).
 */
export type ErrorKind = 'ShapeError' | 'DTypeError' | 'NotDifferentiableError' | 'NumericalError' | 'DomainError'

/** An error as plain data, e.g. in a worker's reply. */
export interface ErrorInfo {
  /** The class of the error. */
  readonly kind: ErrorKind
  /** The operation that raised it, e.g. `matmul`. */
  readonly op: string
  /** The error's message, as the class would show it. */
  readonly message: string
  /** Why a `NumericalError` failed (its `kind`); absent for the other classes. */
  readonly numerical?: NumericalErrorKind
}
