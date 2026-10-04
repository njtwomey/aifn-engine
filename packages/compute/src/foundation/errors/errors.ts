/**
 * The error hierarchy of aifn: one base class, `AifnError`, and a subclass per kind of failure, so that a caller (or
 * the lab) can branch on what went wrong rather than on message text. Messages keep one style: the operation first,
 * then what was wrong, e.g. `matmul: shapes [2, 3] and [4, 5] do not align (3 ≠ 4)`.
 *
 * - `ShapeError`: shapes that do not fit the operation (rank, broadcasting, alignment, out-of-range axes).
 * - `DTypeError`: dtypes the operation does not accept.
 * - `NotDifferentiableError`: a derivative was requested through an operation that has no derivative rule.
 * - `NumericalError`: the computation cannot produce a meaningful result (singular, not finite, not converged, not
 *   positive definite, degenerate). `LinAlgError` in `aifn-compute/numerics/linalg` is a subclass, so existing catches keep working.
 * - `DomainError`: parameters outside their domain (a negative scale, probabilities that do not sum to one).
 *
 * Elementwise primitives never throw on a domain error of their inputs: they follow IEEE arithmetic (NaN, ±∞).
 */

/** The base class of every error aifn raises on purpose. */
export class AifnError extends Error {
  /** The operation that raised it, e.g. `matmul` or `cholesky`. */
  readonly op: string

  constructor(op: string, message: string) {
    super(message)
    this.name = 'AifnError'
    this.op = op
  }
}

/** Shapes that do not fit an operation. `shapes` lists the offending shapes (may be empty when unknown). */
export class ShapeError extends AifnError {
  readonly shapes: readonly (readonly number[])[]

  constructor(op: string, message: string, shapes: readonly (readonly number[])[] = []) {
    super(op, message)
    this.name = 'ShapeError'
    this.shapes = shapes.map((s) => [...s])
  }
}

/** Dtypes an operation does not accept. */
export class DTypeError extends AifnError {
  readonly dtypes: readonly string[]

  constructor(op: string, message: string, dtypes: readonly string[] = []) {
    super(op, message)
    this.name = 'DTypeError'
    this.dtypes = [...dtypes]
  }
}

/**
 * A derivative was requested through an operation without a derivative rule, on a path to the output. `argument` is
 * the input position (0-based) when only one argument lacks a rule.
 */
export class NotDifferentiableError extends AifnError {
  readonly argument: number | undefined

  constructor(op: string, message?: string, argument?: number) {
    super(
      op,
      message ??
        `autodiff: no derivative: ${op} has no derivative rule, and the output depends on its inputs through it`,
    )
    this.name = 'NotDifferentiableError'
    this.argument = argument
  }
}

/** What made a computation fail numerically. */
export type NumericalKind = 'singular' | 'not-finite' | 'not-converged' | 'not-positive-definite' | 'degenerate'

/** A computation that cannot produce a meaningful result. `kind` says why. */
export class NumericalError<K extends string = NumericalKind> extends AifnError {
  readonly kind: K

  constructor(op: string, message: string, kind: K) {
    super(op, message)
    this.name = 'NumericalError'
    this.kind = kind
  }
}

/** Parameters outside their domain (a sampler's negative scale, weights that are all zero). */
export class DomainError extends AifnError {
  constructor(op: string, message: string) {
    super(op, message)
    this.name = 'DomainError'
  }
}
