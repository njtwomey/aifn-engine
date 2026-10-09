/**
 * The error hierarchy of aifn: one base class, `AifnError`, and a subclass per kind of failure, so that a caller (or
 * the lab) can branch on what went wrong rather than on message text. Messages keep one style: the operation first,
 * then what was wrong, e.g. `matmul: shapes [2, 3] and [4, 5] do not align (3 ≠ 4)`.
 *
 * - `ShapeError`: shapes that do not fit the operation (rank, broadcasting, alignment, out-of-range axes).
 * - `DTypeError`: dtypes the operation does not accept.
 * - `NotDifferentiableError`: a derivative was requested through an operation that has no derivative rule.
 * - `NumericalError`: the computation cannot produce a meaningful result (singular, not finite, not converged, not
 *   positive definite, degenerate). `LinAlgError` in `aifn-compute/numerics/linalg` is a subclass, so existing catches
 *   keep working.
 * - `DomainError`: parameters outside their domain (a negative scale, probabilities that do not sum to one).
 *
 * Elementwise primitives never throw on a domain error of their inputs: they follow IEEE arithmetic (NaN, $\pm\infty$).
 */

/**
 * The base class of every error aifn raises on purpose. A `catch` that tests `instanceof AifnError` separates the
 * failures aifn reports from bugs (a `TypeError` or `RangeError` of the runtime).
 *
 * @example Every deliberate error is an AifnError
 * try {
 *   matmul(tensor([[1, 2, 3]]), tensor([[1, 2]]))
 * } catch (e) {
 *   print(e.name, 'raised by', e.op)
 *   print('an AifnError:', e instanceof AifnError)
 * }
 */
export class AifnError extends Error {
  /** The operation that raised it, e.g. `matmul` or `cholesky`. */
  readonly op: string

  /**
   * @param op The name of the operation that raised the error, kept as `op`.
   * @param message The message, written as the operation's name, a colon, then what was wrong.
   */
  constructor(op: string, message: string) {
    super(message)
    this.name = 'AifnError'
    this.op = op
  }
}

/**
 * Shapes that do not fit an operation. `shapes` lists the offending shapes (may be empty when unknown).
 *
 * @example Shapes that do not broadcast
 * try {
 *   add(tensor([1, 2, 3]), tensor([1, 2]))
 * } catch (e) {
 *   print(e.name, ':', e.message)
 *   print('shapes =', e.shapes)
 * }
 */
export class ShapeError extends AifnError {
  /** The offending shapes, copied from the constructor's argument (empty when the raiser did not give them). */
  readonly shapes: readonly (readonly number[])[]

  /**
   * @param op The name of the operation that raised the error.
   * @param message The message, the operation's name first.
   * @param shapes The shapes that did not fit; each is copied.
   */
  constructor(op: string, message: string, shapes: readonly (readonly number[])[] = []) {
    super(op, message)
    this.name = 'ShapeError'
    this.shapes = shapes.map((s) => [...s])
  }
}

/**
 * Dtypes an operation does not accept.
 *
 * @example A complex value cannot become a real one
 * try {
 *   astype(complex(1, 2), 'float64')
 * } catch (e) {
 *   print(e.name, ':', e.message)
 *   print('dtypes =', e.dtypes)
 * }
 */
export class DTypeError extends AifnError {
  /** The offending dtypes, copied from the constructor's argument (empty when the raiser did not give them). */
  readonly dtypes: readonly string[]

  /**
   * @param op The name of the operation that raised the error.
   * @param message The message, the operation's name first.
   * @param dtypes The dtypes involved (`astype` gives the source and the target); copied.
   */
  constructor(op: string, message: string, dtypes: readonly string[] = []) {
    super(op, message)
    this.name = 'DTypeError'
    this.dtypes = [...dtypes]
  }
}

/**
 * A derivative was requested through an operation without a derivative rule, on a path to the output. `argument` is
 * the input position (0-based) when only one argument lacks a rule.
 *
 * @example Differentiating through an opaque scalar function
 * try {
 *   grad((x) => map(x, Math.cbrt))(8)
 * } catch (e) {
 *   print(e.name, 'in', e.op)
 *   print(e.message)
 * }
 */
export class NotDifferentiableError extends AifnError {
  /** The 0-based position of the argument without a rule, or undefined when the whole operation has none. */
  readonly argument: number | undefined

  /**
   * @param op The name of the operation without a derivative rule.
   * @param message The message; left out, a standard one naming `op` is used.
   * @param argument The 0-based position of the one argument that has no rule, if only one lacks it.
   */
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

/**
 * A computation that cannot produce a meaningful result. `kind` says why: one of `NumericalKind` by default, or a
 * module's own union through the type parameter.
 *
 * @example Raise and branch on the kind
 * try {
 *   throw new NumericalError('newton', 'newton: no convergence in 50 iterations', 'not-converged')
 * } catch (e) {
 *   print(e.name, 'of kind', e.kind)
 *   print('an AifnError:', e instanceof AifnError)
 * }
 */
export class NumericalError<K extends string = NumericalKind> extends AifnError {
  /** Why the computation failed, e.g. `'singular'` or `'not-converged'`. */
  readonly kind: K

  /**
   * @param op The name of the operation that raised the error.
   * @param message The message, the operation's name first.
   * @param kind Why it failed, kept as `kind` for a caller to branch on.
   */
  constructor(op: string, message: string, kind: K) {
    super(op, message)
    this.name = 'NumericalError'
    this.kind = kind
  }
}

/**
 * Parameters outside their domain (a sampler's negative scale, weights that are all zero).
 *
 * @example Weights that cannot be normalised
 * try {
 *   categorical(stream(0), [0, 0])
 * } catch (e) {
 *   print(e.name, ':', e.message)
 * }
 */
export class DomainError extends AifnError {
  /**
   * @param op The name of the operation that raised the error.
   * @param message The message, the operation's name first.
   */
  constructor(op: string, message: string) {
    super(op, message)
    this.name = 'DomainError'
  }
}
