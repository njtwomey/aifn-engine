/**
 * Covariance kernels (design S §2.10): positive-definite functions $k(\xvec, \xvec')$ as immutable objects whose
 * hyperparameters are a tree of values, so a Gram matrix is differentiable in them. `evaluate` has the `KernelFn`
 * signature. The word "kernel" means only this in aifn; smoothing kernels and filter taps have other names.
 */

import type { Kinded } from './kinds'
import type { Value } from './numbers'

/** A hyperparameter tree: named values (numbers, tensors, traced values) and nested trees of combined kernels. */
export type KernelParams = { readonly [name: string]: Value | KernelParams | readonly KernelParams[] }

/**
 * A covariance function (`kind: 'kernel'`). `params` holds every continuous hyperparameter (all positive); structural
 * choices (the Matérn $\nu$, a polynomial's degree) are fixed when the kernel is made.
 */
export interface Kernel<P extends KernelParams = KernelParams> extends Kinded<'kernel'> {
  /** A readable name, e.g. "rbf", "sum(rbf, white)". */
  readonly name: string
  /** The hyperparameters: named values, and the trees of combined kernels. */
  readonly params: P
  /** True when $k(\xvec, \xvec')$ depends only on $\xvec - \xvec'$. */
  readonly stationary: boolean
  /**
   * The $n \times m$ cross-covariance between the rows of `x` ($n \times d$) and `y` ($m \times d$). `y === null`
   * means the same inputs twice, which is where `white` noise contributes.
   */
  evaluate(x: Value, y: Value | null): Value
  /** $k(\xvec_i, \xvec_i)$ for each row $\xvec_i$ of `x`, shape `[n]`. */
  diagonal(x: Value): Value
  /** The same kernel with new hyperparameters (the same tree shape). */
  withParams(params: P): Kernel<P>
}
