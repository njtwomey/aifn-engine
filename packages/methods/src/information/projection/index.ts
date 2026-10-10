/**
 * `aifn-methods/information/projection`: KL projection of a univariate distribution onto the normal family, by L-BFGS.
 *
 * - `normalProjection`: the normal $q$ closest to $p$ in either direction of KL. `forward`,
 *   $\argmin_q \KL(p \,\Vert\, q)$, is the moment projection: it matches the mean and variance of $p$ and covers all
 *   its mass. `reverse`, $\argmin_q \KL(q \,\Vert\, p)$, is the information projection of variational inference: it
 *   seeks a mode, and which one depends on the start.
 * - The registry: `projectionFunctions`.
 *
 * Expectations are by quadrature (Gauss–Legendre over $p$'s quantile range, Gauss–Hermite under $q$) and gradients
 * by autodiff; every fit keeps its optimiser path. The distribution must be unbatched, continuous and univariate, or
 * `DomainError` is thrown.
 */

export {
  normalProjection,
  type KlDirection,
  type NormalFitStep,
  type NormalProjection,
  type NormalProjectionOptions,
} from './projection'
export { projectionFunctions } from './registry'
