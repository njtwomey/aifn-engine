/**
 * Class sizes for labelled generators: a total with class proportions, or explicit per-class counts. Counts are exact
 * (never drawn at random), so a figure that asks for 20% positives gets exactly that share, up to rounding.
 */

import { checkCount } from './types'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A total number of points, or one count per class. */
export type ClassSizes = number | readonly number[]

/**
 * How many points of each class a labelled generator draws. Give either `n` as an array of per-class counts, or a
 * total `n` with at most one of `prevalence` (two classes: the share of class 1) and `classWeights` (one non-negative
 * weight per class, normalised). With a total alone the classes are equal, the first `n mod k` getting one extra.
 */
export interface ClassSizeOptions {
  /** Total points, or one count per class. */
  n?: ClassSizes
  /** Binary generators only: the share of class 1 (the positive class), in [0, 1]. */
  prevalence?: number
  /** One weight per class (normalised to sum to one). */
  classWeights?: readonly number[]
}

/**
 * Split `n` into counts proportional to `weights` by the largest-remainder (Hamilton) method: class $j$ gets
 * $\lfloor n w_j / \sum_i w_i \rfloor$, and the units left over go to the classes with the largest fractional parts,
 * ties to the lower index. The counts sum to `n` exactly and each is within one of its quota; equal weights reproduce
 * scikit-learn's split (the first $n \bmod k$ classes get one extra). Throws `DomainError` unless `n` is a
 * non-negative integer and the weights are non-negative with a positive sum.
 *
 * @param n The total number of points to split.
 * @param weights One non-negative weight per class (not necessarily normalised).
 * @returns One count per class, summing to `n`.
 *
 * @example Equal weights, and a tie broken to the lower index
 * print('10 over three equal classes:', classCounts(10, [1, 1, 1]))
 * print('10 at weights 1 : 3:', classCounts(10, [1, 3]))
 * print('5 at weights 1 : 1 : 2:', classCounts(5, [1, 1, 2]))
 */
export function classCounts(n: number, weights: readonly number[]): number[] {
  checkCount(n, 'classCounts')
  const total = weights.reduce((a, b) => a + b, 0)
  if (!(total > 0) || weights.some((w) => !(w >= 0)))
    throw new DomainError(
      'classCounts',
      `classCounts: weights must be non-negative with a positive sum, got [${weights.join(', ')}]`,
    )
  const quotas = weights.map((w) => (n * w) / total)
  const counts = quotas.map(Math.floor)
  let left = n - counts.reduce((a, b) => a + b, 0)
  const order = quotas.map((_, j) => j).sort((a, b) => quotas[b] - counts[b] - (quotas[a] - counts[a]) || a - b)
  for (let i = 0; left > 0; i = (i + 1) % order.length, left--) counts[order[i]]++
  return counts
}

/**
 * Normalised class proportions from a prevalence or weights (equal when neither is given). A prevalence $p$ gives
 * $(1 - p, p)$. Throws `DomainError` when both are given, when a prevalence is given for other than two classes or is
 * outside $[0, 1]$, or when the weights are not non-negative with a positive sum; throws `ShapeError` when there is
 * not one weight per class.
 *
 * @param options The `prevalence` or the `classWeights` to read (`n` is not read).
 * @param k The number of classes.
 * @param what The caller's name, for error messages.
 * @returns $k$ proportions summing to one.
 */
export function classWeightsOf(options: ClassSizeOptions, k: number, what: string): number[] {
  const { prevalence, classWeights } = options
  if (prevalence !== undefined && classWeights !== undefined)
    throw new DomainError(what, `${what}: give prevalence or classWeights, not both`)
  if (prevalence !== undefined) {
    if (k !== 2) throw new DomainError(what, `${what}: prevalence needs two classes (use classWeights for ${k})`)
    if (!(prevalence >= 0 && prevalence <= 1)) throw new DomainError(what, `${what}: prevalence must be in [0, 1]`)
    return [1 - prevalence, prevalence]
  }
  if (classWeights !== undefined) {
    if (classWeights.length !== k)
      throw new ShapeError(what, `${what}: ${classWeights.length} class weights for ${k} classes`)
    const total = classWeights.reduce((a, b) => a + b, 0)
    if (!(total > 0) || classWeights.some((w) => !(w >= 0)))
      throw new DomainError(what, `${what}: class weights must be non-negative with a positive sum`)
    return classWeights.map((w) => w / total)
  }
  return Array<number>(k).fill(1 / k)
}

/**
 * Resolve `ClassSizeOptions` into per-class counts and the population class proportions (`priors`) the counts
 * represent: the requested proportions, or the counts' own shares when explicit counts are given (equal shares when
 * they are all zero). Throws `DomainError` when counts and proportions are both given or a count is not a non-negative
 * integer, and `ShapeError` when there is not one count per class; see `classWeightsOf` for the proportions.
 *
 * @param options The total or the per-class counts, and the prevalence or class weights.
 * @param k The number of classes.
 * @param defaultN The total used when `options.n` is not given.
 * @param what The caller's name, for error messages.
 * @returns `sizes`, the count of each class; `priors`, the proportions they stand for; and `controlled`, true when
 *   the sizes were asked for (explicit counts, a prevalence or class weights), false for the default equal split.
 */
export function resolveClassSizes(
  options: ClassSizeOptions,
  k: number,
  defaultN: number,
  what: string,
): { sizes: number[]; priors: number[]; controlled: boolean } {
  const n = options.n ?? defaultN
  if (typeof n !== 'number') {
    if (options.prevalence !== undefined || options.classWeights !== undefined)
      throw new DomainError(what, `${what}: give per-class counts or proportions, not both`)
    if (n.length !== k) throw new ShapeError(what, `${what}: ${n.length} sizes for ${k} classes`)
    n.forEach((v) => checkCount(v, what))
    const total = n.reduce((a, b) => a + b, 0)
    return { sizes: [...n], priors: n.map((v) => (total > 0 ? v / total : 1 / k)), controlled: true }
  }
  checkCount(n, what)
  const priors = classWeightsOf(options, k, what)
  return {
    sizes: classCounts(n, priors),
    priors,
    controlled: options.prevalence !== undefined || options.classWeights !== undefined,
  }
}
