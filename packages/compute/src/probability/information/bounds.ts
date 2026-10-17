/**
 * Variational lower bounds on the mutual information $I(X; Y)$, estimated from one batch of $n$ pairs
 * $(x_i, y_i) \sim p(x, y)$ scored by a critic $T$, as compared by Poole, Ozair, van den Oord, Alemi and Tucker
 * (2019), "On variational bounds of mutual information", ICML.
 *
 * The scores are an $n \times n$ row-major matrix with $T_{ij} = T(x_i, y_j)$: the diagonal holds the $n$ positive
 * (joint) pairs, the off-diagonal entries the $n(n - 1)$ pairs that stand in for draws from the product of the
 * marginals $p(x)\,p(y)$.
 *
 * - InfoNCE (van den Oord, Li and Vinyals, 2018; Poole et al., 2019, eq. 12):
 *   $\frac1n \sum_i \big[T_{ii} - \log \frac1n \sum_j e^{T_{ij}}\big]$, which never exceeds $\log n$. It is
 *   $\log n$ minus the InfoNCE loss of the score rows (`infoNce` of `aifn-compute/learning/losses` computes that loss
 *   from embeddings and a temperature).
 * - NWJ (Nguyen, Wainwright and Jordan, 2010; Poole et al., 2019, eq. 9):
 *   $\frac1n \sum_i T_{ii} - e^{-1} \operatorname{mean}_{i \ne j} e^{T_{ij}}$.
 * - Donsker–Varadhan, the bound MINE maximises (Belghazi et al., 2018, eq. 4):
 *   $\frac1n \sum_i T_{ii} - \log \operatorname{mean}_{i \ne j} e^{T_{ij}}$.
 *
 * For a Gaussian pair the truth is known in closed form, and `correlationForMutualInformation` inverts it, so an
 * estimate can be compared with the value it bounds. Every function works on plain numbers; the log-sum-exps are
 * computed stably.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * Throws unless `scores` is an $n \times n$ matrix with $n \ge 2$.
 *
 * @param scores The critic values, row-major.
 * @param n The batch size.
 * @param where The caller's name, for error messages.
 */
function checkScores(scores: ArrayLike<number>, n: number, where: string): void {
  if (!(Number.isInteger(n) && n >= 2)) throw new DomainError(where, `${where}: n must be an integer of at least 2`)
  if (scores.length !== n * n)
    throw new ShapeError(where, `${where}: scores must hold n² = ${n * n} values, got ${scores.length}`)
}

/**
 * The InfoNCE estimate $\log n - \mathcal{L}_{\text{NCE}}$ from one batch: each row's positive (the diagonal) against
 * all $n$ candidates in its row, by a stable log-sum-exp. It is at most $\log n$ for any scores, so it saturates when
 * the true $I$ exceeds $\log n$. Throws `DomainError` for $n < 2$ and `ShapeError` when `scores` does not hold $n^2$
 * values.
 *
 * @param scores The critic values $T_{ij} = T(x_i, y_j)$, row-major $n \times n$, positives on the diagonal.
 * @param n The batch size $n \ge 2$.
 * @returns The estimate in nats.
 *
 * @example A critic that separates every pair reaches log n and no further
 * const n = 4
 * const separated = Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 50 : 0))
 * print('InfoNCE:', infoNceBound(separated, n), ' log n:', Math.log(n))
 * print('a worked 3 × 3 batch:', infoNceBound([2, 0.1, -0.3, 0.4, 1.5, 0, -0.2, 0.3, 2.5], 3))
 */
export function infoNceBound(scores: ArrayLike<number>, n: number): number {
  checkScores(scores, n, 'infoNceBound')
  let total = 0
  for (let i = 0; i < n; i++) {
    let top = -Infinity
    for (let j = 0; j < n; j++) top = Math.max(top, scores[i * n + j])
    let sum = 0
    for (let j = 0; j < n; j++) sum += Math.exp(scores[i * n + j] - top)
    total += scores[i * n + i] - (top + Math.log(sum / n))
  }
  return total / n
}

/**
 * The mean of the diagonal and the logarithm of the mean of $e^{T_{ij}}$ over the off-diagonal entries (stably).
 *
 * @param scores The critic values, row-major $n \times n$.
 * @param n The batch size.
 * @returns `diagonal`, $\frac1n \sum_i T_{ii}$, and `logMeanExp`, $\log \operatorname{mean}_{i \ne j} e^{T_{ij}}$.
 */
function joints(scores: ArrayLike<number>, n: number): { diagonal: number; logMeanExp: number } {
  let diagonal = 0
  let top = -Infinity
  for (let i = 0; i < n; i++) {
    diagonal += scores[i * n + i]
    for (let j = 0; j < n; j++) if (j !== i) top = Math.max(top, scores[i * n + j])
  }
  let sum = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (j !== i) sum += Math.exp(scores[i * n + j] - top)
  return { diagonal: diagonal / n, logMeanExp: top + Math.log(sum / (n * (n - 1))) }
}

/**
 * The NWJ estimate $\operatorname{mean}_i T_{ii} - e^{-1}\operatorname{mean}_{i \ne j} e^{T_{ij}}$. Unbiased for the
 * bound, and equal in expectation to $I$ at the critic $1 + \log p(y \mid x)/p(y)$, but its variance grows like
 * $e^{I}$. Throws as `infoNceBound` does.
 *
 * @param scores The critic values, row-major $n \times n$, positives on the diagonal.
 * @param n The batch size $n \ge 2$.
 * @returns The estimate in nats.
 *
 * @example The three bounds on one worked batch
 * const T = [2, 0.1, -0.3, 0.4, 1.5, 0, -0.2, 0.3, 2.5]
 * print('NWJ:', nwjBound(T, 3), ' DV:', donskerVaradhanBound(T, 3), ' InfoNCE:', infoNceBound(T, 3))
 */
export function nwjBound(scores: ArrayLike<number>, n: number): number {
  checkScores(scores, n, 'nwjBound')
  const { diagonal, logMeanExp } = joints(scores, n)
  return diagonal - Math.exp(logMeanExp - 1)
}

/**
 * The Donsker–Varadhan estimate $\operatorname{mean}_i T_{ii} - \log \operatorname{mean}_{i \ne j} e^{T_{ij}}$, the
 * objective of MINE. The logarithm of a sample mean makes it biased; MINE corrects the gradient with a moving average.
 * Throws as `infoNceBound` does.
 *
 * @param scores The critic values, row-major $n \times n$, positives on the diagonal.
 * @param n The batch size $n \ge 2$.
 * @returns The estimate in nats.
 *
 * @example Adding a constant to the critic leaves DV unchanged
 * const T = [2, 0.1, -0.3, 0.4, 1.5, 0, -0.2, 0.3, 2.5]
 * print('DV:', donskerVaradhanBound(T, 3), ' DV of T + 5:', donskerVaradhanBound(T.map((v) => v + 5), 3))
 */
export function donskerVaradhanBound(scores: ArrayLike<number>, n: number): number {
  checkScores(scores, n, 'donskerVaradhanBound')
  const { diagonal, logMeanExp } = joints(scores, n)
  return diagonal - logMeanExp
}

/**
 * The correlation $\rho \in [0, 1)$ at which a $d$-dimensional Gaussian pair with independent coordinate pairs of
 * correlation $\rho$ has mutual information $I = -\frac d2 \log(1 - \rho^2)$: $\rho = \sqrt{1 - e^{-2 I / d}}$. The
 * inverse of the closed form that `gaussianMutualInformation` evaluates. Throws `DomainError` for a negative $I$.
 *
 * @param mi The mutual information $I \ge 0$ in nats.
 * @param dims The dimension $d \ge 1$.
 * @returns $\rho$.
 *
 * @example The correlation for 2 nats, and back
 * const rho = correlationForMutualInformation(2)
 * print('rho:', rho, ' -½ log(1 - rho²):', -0.5 * Math.log(1 - rho * rho))
 */
export function correlationForMutualInformation(mi: number, dims = 1): number {
  if (!(mi >= 0))
    throw new DomainError('correlationForMutualInformation', 'correlationForMutualInformation: mi must be non-negative')
  return Math.sqrt(-Math.expm1((-2 * mi) / dims))
}
