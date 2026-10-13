/**
 * Max-value entropy search (Wang and Jegelka, 2017, "Max-value entropy search for efficient Bayesian optimization",
 * ICML): the information a new evaluation gives about the optimal value $y^*$, rather than about the optimiser's
 * location as in entropy search (Hennig and Schuler, 2012).
 *
 * For maximisation, with $K$ samples $y^*_k$ of the maximum and $\gamma_k(\xvec) = (y^*_k - \mu(\xvec)) /
 * \sigma(\xvec)$, the acquisition (their eq. 6) is
 *
 * $$ \alpha(\xvec) = \frac{1}{K} \sum_{k=1}^{K} \left[ \frac{\gamma_k(\xvec)\,\phi(\gamma_k(\xvec))}
 * {2\,\Phi(\gamma_k(\xvec))} - \log \Phi(\gamma_k(\xvec)) \right], $$
 *
 * the expected drop in the entropy of $y(\xvec)$ once $y^*$ is known, which is never negative. The samples come from
 * their Gumbel approximation (§3.1): treating the posterior values on the candidates as independent,
 * $\Pr[y^* < y] \approx \prod_i \Phi((y - \mu_i) / \sigma_i)$; a Gumbel law
 * $\exp(-e^{-(y - a)/b})$ is matched to this cdf at its quartiles and median, and sampled by inversion,
 * $y^* = a - b \log(-\log u)$. Minimisation (the default, as in the rest of the module) negates the objective: $y^*$
 * is then a sample of the minimum and the formula is applied to $-\mu$ and $-y^*$. Where $\sigma = 0$ the acquisition
 * is 0.
 */

import { normalLogCdf, normalLogPdf } from 'aifn-compute/numerics/special'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { mapMoments, moments } from './acquisitions'

/** Samples of the optimal value by the Gumbel approximation, and the Gumbel law they were drawn from. */
export type MaxValueSamples = {
  /** The samples $y^*_k$, `[K]`, in the objective's units: of the minimum when minimising, else of the maximum. */
  samples: Tensor
  /**
   * The Gumbel location $a$ of the law of the maximum of the (signed) objective, $-y^*$ when minimising. Its median
   * is $a - b \log\log 2$.
   */
  location: number
  /** The Gumbel scale $b \ge 0$ (0 when every $\sigma_i$ is 0, and the samples all equal the best mean). */
  scale: number
}

/** Options of `maxValueSamples`. */
export type MaxValueSamplesOptions = {
  /** The number $K$ of samples (default 10). */
  count?: number
  /** The stream the uniforms are drawn from (required when `count` is positive). */
  stream?: Stream
  /**
   * The incumbent: the best value observed. Samples beyond it are kept, those short of it are moved to it, since the
   * optimum cannot be worse than a value already seen (as Wang and Jegelka do). Left out, the samples are not
   * clipped.
   */
  best?: number
  /** Minimise the objective (default true): the samples are then of its minimum. */
  minimise?: boolean
}

/**
 * $\log(-\log q)$: the Gumbel quantile at $q$ is $a - b\log(-\log q)$.
 *
 * @param q A probability in $(0, 1)$.
 * @returns $\log(-\log q)$.
 */
function logNegLog(q: number): number {
  return Math.log(-Math.log(q))
}

/**
 * Samples of the optimal value $y^*$ of the objective under a GP posterior with means $\mu_i$ and standard deviations
 * $\sigma_i$ on a set of candidates, by the Gumbel approximation of Wang and Jegelka (2017, §3.1). The cdf
 * $F(y) = \prod_i \Phi((y - \mu_i) / \sigma_i)$ of the maximum (of $-f$ when minimising) is inverted by bisection at
 * $q = \tfrac14, \tfrac12, \tfrac34$, giving the Gumbel scale $b = (y_{3/4} - y_{1/4}) / (\log(-\log\tfrac14) -
 * \log(-\log\tfrac34))$ and location $a = y_{1/2} + b \log(-\log\tfrac12)$, and each sample is $a - b\log(-\log u)$
 * for a uniform $u$ from the stream. Throws `DomainError` for a negative standard deviation, for a positive `count`
 * without a stream, and when there are no candidates.
 *
 * @param mu The posterior means at the candidates (a tensor of any shape, read flat).
 * @param sd The posterior standard deviations, as many as `mu` (or one number for all).
 * @param options The count, the stream, the incumbent and the direction, as `MaxValueSamplesOptions`.
 * @param options.count The number $K$ of samples; 0 fits the Gumbel law without drawing.
 * @param options.stream The stream the uniforms are drawn from (required when `count` is positive).
 * @param options.best The incumbent; samples short of it are moved to it. Left out, nothing is clipped.
 * @param options.minimise Whether the objective is minimised, so that the samples are of its minimum.
 * @returns The samples, and the Gumbel location and scale (of the maximum of the signed objective).
 *
 * @example Two candidates with the same spread: the minimum is likely below both means
 * const mu = tensor([0, 0.5])
 * const { samples, location, scale } = maxValueSamples(mu, tensor([1, 1]), { count: 5, stream: stream(0) })
 * print('samples of min f:', samples)
 * print('Gumbel location and scale (of max -f):', location, scale)
 */
export function maxValueSamples(
  mu: number | Tensor,
  sd: number | Tensor,
  { count = 10, stream, best, minimise = true }: MaxValueSamplesOptions = {},
): MaxValueSamples {
  // Signed values: the law of the maximum of g = f (maximise) or g = -f (minimise).
  const { m, s } = moments(mu, sd, 'maxValueSamples')
  if (minimise) for (let i = 0; i < m.length; i++) m[i] = -m[i]
  const n = m.length
  if (n === 0) throw new DomainError('maxValueSamples', 'maxValueSamples: needs at least one candidate')
  const logF = (y: number): number => {
    let total = 0
    for (let i = 0; i < n; i++) {
      if (s[i] === 0) {
        if (y < m[i]) return -Infinity
      } else total += normalLogCdf((y - m[i]) / s[i])
    }
    return total
  }
  let lo = -Infinity
  let hi = -Infinity
  for (let i = 0; i < n; i++) {
    lo = Math.max(lo, m[i] - 5 * s[i])
    hi = Math.max(hi, m[i] + 5 * s[i])
  }
  let location = hi
  let scale = 0
  if (hi > lo) {
    const quantile = (q: number): number => {
      const target = Math.log(q)
      let a = lo
      let b = hi
      // Widen the bracket until it holds the quantile (F(lo) <= Phi(-5) already; F(hi) may be short of q).
      while (logF(b) < target) b += b - a
      while (logF(a) > target) a -= b - a
      for (let it = 0; it < 200 && b - a > 4 * Number.EPSILON * Math.max(1, Math.abs(a), Math.abs(b)); it++) {
        const c = 0.5 * (a + b)
        if (logF(c) < target) a = c
        else b = c
      }
      return 0.5 * (a + b)
    }
    const [y1, y2, y3] = [quantile(0.25), quantile(0.5), quantile(0.75)]
    scale = (y3 - y1) / (logNegLog(0.25) - logNegLog(0.75))
    location = y2 + scale * logNegLog(0.5)
  }
  const out = new Float64Array(count)
  if (count > 0) {
    if (!stream) throw new DomainError('maxValueSamples', 'maxValueSamples: needs a stream to draw from')
    const u = toFlat(uniform(stream, 0, 1, { shape: [count] }) as Tensor)
    const signedBest = best === undefined ? -Infinity : minimise ? -best : best
    for (let k = 0; k < count; k++) {
      const uk = Math.min(Math.max(u[k], 1e-300), 1 - 1e-16)
      const g = Math.max(location - scale * Math.log(-Math.log(uk)), signedBest)
      out[k] = minimise ? -g : g
    }
  }
  return { samples: fromData(out, [count]), location, scale }
}

/** Options of `maxValueEntropySearch`. */
export type MaxValueEntropyOptions = {
  /**
   * The samples of the optimal value: a count $K$ (default 10), drawn by `maxValueSamples` from `stream` on the same
   * candidates, or the samples themselves (a tensor or an array), for instance the optima of posterior draws.
   */
  samples?: number | Tensor | readonly number[]
  /** The stream the samples are drawn from, when `samples` is a count. */
  stream?: Stream
  /** The incumbent, passed to `maxValueSamples` (samples short of it are moved to it); ignored for given samples. */
  best?: number
  /** Minimise the objective (default true): the samples are then of its minimum. */
  minimise?: boolean
}

/**
 * Max-value entropy search (Wang and Jegelka, 2017, eq. 6): the mutual information between a new evaluation and the
 * optimal value $y^*$, averaged over samples of $y^*$, at each candidate. Larger is better, and it is 0 where
 * $\sigma = 0$. With a count of samples they are drawn by the Gumbel approximation (`maxValueSamples`) from the same
 * means and standard deviations, so `mu` and `sd` should cover the whole candidate set; given samples are used as they
 * are. Throws `DomainError` for a negative standard deviation, for an empty set of samples and for a count without a
 * stream.
 *
 * @param mu The posterior means $\mu(\xvec)$ at the candidates: a number or a tensor of any shape.
 * @param sd The posterior standard deviations $\sigma(\xvec) \ge 0$, as many as `mu` (or one number for all).
 * @param options The samples (a count or the values), the stream, the incumbent and the direction, as
 *   `MaxValueEntropyOptions`.
 * @returns The acquisition, with the kind and shape of `mu`.
 *
 * @example With one sample of the minimum, the score grows with the chance of reaching it
 * const mu = tensor([0, 1, 2])
 * print('MES:', maxValueEntropySearch(mu, tensor([1, 1, 1]), { samples: [-1] }))
 *
 * @example Gumbel samples from a posterior on a grid
 * const x = linspace(0, 1, 5)
 * const mu = sin(mul(6, x))
 * const sd = tensor([0.1, 0.4, 0.4, 0.4, 0.1])
 * print('MES:', maxValueEntropySearch(mu, sd, { samples: 20, stream: stream(1) }))
 */
export function maxValueEntropySearch(mu: number, sd: number, options?: MaxValueEntropyOptions): number
export function maxValueEntropySearch(mu: Tensor, sd: number | Tensor, options?: MaxValueEntropyOptions): Tensor
export function maxValueEntropySearch(
  mu: number | Tensor,
  sd: number | Tensor,
  options?: MaxValueEntropyOptions,
): number | Tensor
export function maxValueEntropySearch(
  mu: number | Tensor,
  sd: number | Tensor,
  { samples = 10, stream, best, minimise = true }: MaxValueEntropyOptions = {},
): number | Tensor {
  const ys =
    typeof samples === 'number'
      ? toFlat(maxValueSamples(mu, sd, { count: samples, stream, best, minimise }).samples)
      : isTensor(samples)
        ? toFlat(samples)
        : Array.from(samples)
  if (ys.length === 0) throw new DomainError('maxValueEntropySearch', 'maxValueEntropySearch: needs samples of y*')
  const signed = ys.map((y) => (minimise ? -y : y))
  return mapMoments(mu, sd, 'maxValueEntropySearch', (v, w) => {
    if (w === 0) return 0
    const m = minimise ? -v : v
    let total = 0
    for (const y of signed) {
      const g = (y - m) / w
      const logCdf = normalLogCdf(g)
      total += (g * Math.exp(normalLogPdf(g) - logCdf)) / 2 - logCdf
    }
    return Math.max(total / signed.length, 0)
  })
}
