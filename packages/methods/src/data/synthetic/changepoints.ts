/**
 * Piecewise series with known changepoints, for changepoint detection: shifts in the mean, in the variance, in a
 * Poisson rate, and switches between autoregressive regimes. Each is a `Dataset` with the values in `x` [n, 1], the
 * segment index of every step in `y` (int32), the time index in `t`, and the segments in `meta.truth` (a
 * `ChangepointTruth`).
 *
 * Segment boundaries are given (`changepoints`), or drawn with geometric gaps of mean $g$ = `meanGap` (a constant
 * hazard $1/g$, the memoryless prior of Adams and MacKay, 2007), each raised to `minGap` when shorter. Segment
 * parameters are given, or drawn independently from a prior, as the model of BOCPD assumes; a drawn parameter is
 * redrawn (up to 100 times) until it differs from its neighbour's by at least `minJump` or `minRatio`, so that every
 * changepoint is visible. Every draw has its own substream: `child(s, 'gaps')`, `child(s, 'params')`,
 * `child(s, 'values')`.
 */

import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { Gamma, Geometric, Poisson } from 'aifn-compute/probability/distributions'
import { changepointTruth, type ChangepointFamily, type Segment } from '../truth'
import { checkCount, generatorRecipe, labels, matrix, type Dataset } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Where the segments start: given changepoints, or geometric gaps. */
export interface SegmentOptions {
  /** Length of the series. Default 300. */
  n?: number
  /** Indices where new segments begin: integers $c$ with $0 < c < n$, ascending. Drawn when omitted. */
  changepoints?: readonly number[]
  /** Mean gap $g$ between drawn changepoints (a hazard of $1/g$); at least 1. Default 60. */
  meanGap?: number
  /** Shortest drawn gap: a shorter geometric draw is raised to it. Default 5. */
  minGap?: number
}

/**
 * Changepoints with geometric gaps of mean `meanGap`, each at least `minGap`, from 0 until one reaches $n$. Throws
 * `DomainError` when `meanGap` is below 1.
 *
 * @param s The stream the gaps are drawn from (one child per gap).
 * @param n The length of the series: the changepoints are below it.
 * @param meanGap The mean $g$ of the geometric gaps, on $1, 2, \dots$
 * @param minGap The least gap; a shorter draw is raised to it.
 * @returns The changepoints, ascending (0 excluded).
 */
function drawnBoundaries(s: Stream, n: number, meanGap: number, minGap: number): number[] {
  if (!(meanGap >= 1)) throw new DomainError('changepoints', `changepoints: meanGap must be at least 1, got ${meanGap}`)
  const gap = Geometric(1 / meanGap)
  const out: number[] = []
  let at = 0
  for (let i = 0; ; i++) {
    at += Math.max(minGap, gap.sample(child(s, 'gap', i)) as number)
    if (at >= n) return out
    out.push(at)
  }
}

/**
 * The segment starts of a series: the given changepoints, checked, or drawn ones. Throws `DomainError` when $n$ is not
 * a non-negative integer or a changepoint is not an integer in $(0, n)$ above the one before.
 *
 * @param s The generator's stream; drawn changepoints come from `child(s, 'gaps')`.
 * @param options The length and the changepoints, or how to draw them.
 * @returns `n`, the length, and `starts`, the first index of every segment (0, then the changepoints).
 */
function boundaries(s: Stream, options: SegmentOptions): { n: number; starts: number[] } {
  const { n = 300, meanGap = 60, minGap = 5 } = options
  checkCount(n, 'changepoints')
  const cps = options.changepoints ? [...options.changepoints] : drawnBoundaries(child(s, 'gaps'), n, meanGap, minGap)
  cps.forEach((c, i) => {
    if (!Number.isInteger(c) || c <= 0 || c >= n || (i > 0 && c <= cps[i - 1]))
      throw new DomainError('changepoints', `changepoints: ${c} is not an ascending index in (0, ${n})`)
  })
  return { n, starts: [0, ...cps] }
}

/**
 * The dataset of a piecewise series: the values in `x` ($n \times 1$), the segment index of every step in `y`, the time
 * index in `t`, the segments in `meta.truth` and the call in `meta.recipe`.
 *
 * @param s The generator's stream, whose key is recorded.
 * @param base The generator's name, used as the dataset's name and the recipe's base.
 * @param family What changes between segments, for the truth.
 * @param n The length of the series.
 * @param segments The segments, contiguous from 0 to $n$.
 * @param values The $n$ values of the series; kept as `x`, not copied.
 * @param description The dataset's one-line description.
 * @param knobs The generator's options, recorded in the recipe without undefined values and functions.
 * @returns The dataset.
 */
function build(
  s: Stream,
  base: string,
  family: ChangepointFamily,
  n: number,
  segments: Segment[],
  values: Float64Array,
  description: string,
  knobs: Record<string, unknown>,
): Dataset {
  const y = new Int32Array(n)
  segments.forEach((g, j) => y.fill(j, g.start, g.end))
  const t: Tensor = fromData(Float64Array.from({ length: n }, (_, i) => i))
  return {
    kind: 'dataset',
    x: matrix(values, n, 1),
    y: labels(y),
    t,
    meta: {
      name: base,
      description,
      task: 'sequence',
      featureNames: ['value'],
      labelNames: segments.map((_, j) => `segment ${j}`),
      source: 'Piecewise model of Adams and MacKay (2007), "Bayesian Online Changepoint Detection", arXiv:0710.3742',
      key: s.key,
      truth: changepointTruth(base, family, segments),
      recipe: generatorRecipe(
        base,
        s.key,
        Object.fromEntries(Object.entries(knobs).filter(([, v]) => v !== undefined && typeof v !== 'function')),
      ),
    },
  }
}

/**
 * A segment's parameter: the given one, or a draw when none are given. Throws `ShapeError` when values are given but
 * fewer than the segments.
 *
 * @param given The values given by the caller, one per segment (extras are ignored), or undefined to draw.
 * @param j The segment's index.
 * @param draw Draws a value; called only when nothing is given.
 * @param what The caller's name for error messages.
 * @returns The segment's value.
 */
function pick<T>(given: readonly T[] | undefined, j: number, draw: () => T, what: string): T {
  if (!given) return draw()
  if (j >= given.length) throw new ShapeError(what, `${what}: ${given.length} values given for more segments`)
  return given[j]
}

/**
 * Draw until the value differs enough from the previous segment's (at most 100 tries, then the last draw), so that
 * drawn neighbours are distinguishable.
 *
 * @param draw Draws a candidate value.
 * @param previous The previous segment's value, or undefined for the first segment (whose first draw is kept).
 * @param farEnough Whether a candidate (first argument) is far enough from the previous value (second).
 * @returns The first draw far enough from `previous`, or the 101st draw.
 */
function apart(draw: () => number, previous: number | undefined, farEnough: (a: number, b: number) => boolean): number {
  let v = draw()
  for (let i = 0; i < 100 && previous !== undefined && !farEnough(v, previous); i++) v = draw()
  return v
}

/**
 * A count with its noun, in the plural unless it is 1: "1 segment", "3 segments".
 *
 * @param k The count.
 * @param word The noun in the singular.
 * @returns The phrase.
 */
const plural = (k: number, word: string) => `${k} ${word}${k === 1 ? '' : 's'}`

// ── Mean shifts ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `meanShifts`. */
export interface MeanShiftOptions extends SegmentOptions {
  /** Segment means, one per segment. Drawn from $\Gauss(0, \text{jump}^2)$ when omitted. */
  means?: readonly number[]
  /** Spread of drawn segment means. Default 3. */
  jump?: number
  /**
   * Least difference between drawn neighbouring means. Default 1 (one noise standard deviation at the default `sd`).
   */
  minJump?: number
  /** Noise standard deviation within every segment. Default 1. */
  sd?: number
}

/**
 * Gaussian noise around a mean that jumps at each changepoint (the setting of the well-log data): segment $j$ holds
 * $x_t \sim \Gauss(\mu_j, \sigma^2)$. Throws `ShapeError` when fewer `means` than segments are given, and
 * `DomainError` for bad changepoints.
 *
 * @param s The stream the changepoints, the means and the values are drawn from.
 * @param options The length and the changepoints (`SegmentOptions`), the means or their prior, and the noise.
 * @returns The values in `x` ($n \times 1$), the segment of every step in `y`, the time in `t`, and the segments (with
 *   `params: { mean, sd }`) in `meta.truth`.
 *
 * @example Given changepoints and means
 * const d = meanShifts(stream(1), { n: 200, changepoints: [60, 140], means: [0, 4, -2] })
 * const x = toArray(d.x).map(([v]) => v)
 * const y = toArray(d.y)
 * const avg = (j) => x.filter((_, i) => y[i] === j).reduce((a, v) => a + v, 0) / y.filter((g) => g === j).length
 * print('x:', d.x.shape, ' first values:', x.slice(0, 3))
 * print('changepoints:', d.meta.truth.changepoints)
 * print('segment means (0, 4, -2):', [0, 1, 2].map(avg))
 *
 * @example Drawn changepoints and means
 * const d = meanShifts(stream(2), { n: 300, meanGap: 60 })
 * print('changepoints:', d.meta.truth.changepoints)
 * print('drawn means:', d.meta.truth.segments.map((g) => g.mean))
 */
export function meanShifts(s: Stream, options: MeanShiftOptions = {}): Dataset {
  const { jump = 3, sd = 1, minJump = 1 } = options
  const { n, starts } = boundaries(s, options)
  const ps = child(s, 'params')
  const vs = child(s, 'values')
  const values = new Float64Array(n)
  let previous: number | undefined
  const segments = starts.map((start, j): Segment => {
    const end = starts[j + 1] ?? n
    const draw = () =>
      apart(
        () => jump * normal(ps),
        previous,
        (a, b) => Math.abs(a - b) >= minJump,
      )
    const mean = pick(options.means, j, draw, 'meanShifts means')
    previous = mean
    for (let i = start; i < end; i++) values[i] = mean + sd * normal(vs)
    return { start, end, params: { mean, sd }, mean, variance: sd * sd, risk: sd * sd }
  })
  return build(
    s,
    'meanShifts',
    'mean',
    n,
    segments,
    values,
    `${n} values in ${plural(segments.length, 'segment')}; the mean changes at each changepoint, the noise sd is ${sd}.`,
    { ...options },
  )
}

// ── Variance shifts ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `varianceShifts`. */
export interface VarianceShiftOptions extends SegmentOptions {
  /** Segment standard deviations. Drawn log-uniformly between the ends of `sdRange` when omitted. */
  sds?: readonly number[]
  /** Range of drawn standard deviations, low then high. Default [0.3, 3]. */
  sdRange?: readonly [number, number]
  /** Least ratio (larger over smaller) between drawn neighbouring standard deviations. Default 1.5. */
  minRatio?: number
  /** The mean, shared by every segment. Default 0. */
  mean?: number
}

/**
 * Noise of a fixed mean whose standard deviation changes at each changepoint (the setting of the Dow Jones returns):
 * segment $j$ holds $x_t \sim \Gauss(\mu, \sigma_j^2)$. Throws `ShapeError` when fewer `sds` than segments are given,
 * and `DomainError` for bad changepoints.
 *
 * @param s The stream the changepoints, the standard deviations and the values are drawn from.
 * @param options The length and the changepoints (`SegmentOptions`), the standard deviations or their range, and the
 *   mean.
 * @returns The values in `x` ($n \times 1$), the segment of every step in `y`, the time in `t`, and the segments (with
 *   `params: { mean, sd }`) in `meta.truth`.
 *
 * @example The spread changes, the mean does not
 * const d = varianceShifts(stream(1), { n: 300, changepoints: [150], sds: [0.5, 2] })
 * const x = toArray(d.x).map(([v]) => v)
 * const sd = (v) => Math.sqrt(v.reduce((a, u) => a + u * u, 0) / v.length)
 * print('x:', d.x.shape, ' changepoints:', d.meta.truth.changepoints)
 * print('sample sd before and after (0.5, 2):', sd(x.slice(0, 150)), sd(x.slice(150)))
 */
export function varianceShifts(s: Stream, options: VarianceShiftOptions = {}): Dataset {
  const { sdRange = [0.3, 3], mean = 0, minRatio = 1.5 } = options
  const { n, starts } = boundaries(s, options)
  const ps = child(s, 'params')
  const vs = child(s, 'values')
  const values = new Float64Array(n)
  const [lo, hi] = [Math.log(sdRange[0]), Math.log(sdRange[1])]
  let previous: number | undefined
  const segments = starts.map((start, j): Segment => {
    const end = starts[j + 1] ?? n
    const draw = () =>
      apart(
        () => Math.exp(uniform(ps, lo, hi)),
        previous,
        (a, b) => Math.max(a / b, b / a) >= minRatio,
      )
    const sd = pick(options.sds, j, draw, 'varianceShifts sds')
    previous = sd
    for (let i = start; i < end; i++) values[i] = mean + sd * normal(vs)
    return { start, end, params: { mean, sd }, mean, variance: sd * sd, risk: sd * sd }
  })
  return build(
    s,
    'varianceShifts',
    'variance',
    n,
    segments,
    values,
    `${n} values with mean ${mean} in ${plural(segments.length, 'segment')}; the noise sd changes at each changepoint.`,
    { ...options },
  )
}

// ── Poisson rate shifts ──────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `poissonShifts`. */
export interface PoissonShiftOptions extends SegmentOptions {
  /** Segment rates. Drawn from $\GammaD(\text{shape}, \text{rate})$ when omitted. */
  rates?: readonly number[]
  /** Shape of the Gamma prior of drawn rates, whose mean is shape over rate. Default 2. */
  shape?: number
  /** Rate (inverse scale) of the Gamma prior of drawn rates. Default 0.5. */
  rate?: number
  /** Least ratio (larger over smaller) between drawn neighbouring rates. Default 1.5. */
  minRatio?: number
}

/**
 * Counts per step from a Poisson rate that changes at each changepoint (the setting of the coal-mining disasters):
 * segment $j$ holds $x_t \sim \Poisson(\lambda_j)$. Throws `ShapeError` when fewer `rates` than segments are given,
 * and `DomainError` for bad changepoints.
 *
 * @param s The stream the changepoints, the rates and the counts are drawn from.
 * @param options The length and the changepoints (`SegmentOptions`), the rates or their Gamma prior.
 * @returns The counts in `x` ($n \times 1$), the segment of every step in `y`, the time in `t`, and the segments (with
 *   `params: { rate }`) in `meta.truth`.
 *
 * @example The mean count of each segment is near its rate
 * const d = poissonShifts(stream(1), { n: 300, changepoints: [100, 200], rates: [1, 6, 2] })
 * const x = toArray(d.x).map(([v]) => v)
 * const avg = (v) => v.reduce((a, u) => a + u, 0) / v.length
 * print('x:', d.x.shape, ' first counts:', x.slice(0, 8))
 * print('mean counts (1, 6, 2):', [x.slice(0, 100), x.slice(100, 200), x.slice(200)].map(avg))
 */
export function poissonShifts(s: Stream, options: PoissonShiftOptions = {}): Dataset {
  const { shape = 2, rate = 0.5, minRatio = 1.5 } = options
  const { n, starts } = boundaries(s, options)
  const prior = Gamma(shape, rate)
  const ps = child(s, 'params')
  const vs = child(s, 'values')
  const values = new Float64Array(n)
  let previous: number | undefined
  const segments = starts.map((start, j): Segment => {
    const end = starts[j + 1] ?? n
    let tries = 0
    const draw = () =>
      apart(
        () => prior.sample(child(ps, j, tries++)) as number,
        previous,
        (a, b) => Math.max(a / b, b / a) >= minRatio,
      )
    const lambda = pick(options.rates, j, draw, 'poissonShifts rates')
    previous = lambda
    const law = Poisson(lambda)
    for (let i = start; i < end; i++) values[i] = law.sample(child(vs, i)) as number
    return { start, end, params: { rate: lambda }, mean: lambda, variance: lambda, risk: lambda }
  })
  return build(
    s,
    'poissonShifts',
    'poisson',
    n,
    segments,
    values,
    `${n} Poisson counts in ${plural(segments.length, 'segment')}; the rate changes at each changepoint.`,
    { ...options },
  )
}

// ── Autoregressive regimes ───────────────────────────────────────────────────────────────────────────────────────────

/** Options for `arRegimes`. */
export interface ArRegimeOptions extends SegmentOptions {
  /**
   * The regimes' AR coefficients $a_1, \dots, a_p$ (all of one order $p$). Segments cycle through them in order, so
   * neighbours differ when there are two or more. Default [[0.9], [-0.7]]: a slowly wandering regime and a rapidly
   * alternating one.
   */
  regimes?: readonly (readonly number[])[]
  /** Standard deviation of the innovations. Default 1. */
  sd?: number
}

/**
 * The stationary variance of an AR($p$) process with innovation standard deviation $\sigma$:
 * $\sigma^2 \sum_j \psi_j^2$, with the MA($\infty$) weights $\psi_0 = 1$, $\psi_j = \sum_{k=1}^{p} a_k \psi_{j-k}$
 * ($\psi_j = 0$ for $j < 0$). The sum stops once $p + 1$ weights in a row have $\psi_j^2 < 10^{-20}$, and is infinite
 * once it passes $10^{12}$ (an explosive regime). A unit-root regime, whose weights neither grow nor die out, gets the
 * sum of its first $10^5$ terms instead.
 *
 * @param a The coefficients $a_1, \dots, a_p$.
 * @param sd The innovations' standard deviation $\sigma$.
 * @returns The stationary variance, or infinity.
 */
function stationaryVariance(a: readonly number[], sd: number): number {
  const psi = [1]
  let total = 1
  for (let j = 1; j < 100000; j++) {
    let v = 0
    for (let k = 0; k < a.length; k++) if (j - k - 1 >= 0) v += a[k] * psi[j - k - 1]
    psi.push(v)
    total += v * v
    if (!Number.isFinite(total) || total > 1e12) return Infinity
    if (j > a.length && psi.slice(-a.length - 1).every((u) => u * u < 1e-20)) break
  }
  return sd * sd * total
}

/**
 * An autoregression $x_t = \sum_{k=1}^{p} a_k x_{t-k} + e_t$, $e_t \sim \Gauss(0, \sigma^2)$, whose coefficients
 * switch between regimes at each changepoint. The recursion runs straight across boundaries (the lags carry over),
 * from zeros with a burn-in of 200 steps in the first regime. The truth's per-segment mean (0) and variance are the
 * regime's stationary ones. Throws `DomainError` when no regime is given or the regimes' orders differ.
 *
 * @param s The stream the changepoints (`child(s, 'gaps')`) and the innovations (`child(s, 'values')`) are drawn from.
 * @param options The length and the changepoints (`SegmentOptions`), the regimes and the innovations' standard
 *   deviation.
 * @returns The values in `x` ($n \times 1$), the segment of every step in `y`, the time in `t`, and the segments (with
 *   `params: { coefficients, sd, regime }`) in `meta.truth`.
 *
 * @example The lag-1 autocorrelation follows the regime's coefficient
 * const d = arRegimes(stream(1), { n: 400, changepoints: [200] })
 * const x = toArray(d.x).map(([v]) => v)
 * const r1 = (v) => v.slice(1).reduce((a, u, i) => a + u * v[i], 0) / v.reduce((a, u) => a + u * u, 0)
 * print('x:', d.x.shape, ' changepoints:', d.meta.truth.changepoints)
 * print('lag-1 autocorrelation (0.9, -0.7):', r1(x.slice(0, 200)), r1(x.slice(200)))
 * print('stationary variances:', d.meta.truth.segments.map((g) => g.variance))
 */
export function arRegimes(s: Stream, options: ArRegimeOptions = {}): Dataset {
  const { regimes = [[0.9], [-0.7]], sd = 1 } = options
  if (regimes.length === 0) throw new DomainError('arRegimes', 'arRegimes: give at least one regime')
  const p = regimes[0].length
  if (regimes.some((a) => a.length !== p))
    throw new DomainError('arRegimes', 'arRegimes: every regime must have the same order')
  const { n, starts } = boundaries(s, options)
  const vs = child(s, 'values')
  const burn = 200
  const x = new Float64Array(n + burn)
  const regimeAt = new Int32Array(n + burn)
  starts.forEach((start, j) => regimeAt.fill(j % regimes.length, start + burn, (starts[j + 1] ?? n) + burn))
  for (let t = 0; t < n + burn; t++) {
    const a = regimes[regimeAt[t]]
    let v = sd * normal(vs)
    for (let k = 0; k < p; k++) if (t - k - 1 >= 0) v += a[k] * x[t - k - 1]
    x[t] = v
  }
  const variances = regimes.map((a) => stationaryVariance(a, sd))
  const segments = starts.map((start, j): Segment => {
    const r = j % regimes.length
    return {
      start,
      end: starts[j + 1] ?? n,
      params: { coefficients: regimes[r], sd, regime: r },
      mean: 0,
      variance: variances[r],
      risk: sd * sd,
    }
  })
  return build(
    s,
    'arRegimes',
    'autoregressive',
    n,
    segments,
    x.slice(burn),
    `${n} steps of an AR(${p}) process that switches between ${plural(regimes.length, 'regime')} at each changepoint.`,
    { ...options },
  )
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'meanShifts',
    name: 'Mean shifts',
    summary: 'A piecewise-constant mean in Gaussian noise, with changepoints drawn at a constant hazard.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      meanGap: real(5, 1000, { default: 60 }),
      minGap: int(1, 100, { default: 5 }),
      jump: real(0, 10, { default: 3 }),
      minJump: real(0, 10, { default: 1 }),
      sd: real(0.01, 10, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: ['bayesian-online-changepoint-detection'],
  },
  meanShifts,
)

dataset(
  {
    key: 'varianceShifts',
    name: 'Variance shifts',
    summary: 'Gaussian noise whose standard deviation changes between segments.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      meanGap: real(5, 1000, { default: 60 }),
      minGap: int(1, 100, { default: 5 }),
      minRatio: real(1, 10, { default: 1.5 }),
      mean: real(-10, 10, { default: 0 }),
    }),
    truth: true,
    random: true,
    notes: ['bayesian-online-changepoint-detection'],
  },
  varianceShifts,
)

dataset(
  {
    key: 'poissonShifts',
    name: 'Poisson rate shifts',
    summary: 'Counts whose Poisson rate changes between segments.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      meanGap: real(5, 1000, { default: 60 }),
      minGap: int(1, 100, { default: 5 }),
      shape: real(0.1, 20, { default: 2 }),
      rate: real(0.01, 10, { default: 0.5 }),
      minRatio: real(1, 10, { default: 1.5 }),
    }),
    truth: true,
    random: true,
    notes: ['bayesian-online-changepoint-detection'],
  },
  poissonShifts,
)

dataset(
  {
    key: 'arRegimes',
    name: 'AR regimes',
    summary: 'A series that switches between autoregressive regimes at changepoints.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(10, 5000, { default: 300 }),
      meanGap: real(5, 1000, { default: 60 }),
      minGap: int(1, 100, { default: 5 }),
      sd: real(0.01, 10, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: ['autoregressive-model'],
  },
  arRegimes,
)
