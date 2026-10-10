/**
 * Known truth for synthetic datasets. When the generating process has a closed form, a dataset carries it in
 * `meta.truth` as a model (`kind: 'model'`) with the contract's capabilities: the Bayes rule (`decide`), the Bayes
 * posterior or the conditional law of $y$ (`predictive`), the regression function (`expect`), and the lowest risk any
 * predictor can reach (`bayesRisk`). A figure draws the Bayes-optimal boundary and curves beside a fitted model's by
 * calling the same methods on both.
 *
 * A classification truth is built from class-conditional densities $p(\xvec \mid y = j)$, evaluated with the
 * registered distributions of `aifn-compute/probability/distributions`, and clean class priors $\pi_j$, followed by a
 * list of label operations: noise matrices $\Tmat$ ($T_{ij} = \Pr(\text{observed } j \mid \text{label } i)$) and class
 * reweightings $\wvec$ (from resampling by label). The joint of $\xvec$ and the observed label $\tilde y$ is then
 *
 * $p(\xvec, \tilde y) \propto \left((\pivec \circ \pvec(\xvec)) \Tmat_1 \circ \wvec_1 \cdots\right)_{\tilde y}$,
 *
 * applied left to right, and the Bayes posterior is that vector normalised (Duda, Hart and Stork, 2001, "Pattern
 * Classification", §2.2). Modifiers compose by editing this model. Every method takes a batch of points, an
 * $n \times d$ float64 matrix.
 *
 * The other truths follow the same contract: regression $y = m(\xvec) + \varepsilon$ (`regressionTruth`), additive
 * models (`additiveTruth`), one-dimensional curves with their expectiles (`curve1dTruth`), series with a known power
 * spectrum (`spectralTruth`), piecewise series (`changepointTruth`), gated mixtures of regimes (`regimeTruth`) and
 * many-to-one inverse problems (`inverseTruth`).
 */

import type { Distribution, Scores, Size, Truth as TruthContract } from 'aifn-compute/foundation/contracts'
import {
  complex,
  fromData,
  isTensor,
  logsumexp,
  reshape,
  stack,
  toFlat,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import {
  family as familyByName,
  link as linkByName,
  type FamilyName,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'
import { gaussHermite } from 'aifn-compute/numerics/quadrature'
import { normalCdf } from 'aifn-compute/numerics/special'
import {
  Bernoulli,
  Categorical,
  Gamma,
  LogNormal,
  MultivariateNormal,
  Normal,
  Poisson,
  Transformed,
  type AnyUnivariate,
} from 'aifn-compute/probability/distributions'
import { affineBijector } from 'aifn-compute/probability/bijectors'
import { expectiles } from 'aifn-compute/probability/stats'
import { armaSpectrum } from 'aifn-compute/signal/statistical'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A tensor's values, flattened, as a fresh Float64Array.
 *
 * @param t Any tensor.
 * @returns A copy of its values in row-major order.
 */
const toFlatArray = (t: Tensor) => Float64Array.from(toFlat(t))

/** A point, one number per feature (used inside regression functions). */
export type Row = ArrayLike<number>

/**
 * A label operation applied after the clean labels are drawn: label noise, whose `matrix` $\Tmat$ has
 * $T_{ij} = \Pr(\text{observed } j \mid \text{label } i)$ ($k$ rows, each a probability vector), or a reweighting of
 * the classes.
 */
export type LabelOp =
  | { kind: 'noise'; matrix: number[][] }
  | {
      kind: 'weights'
      /** Relative weight of each observed class (resampling by label). */
      weights: number[]
    }

/**
 * Points drawn from the population's marginal of $\xvec$ with importance weights summing to one, for Monte Carlo
 * estimates (the Bayes error when no closed form applies).
 */
export interface Reference {
  /** The $m$ points, an $m \times d$ matrix. */
  x: Tensor
  /** One weight per point, summing to one. */
  weights: Float64Array
}

/** The parts a classification truth is built from; modifiers edit these. */
export interface ClassModel {
  /** The number of clean classes $k$. */
  classes: Size
  /** Clean class proportions $\pi_j$ in the population. */
  priors: number[]
  /**
   * $\log p(\xvec \mid \text{clean } y = j)$ for every row of $\xvec$ ($n \times d$) and class $j$: an $n \times k$
   * matrix, up to a term per row shared by all classes. $-\infty$ outside a class's support.
   */
  logDensity: (x: Tensor) => Tensor
  /** Label operations in order. */
  ops: LabelOp[]
  /** A weighted sample of the population's x (lazy; computed on first use of the Bayes error). */
  reference: () => Reference
  /** The Bayes error of the clean problem for given priors, when it has a closed form. */
  closedForm?: (priors: number[]) => number
  /** True once x has been reweighted (covariate shift), so class proportions must be estimated. */
  shifted?: boolean
  /** What the densities are, for captions: e.g. "two Gaussian classes". */
  family: string
}

/**
 * The Bayes-optimal classifier of a synthetic classification problem, as a model. Methods take points ($n \times d$)
 * and refer to the observed labels `y` of the dataset, after any label noise or resampling.
 */
export interface ClassificationTruth extends TruthContract, Scores<Tensor> {
  /** The task, always classification. */
  readonly task: 'classification'
  /** What the class densities are, e.g. "two moons". */
  readonly name: string
  /** The number of classes $k$. */
  readonly classes: Size
  /** Clean class proportions in the population. */
  readonly priors: readonly number[]
  /** Observed class proportions in the population. */
  readonly prevalence: readonly number[]
  /** $\log p(\xvec \mid \text{clean } y = j)$ per row and class ($n \times k$), up to a term shared by all classes. */
  logDensity(x: Tensor): Tensor
  /** $\Pr(\text{clean } y = j \mid \xvec)$ per row and class ($n \times k$). */
  cleanPosterior(x: Tensor): Tensor
  /**
   * $\Pr(\text{observed } y = j \mid \xvec)$ per row and class ($n \times k$): the Bayes posterior. NaN where $\xvec$
   * has zero density under every class.
   */
  posterior(x: Tensor): Tensor
  /**
   * The Bayes posterior as a distribution over the batch: a Bernoulli over class 1 for two classes, else a
   * Categorical. Rows with zero density under every class get the uniform law.
   */
  predictive(x: Tensor): AnyUnivariate
  /** The Bayes rule: the most probable observed class per row (int32, $n$); $-1$ where the posterior is undefined. */
  decide(x: Tensor): Tensor
  /**
   * $\expect[f(\tilde y) \mid \xvec]$ under the posterior ($n$ values); the mean class index without `f`
   * ($\Pr(\tilde y = 1 \mid \xvec)$ for two classes).
   */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /**
   * The log odds of class 1 against the rest ($n$ values): a Bayes-optimal score for ROC and precision–recall curves.
   */
  score(x: Tensor): Tensor
  /**
   * The Bayes error $\expect[1 - \max_j \Pr(\tilde y = j \mid \xvec)]$, the lowest error any classifier can reach
   * (computed on first access).
   */
  readonly bayesError: number
  /** How `bayesError` is computed: exactly, or by Monte Carlo over a reference sample of the population. */
  readonly bayesErrorMethod: 'closed form' | 'monte carlo'
  /** Standard error of a Monte Carlo `bayesError` (0 in closed form). */
  readonly bayesErrorSe: number
  /** The model the truth is built from (for modifiers). */
  readonly model: ClassModel
}

/**
 * The parts a regression truth is built from: $y = m(\xvec) + \varepsilon$,
 * $\varepsilon \sim \Gauss(0, \sigma(\xvec)^2)$. Modifiers edit these.
 */
export interface RegressionModel {
  /** The regression function $m(\xvec) = \expect[y \mid \xvec]$ at one point. */
  mean: (x: Row) => number
  /** $\sigma(\xvec)$, the noise standard deviation at one point. */
  sdAt: (x: Row) => number
  /** The noise standard deviation (at the left end of the input range when it varies). */
  noiseSd: number
  /** The Bayes risk under squared loss, $\expect[\sigma(\xvec)^2]$. */
  bayesRisk: number
  /** Fraction of targets replaced by gross outliers (see `withOutliers`), 0 by default. */
  outlierFraction: number
  /** What the regression function is, for captions. */
  family: string
}

/**
 * The truth of a synthetic regression problem $y = m(\xvec) + \varepsilon$,
 * $\varepsilon \sim \Gauss(0, \sigma(\xvec)^2)$, as a model.
 */
export interface RegressionTruth extends TruthContract {
  /** The task, always regression. */
  readonly task: 'regression'
  /** What the regression function is, for captions (the model's `family`). */
  readonly name: string
  /** The regression function $m(\xvec)$ per row ($n$ values); the same as `expect`. */
  mean(x: Tensor): Tensor
  /** $\sigma(\xvec)$ per row ($n$ values). */
  noiseSdAt(x: Tensor): Tensor
  /**
   * The conditional law of $y$: $\Gauss(m(\xvec), \sigma(\xvec)^2)$ over the batch. Outliers (if any) are not part of
   * it.
   */
  predictive(x: Tensor): AnyUnivariate
  /** The point prediction that minimises squared loss: $m(\xvec)$ ($n$ values). */
  decide(x: Tensor): Tensor
  /** $\expect[f(y) \mid \xvec]$ ($n$ values); $m(\xvec)$ without `f`, else by 32-point Gauss–Hermite quadrature. */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** The noise standard deviation (at the left end of the input range when it varies). */
  readonly noiseSd: number
  /** Fraction of targets replaced by gross outliers. */
  readonly outlierFraction: number
  /** The model the truth is built from (for modifiers). */
  readonly model: RegressionModel
}

/**
 * One segment of a piecewise series: indices `start` to `end - 1`, its parameters and the law of a value inside it.
 */
export interface Segment {
  /** First index of the segment. */
  readonly start: Size
  /** One past its last index. */
  readonly end: Size
  /** The generating parameters, e.g. `{ mean, sd }`, `{ rate }` or `{ coefficients, constant, sd }`. */
  readonly params: Readonly<Record<string, number | readonly number[]>>
  /** Mean of a value in the segment (the stationary mean for an autoregression). */
  readonly mean: number
  /** Variance of a value in the segment (the stationary variance for an autoregression). */
  readonly variance: number
  /**
   * Mean squared error of predicting a value from the true parameters (the innovation variance for an
   * autoregression).
   */
  readonly risk: number
}

/** What changes between segments. */
export type ChangepointFamily = 'mean' | 'variance' | 'poisson' | 'autoregressive'

/**
 * The truth of a piecewise series: its segments and changepoints, as a model whose inputs are times $t$ (a float64
 * vector of integer indices). `decide` gives the segment index (int32), `predictive` the law of the value at $t$
 * (normal, or Poisson for counts; the stationary marginal for an autoregression), `expect` its mean.
 */
export interface ChangepointTruth extends TruthContract {
  /** The task, always changepoint detection. */
  readonly task: 'changepoint'
  /** What the series is, for captions. */
  readonly name: string
  /** What changes between segments. */
  readonly family: ChangepointFamily
  /** Length of the series. */
  readonly n: Size
  /** Indices where a new segment begins (0 excluded), ascending. */
  readonly changepoints: readonly Size[]
  /** The segments in order, covering $0, \dots, n - 1$. */
  readonly segments: readonly Segment[]
  /** The segment index at time $t$ (rounded to the nearest index and clamped to the series). */
  segmentAt(t: number): Size
  /** The segment index at each time (int32). */
  decide(t: Tensor): Tensor
  /** The law of the value at each time: normal, or Poisson for counts. */
  predictive(t: Tensor): AnyUnivariate
  /**
   * The mean of the value at each time without `f`; with `f`, $\expect[f(y)]$ under the normal law by Gauss–Hermite
   * quadrature (counts: the Poisson sum up to a far tail).
   */
  expect(t: Tensor, f?: (y: number) => number): Tensor
}

/**
 * The truth of a generalised additive model $g(\expect[y \mid \xvec]) = \alpha + \sum_j f_j(x_j)$ with $y$ from an
 * exponential-dispersion family (`additiveData`): the true partial effects $f_j$ on the link scale, each centred to
 * mean zero over $x_j \sim \Unif(0, 1)$, and the conditional law of $y$.
 */
export interface AdditiveTruth extends TruthContract {
  /** The task, always regression. */
  readonly task: 'regression'
  /** The family and link, for captions. */
  readonly name: string
  /** The exponential-dispersion family of $y$. */
  readonly family: FamilyName
  /** The link $g$. */
  readonly link: LinkName
  /** $\alpha$: the mean of $\eta$ over the population. */
  readonly intercept: number
  /** The dispersion $\phi$ (Gaussian: $\sigma^2$; gamma: the squared coefficient of variation; 1 otherwise). */
  readonly dispersion: number
  /** Each feature's shape name. */
  readonly shapes: readonly string[]
  /**
   * The true partial effect of feature $j$ on a grid of $m$ values, centred to mean zero over $\Unif(0, 1)$, or over
   * the $n$ values `centreOn` when given (as a fitted GAM centres its smooths on the training data).
   */
  partial(j: Size, grid: Tensor, centreOn?: Tensor): Tensor
  /** $\eta(\xvec) = \alpha + \sum_j f_j(x_j)$, $n$ values. */
  linearPredictor(x: Tensor): Tensor
  /** $\mu(\xvec) = g^{-1}(\eta(\xvec))$, $n$ values. */
  mean(x: Tensor): Tensor
  /** The family's law of $y$ at $\mu(\xvec)$ and $\phi$. */
  predictive(x: Tensor): Distribution
  /** The prediction under squared loss, $\mu(\xvec)$. */
  decide(x: Tensor): Tensor
  /** $\mu(\xvec)$ without `f`; with `f`, Gaussian only (Gauss–Hermite). */
  expect(x: Tensor, f?: (y: number) => number): Tensor
}

/** The parts of an additive truth. */
export interface AdditiveModel {
  /** The exponential-dispersion family of $y$. */
  family: FamilyName
  /** The link $g$. */
  link: LinkName
  /** The intercept $\alpha$ on the link scale. */
  intercept: number
  /** The dispersion $\phi$. */
  dispersion: number
  /** Each feature's effect on $[0, 1]$, already centred over $\Unif(0, 1)$, with its name. */
  effects: readonly { name: string; f: (x: number) => number }[]
}

/**
 * Build an additive truth (see `AdditiveTruth`). Its Bayes risk $\expect[\phi V(\mu(\xvec))]$ ($V$ the family's
 * variance function) is averaged over a midpoint grid of $[0, 1]^d$ of at most 4096 points. Its methods throw
 * `ShapeError` for points with other than one column per effect, `partial` throws `DomainError` for an unknown
 * feature, and `expect` with `f` throws `DomainError` unless the family is Gaussian.
 *
 * @param model The family, link, intercept, dispersion and centred effects.
 * @returns The truth.
 *
 * @example A Gaussian additive model with two effects
 * const truth = additiveTruth({
 *   family: 'gaussian',
 *   link: 'identity',
 *   intercept: 1,
 *   dispersion: 0.25,
 *   effects: [
 *     { name: 'linear', f: (x) => x - 0.5 },
 *     { name: 'sine', f: (x) => Math.sin(2 * Math.PI * x) },
 *   ],
 * })
 * print(truth.name)
 * print('eta at (0.5, 0.25) and (1, 0.5):', truth.linearPredictor(tensor([[0.5, 0.25], [1, 0.5]])))
 * print('Bayes risk (the noise variance):', truth.bayesRisk)
 */
export function additiveTruth(model: AdditiveModel): AdditiveTruth {
  const fam = familyByName(model.family)
  const g = linkByName(model.link)
  const linearPredictor = (x: Tensor) => {
    const { data, n, d } = points(x)
    if (d !== model.effects.length)
      throw new ShapeError('additiveTruth', `additiveTruth: ${model.effects.length} features, given ${d}`)
    return fromData(
      Float64Array.from({ length: n }, (_, i) =>
        model.effects.reduce((acc, e, j) => acc + e.f(data[i * d + j]), model.intercept),
      ),
      [n],
    )
  }
  const mean = (x: Tensor) => g.inverse(linearPredictor(x)) as Tensor
  const predictive = (x: Tensor) => fam.predictive(mean(x), model.dispersion)
  // E[φV(μ(x))] over x ~ U(0, 1)ᵈ on a midpoint grid (12 points a side, at most 4096 points).
  const d = model.effects.length
  const side = Math.max(2, Math.min(12, Math.floor(4096 ** (1 / Math.max(d, 1)))))
  const count = side ** d
  const cells = new Float64Array(count * d)
  for (let c = 0; c < count; c++)
    for (let j = 0, r = c; j < d; j++, r = Math.floor(r / side)) cells[c * d + j] = ((r % side) + 0.5) / side
  const V = toFlatArray(fam.variance(mean(fromData(cells, [count, d]))) as Tensor)
  const bayesRisk = (model.dispersion * V.reduce((a, b) => a + b, 0)) / count
  return {
    kind: 'model',
    task: 'regression',
    name: `${model.family} additive model, ${model.link} link`,
    family: model.family,
    link: model.link,
    intercept: model.intercept,
    dispersion: model.dispersion,
    shapes: model.effects.map((e) => e.name),
    partial: (j, grid, centreOn) => {
      const e = model.effects[j]
      if (!e) throw new DomainError('additiveTruth', `additiveTruth: no feature ${j}`)
      const values = toFlatArray(grid).map(e.f)
      let shift = 0
      if (centreOn) {
        const c = toFlatArray(centreOn)
        shift = c.reduce((a, v) => a + e.f(v), 0) / c.length
      }
      return fromData(
        values.map((v) => v - shift),
        [values.length],
      )
    },
    linearPredictor,
    mean,
    predictive,
    decide: mean,
    expect: (x, f) => {
      if (!f) return mean(x)
      if (model.family !== 'gaussian')
        throw new DomainError('additiveTruth', 'additiveTruth: expect(x, f) is available for the Gaussian family')
      const { nodes, weights } = HERMITE()
      const sd = Math.sqrt(model.dispersion)
      return fromData(
        toFlatArray(mean(x)).map((m) => nodes.reduce((acc, z, q) => acc + weights[q] * f(m + sd * z), 0)),
        [x.shape[0]],
      )
    },
    bayesRisk,
  }
}

/**
 * The truth of a one-dimensional smooth regression (`curve1d`) with $x \sim \Unif(0, 1)$: the mean
 * $\mu(x) = g^{-1}(\eta(x))$ and the whole law of $y$, so its expectiles $e_\tau(x)$ are known. A location–scale law
 * $y = \mu(x) + \sigma(x) Z$ with standardised noise $Z$ (normal, or a skewed shifted log-normal) has
 * $e_\tau(x) = \mu(x) + \sigma(x) e_\tau(Z)$; a gamma law $y = \mu(x) G$ has $e_\tau(x) = \mu(x) e_\tau(G)$; Poisson
 * and Bernoulli expectiles come from their masses at each $x$. Expectiles are computed numerically from the law
 * (`expectiles` on its atoms: 4000 equally weighted quantiles, or the masses), not in closed form.
 */
export interface Curve1dTruth extends TruthContract {
  /** The task, always regression. */
  readonly task: 'regression'
  /** The curve's name, for captions. */
  readonly name: string
  /** The family and link the data were drawn from (location–scale noise reports `gaussian`, `identity`). */
  readonly family: FamilyName
  /** The link $g$. */
  readonly link: LinkName
  /** The law of $y$ around its mean, for captions. */
  readonly law: string
  /** $\eta(x) = g(\mu(x))$, $n$ values. */
  linearPredictor(x: Tensor): Tensor
  /** $\mu(x) = \expect[y \mid x]$, $n$ values. */
  mean(x: Tensor): Tensor
  /** The standard deviation of $y$ at $x$, $n$ values. */
  sdAt(x: Tensor): Tensor
  /** The $\tau$-expectile of $y$ given $x$, $n$ values, for $\tau \in (0, 1)$. */
  expectile(x: Tensor, tau: number): Tensor
  /**
   * $\Pr(y < e_\tau(x))$ averaged over $x \sim \Unif(0, 1)$: the share of the population below the true
   * $\tau$-expectile curve. Throws `DomainError` unless $\tau \in (0, 1)$.
   */
  shareBelow(tau: number): number
  /** The law of $y$ at each $x$: the family's, normal, or the shifted log-normal of the skewed noise. */
  predictive(x: Tensor): Distribution
  /** The prediction under squared loss, $\mu(x)$. */
  decide(x: Tensor): Tensor
  /** $\mu(x)$ without `f`; with `f`, $\expect[f(y) \mid x]$ over the law's atoms. */
  expect(x: Tensor, f?: (y: number) => number): Tensor
}

/**
 * How $y$ varies around $\mu(x)$ in a `Curve1dModel`: location–scale noise of standard deviation `sd(x)`, or the
 * exponential family of the model with dispersion `dispersion`.
 */
export type Curve1dLaw =
  /**
   * $y = \mu(x) + \sigma(x) Z$, $Z$ standardised: normal, or a log-normal with log-scale standard deviation `skew`
   * (default 0.75), shifted.
   */
  | { kind: 'location-scale'; noise: 'normal' | 'skewed'; sd: (x: number) => number; skew?: number }
  /**
   * $y$ from the exponential family at $\mu(x)$ with dispersion $\phi$ (Poisson, Bernoulli, or gamma with squared
   * coefficient of variation $\phi$).
   */
  | { kind: 'family'; dispersion: number }

/** The parts of a `Curve1dTruth`. */
export interface Curve1dModel {
  /** The curve's name, for captions. */
  name: string
  /** The exponential family of $y$ (`gaussian` for location–scale noise). */
  family: FamilyName
  /** The link $g$. */
  link: LinkName
  /** $\eta(x)$ on $[0, 1]$; $\mu = g^{-1}(\eta)$. */
  eta: (x: number) => number
  /** How $y$ varies around $\mu(x)$. */
  law: Curve1dLaw
}

/** The number of equally weighted quantiles that stand for a continuous law. */
const ATOMS = 4000
/** The quantile levels $(i + \tfrac12) / 4000$, $i = 0, \dots, 3999$, made on first use. */
const levels = lazy(() =>
  fromData(
    Float64Array.from({ length: ATOMS }, (_, i) => (i + 0.5) / ATOMS),
    [ATOMS],
  ),
)

/**
 * Build a one-dimensional smooth regression truth (see `Curve1dTruth`). Its Bayes risk is the variance of $y$
 * averaged over 200 midpoints of $[0, 1]$. A family law must be Poisson, binomial (Bernoulli) or gamma: the
 * standard deviation of any other throws `DomainError`, when the truth is built.
 *
 * @param model The name, family, link, linear predictor and law of $y$.
 * @returns The truth.
 *
 * @example A sine with normal noise: expectiles above and below the mean
 * const truth = curve1dTruth({
 *   name: 'sine',
 *   family: 'gaussian',
 *   link: 'identity',
 *   eta: (x) => Math.sin(2 * Math.PI * x),
 *   law: { kind: 'location-scale', noise: 'normal', sd: () => 0.5 },
 * })
 * const x = tensor([0.25, 0.75])
 * print('mean at 0.25 and 0.75:', truth.mean(x))
 * print('0.9-expectile there:', truth.expectile(x, 0.9))
 * print('share below the 0.9-expectile:', truth.shareBelow(0.9))
 */
export function curve1dTruth(model: Curve1dModel): Curve1dTruth {
  const { law } = model
  const g = linkByName(model.link)
  const muAt = (x: number) => {
    const m = g.inverse(model.eta(x))
    return typeof m === 'number' ? m : toFlat(m as Tensor)[0]
  }
  const column = (x: Tensor) => toFlatArray(x)
  const perRow = (x: Tensor, f: (v: number) => number) => {
    const c = column(x)
    return fromData(Float64Array.from(c, f), [c.length])
  }
  const skew = law.kind === 'location-scale' ? (law.skew ?? 0.75) : 0
  // The skewed noise: Z = (V − c)/d for V ~ LogNormal(0, s), c = e^{s²/2} and d² = (e^{s²} − 1)e^{s²}.
  const c = Math.exp((skew * skew) / 2)
  const d = Math.sqrt((Math.exp(skew * skew) - 1) * Math.exp(skew * skew))
  const phi = law.kind === 'family' ? law.dispersion : 1
  // Equally weighted quantiles of the standardised noise (location–scale) or of G = y/μ (gamma); null for the
  // discrete families, whose atoms depend on x.
  const unit = lazy((): Float64Array | null => {
    if (law.kind === 'location-scale') {
      if (law.noise === 'normal') return toFlatArray(Normal(0, 1).quantile(levels()) as Tensor)
      return toFlatArray(LogNormal(0, skew).quantile(levels()) as Tensor).map((v) => (v - c) / d)
    }
    if (model.family === 'gamma') return toFlatArray(Gamma(1 / phi, 1 / phi).quantile(levels()) as Tensor)
    return null
  })
  const sdOf = (x: number) => {
    const mu = muAt(x)
    if (law.kind === 'location-scale') return law.sd(x)
    if (model.family === 'gamma') return Math.sqrt(phi) * mu
    if (model.family === 'poisson') return Math.sqrt(mu)
    if (model.family === 'binomial') return Math.sqrt(mu * (1 - mu))
    throw new DomainError('curve1dTruth', `curve1dTruth: no law for the ${model.family} family`)
  }
  /** The law of y at x as atoms with masses. */
  const atomsAt = (x: number): { values: Float64Array; masses: Float64Array } => {
    const mu = muAt(x)
    const u = unit()
    if (u) {
      const values = law.kind === 'location-scale' ? u.map((z) => mu + law.sd(x) * z) : u.map((v) => mu * v)
      return { values, masses: new Float64Array(u.length).fill(1 / u.length) }
    }
    if (model.family === 'binomial') return { values: Float64Array.of(0, 1), masses: Float64Array.of(1 - mu, mu) }
    // Poisson: the masses up to a far tail.
    const top = Math.ceil(mu + 12 * Math.sqrt(mu) + 20)
    const values = Float64Array.from({ length: top + 1 }, (_, k) => k)
    const masses = new Float64Array(top + 1)
    let p = Math.exp(-mu)
    for (let k = 0; k <= top; k++) {
      masses[k] = p
      p *= mu / (k + 1)
    }
    return { values, masses }
  }
  // The unit law's expectiles, cached per τ (location–scale and gamma: e_τ(x) = μ + σe_τ(Z) or μe_τ(G)).
  const unitExpectile = new Map<number, number>()
  const eUnit = (tau: number) => {
    let e = unitExpectile.get(tau)
    if (e === undefined) {
      e = expectiles(unit()!, [tau])[0]
      unitExpectile.set(tau, e)
    }
    return e
  }
  const expectileAt = (x: number, tau: number) => {
    if (unit()) {
      const e = eUnit(tau)
      return law.kind === 'location-scale' ? muAt(x) + law.sd(x) * e : muAt(x) * e
    }
    const { values, masses } = atomsAt(x)
    return expectiles(values, [tau], { weights: masses })[0]
  }
  // x ~ U(0, 1) on 200 midpoints, for population averages.
  const GRID = Float64Array.from({ length: 200 }, (_, i) => (i + 0.5) / 200)
  const shareBelow = (tau: number) => {
    if (!(tau > 0 && tau < 1)) throw new DomainError('curve1dTruth', `curve1dTruth: τ = ${tau} is not in (0, 1)`)
    const u = unit()
    if (u) {
      // The same at every x: P(Z < e_τ(Z)).
      const e = eUnit(tau)
      let below = 0
      for (const v of u) if (v < e) below++
      return below / u.length
    }
    let total = 0
    for (const x of GRID) {
      const e = expectileAt(x, tau)
      const { values, masses } = atomsAt(x)
      for (let k = 0; k < values.length; k++) if (values[k] < e) total += masses[k]
    }
    return total / GRID.length
  }
  const mean = (x: Tensor) => perRow(x, muAt)
  const predictive = (x: Tensor): Distribution => {
    const mu = mean(x)
    if (law.kind === 'family') return familyByName(model.family).predictive(mu, phi)
    const sd = perRow(x, law.sd)
    if (law.noise === 'normal') return Normal(mu, sd)
    // σ(x)V/d ~ LogNormal(log(σ(x)/d), s), shifted by μ(x) − σ(x)c/d.
    const m = toFlatArray(mu)
    const s = toFlatArray(sd)
    const logScale = fromData(
      Float64Array.from(s, (v) => Math.log(v / d)),
      [s.length],
    )
    const shift = fromData(
      Float64Array.from(m, (v, i) => v - (s[i] * c) / d),
      [m.length],
    )
    return Transformed(LogNormal(logScale, skew), affineBijector(shift, 1))
  }
  let risk = 0
  for (const x of GRID) risk += sdOf(x) ** 2 / GRID.length
  const lawText =
    law.kind === 'location-scale'
      ? law.noise === 'normal'
        ? 'normal noise'
        : `skewed noise (a shifted log-normal, log-scale sd ${skew})`
      : model.family === 'gamma'
        ? `gamma, coefficient of variation ${Math.sqrt(phi).toFixed(2)}`
        : model.family === 'poisson'
          ? 'Poisson counts'
          : 'Bernoulli outcomes'
  return {
    kind: 'model',
    task: 'regression',
    name: model.name,
    family: model.family,
    link: model.link,
    law: lawText,
    linearPredictor: (x) => perRow(x, model.eta),
    mean,
    sdAt: (x) => perRow(x, sdOf),
    expectile: (x, tau) => perRow(x, (v) => expectileAt(v, tau)),
    shareBelow,
    predictive,
    decide: mean,
    expect: (x, f) => {
      if (!f) return mean(x)
      return perRow(x, (v) => {
        const { values, masses } = atomsAt(v)
        let acc = 0
        for (let k = 0; k < values.length; k++) acc += masses[k] * f(values[k])
        return acc
      })
    },
    bayesRisk: risk,
  }
}

// ── Spectra: sinusoids plus a stationary ARMA process, with a known power spectrum ───────────────────────────────────

/**
 * One deterministic sinusoid $A \sin(2\pi f t + \varphi)$ of a spectral model; its power in a one-sided spectrum is
 * $A^2 / 2$.
 */
export interface SpectralLine {
  /** Frequency $f$, in Hz (or cycles per unit of time). */
  readonly frequency: number
  /** Amplitude $A$. */
  readonly amplitude: number
  /** Phase $\varphi$, in radians. */
  readonly phase: number
}

/**
 * An ARMA($p$, $q$) process $x_t = \sum_i \phi_i x_{t-i} + \varepsilon_t + \sum_j \theta_j \varepsilon_{t-j}$,
 * $\varepsilon_t \sim \Gauss(0, \sigma^2)$; white noise has no coefficients.
 */
export interface ArmaParts {
  /** The AR coefficients $\phi_1, \dots, \phi_p$. */
  readonly ar: readonly number[]
  /** The MA coefficients $\theta_1, \dots, \theta_q$. */
  readonly ma: readonly number[]
  /** The innovation variance $\sigma^2$. */
  readonly sigma2: number
}

/**
 * The parts of a `SpectralTruth`: sinusoids plus a stationary ARMA process, sampled at rate $f_s$; optionally a second
 * series $y = h * x + v$ coupled to the first through a rational filter $H = B / A$ and independent ARMA noise $v$.
 */
export interface SpectralModel {
  /** What the series is, for captions. */
  readonly name: string
  /** Sample rate (for uneven sampling, the mean rate $n / T$, which only scales the noise density). */
  readonly fs: number
  /** The sinusoids. */
  readonly lines: readonly SpectralLine[]
  /** The stationary process added to them. */
  readonly noise: ArmaParts
  /** The second series of a coupled pair: the filter's numerator `b` and denominator `a`, and its noise $v$. */
  readonly coupling?: { readonly b: readonly number[]; readonly a: readonly number[]; readonly noise: ArmaParts }
}

/** The true spectra of a coupled pair $(x, y)$, with $y = h * x + v$, at the frequencies $f$ asked for. */
export interface CoupledSpectra {
  /** The one-sided PSD of $y$, $\lvert H \rvert^2 S_{xx} + S_{vv}$. */
  psd(f: Tensor | ArrayLike<number>): Tensor
  /** The cross-spectral density $S_{xy} = H S_{xx}$ (scipy's `conj(X) * Y` convention), complex128. */
  crossSpectrum(f: Tensor | ArrayLike<number>): Tensor
  /**
   * The magnitude-squared coherence $\lvert H \rvert^2 S_{xx} / (\lvert H \rvert^2 S_{xx} + S_{vv})$ (0 where both
   * vanish).
   */
  coherence(f: Tensor | ArrayLike<number>): Tensor
  /** The phase of $S_{xy}$, $\arg H(f)$, in radians (unwrapped along $f$, in the order given). */
  phase(f: Tensor | ArrayLike<number>): Tensor
}

/**
 * The truth of a series with a known power spectrum (task `spectrum`): a line spectrum (sinusoids of known frequency,
 * amplitude and phase) on top of the continuous spectrum of a stationary ARMA process, which is white noise when the
 * process has no coefficients. Methods over times (an $n \times 1$ or length-$n$ tensor): `expect` is the sum of the
 * sinusoids, `predictive` the marginal normal law around it, `bayesRisk` the variance of the stochastic part. `psd(f)`
 * is the continuous part as a one-sided density (`aifn-compute/signal/statistical`'s `armaSpectrum`), and `lines` the
 * line part, each line of power $A^2 / 2$: a periodogram of $n$ samples shows a line as a peak of height about
 * $(A^2 / 2) \cdot n / f_s$ in density units on top of `psd`.
 */
export interface SpectralTruth extends TruthContract {
  /** The task, always spectral estimation. */
  readonly task: 'spectrum'
  /** What the series is, for captions. */
  readonly name: string
  /** The sample rate $f_s$. */
  readonly fs: number
  /** The line part: the sinusoids. */
  readonly lines: readonly SpectralLine[]
  /** The continuous part's one-sided power spectral density at frequencies $f$. */
  psd(f: Tensor | ArrayLike<number>): Tensor
  /** The variance of the stochastic part (the integral of `psd` over $[0, f_s / 2]$). */
  readonly noiseVariance: number
  /** The total variance: `noiseVariance` $+ \sum A^2 / 2$. */
  readonly variance: number
  /** For a coupled pair, the second series' spectra and the cross-spectra. */
  readonly coupled?: CoupledSpectra
  /** The model the truth is built from. */
  readonly model: SpectralModel
}

/**
 * Frequencies as a fresh Float64Array.
 *
 * @param f The frequencies: a tensor (read flat) or an array.
 * @returns A copy of them.
 */
const freqArray = (f: Tensor | ArrayLike<number>): Float64Array => Float64Array.from(isTensor(f) ? toFlat(f) : f)

/**
 * The variance of a stationary ARMA process from its impulse response, $\sigma^2 \sum_j \psi_j^2$ with $\psi_0 = 1$
 * and $\psi_j = \theta_j + \sum_i \phi_i \psi_{j-i}$ ($\theta_j = 0$ past $q$). The sum stops once two successive
 * $\psi_j$ past the first $p + q$ are below $10^{-12}$ in size, or after 200000 terms: a process that is not
 * stationary gets a meaningless (large, infinite or NaN) value, not an error.
 *
 * @param p The process: AR coefficients $\phi$, MA coefficients $\theta$ and innovation variance $\sigma^2$.
 * @returns The variance of $x_t$ ($\sigma^2$ itself for white noise).
 *
 * @example AR(1) and MA(1) against their closed forms
 * print('AR(1), phi = 0.5 (1 / (1 - 0.25)):', armaVariance({ ar: [0.5], ma: [], sigma2: 1 }))
 * print('MA(1), theta = 0.5 (1 + 0.25):', armaVariance({ ar: [], ma: [0.5], sigma2: 1 }))
 */
export function armaVariance(p: ArmaParts): number {
  if (!p.ar.length && !p.ma.length) return p.sigma2
  const psi: number[] = [1]
  let total = 1
  for (let j = 1; j < 200000; j++) {
    let v = j <= p.ma.length ? p.ma[j - 1] : 0
    for (let i = 1; i <= Math.min(j, p.ar.length); i++) v += p.ar[i - 1] * psi[j - i]
    psi.push(v)
    total += v * v
    if (j > p.ma.length + p.ar.length && Math.abs(v) < 1e-12 && Math.abs(psi[j - 1]) < 1e-12) break
  }
  return p.sigma2 * total
}

/**
 * A filter's frequency response $H(e^{i\omega}) = B(e^{-i\omega}) / A(e^{-i\omega})$, with
 * $B(z) = \sum_k b_k z^k$ and $A(z) = \sum_k a_k z^k$.
 *
 * @param b The numerator coefficients $b_0, b_1, \dots$.
 * @param a The denominator coefficients $a_0, a_1, \dots$.
 * @param omega The angular frequency $\omega = 2\pi f / f_s$, in radians per sample.
 * @returns The response as `[re, im]`.
 */
function response(b: readonly number[], a: readonly number[], omega: number): [number, number] {
  const poly = (c: readonly number[]) => {
    let re = 0
    let im = 0
    c.forEach((v, k) => {
      re += v * Math.cos(omega * k)
      im -= v * Math.sin(omega * k)
    })
    return [re, im]
  }
  const [br, bi] = poly(b)
  const [ar, ai] = poly(a)
  const d = ar * ar + ai * ai
  return [(br * ar + bi * ai) / d, (bi * ar - br * ai) / d]
}

/**
 * Build a spectral truth (see `SpectralTruth`). With `model.coupling`, it has the coupled pair's spectra in `coupled`.
 *
 * @param model The sample rate, the lines, the stationary process and any coupling.
 * @returns The truth.
 *
 * @example A tone on white noise
 * const truth = spectralTruth({
 *   name: 'tone in noise',
 *   fs: 1,
 *   lines: [{ frequency: 0.1, amplitude: 2, phase: 0 }],
 *   noise: { ar: [], ma: [], sigma2: 1 },
 * })
 * print('PSD at 0.1 and 0.3 (2 sigma^2 / fs):', truth.psd([0.1, 0.3]))
 * print('noise variance:', truth.noiseVariance, ' total (1 + A^2 / 2):', truth.variance)
 * print('mean at t = 0, 2.5:', truth.expect(tensor([0, 2.5])))
 */
export function spectralTruth(model: SpectralModel): SpectralTruth {
  const { fs, lines, noise } = model
  const psdOf = (p: ArmaParts, f: Float64Array) =>
    armaSpectrum({ ar: [...p.ar], ma: [...p.ma], sigma2: p.sigma2 }, { fs, frequencies: f }).values.data as Float64Array
  const psd = (f: Tensor | ArrayLike<number>) => {
    const fr = freqArray(f)
    return fromData(Float64Array.from(psdOf(noise, fr)), [fr.length])
  }
  const noiseVariance = armaVariance(noise)
  const variance = noiseVariance + lines.reduce((acc, l) => acc + (l.amplitude * l.amplitude) / 2, 0)
  const times = (t: Tensor) => Float64Array.from(toFlat(t))
  const meanAt = (t: number) =>
    lines.reduce((acc, l) => acc + l.amplitude * Math.sin(2 * Math.PI * l.frequency * t + l.phase), 0)
  const mean = (t: Tensor) => {
    const ts = times(t)
    return fromData(Float64Array.from(ts, meanAt), [ts.length])
  }
  const sd = Math.sqrt(noiseVariance)
  let coupled: CoupledSpectra | undefined
  if (model.coupling) {
    const c = model.coupling
    const parts = (f: Tensor | ArrayLike<number>) => {
      const fr = freqArray(f)
      const sxx = psdOf(noise, fr)
      const svv = psdOf(c.noise, fr)
      const H = Array.from(fr, (v) => response(c.b, c.a, (2 * Math.PI * v) / fs))
      return { fr, sxx, svv, H }
    }
    coupled = {
      psd: (f) => {
        const { fr, sxx, svv, H } = parts(f)
        return fromData(
          Float64Array.from(fr, (_, i) => (H[i][0] ** 2 + H[i][1] ** 2) * sxx[i] + svv[i]),
          [fr.length],
        )
      },
      crossSpectrum: (f) => {
        const { fr, sxx, H } = parts(f)
        return complex(
          fromData(
            Float64Array.from(fr, (_, i) => H[i][0] * sxx[i]),
            [fr.length],
          ),
          fromData(
            Float64Array.from(fr, (_, i) => H[i][1] * sxx[i]),
            [fr.length],
          ),
        ) as Tensor
      },
      coherence: (f) => {
        const { fr, sxx, svv, H } = parts(f)
        return fromData(
          Float64Array.from(fr, (_, i) => {
            const g = (H[i][0] ** 2 + H[i][1] ** 2) * sxx[i]
            return g + svv[i] > 0 ? g / (g + svv[i]) : 0
          }),
          [fr.length],
        )
      },
      phase: (f) => {
        const { fr, H } = parts(f)
        const out = new Float64Array(fr.length)
        let offset = 0
        for (let i = 0; i < fr.length; i++) {
          const p = Math.atan2(H[i][1], H[i][0])
          if (i > 0) {
            const prev = out[i - 1] - offset
            if (p - prev > Math.PI) offset -= 2 * Math.PI
            else if (p - prev < -Math.PI) offset += 2 * Math.PI
          }
          out[i] = p + offset
        }
        return fromData(out, [fr.length])
      },
    }
  }
  return {
    kind: 'model',
    task: 'spectrum',
    name: model.name,
    fs,
    lines,
    psd,
    noiseVariance,
    variance,
    ...(coupled ? { coupled } : {}),
    decide: mean,
    predictive: (t) => {
      const m = mean(t)
      return Normal(m, fromData(new Float64Array(m.shape[0]).fill(sd), [m.shape[0]]))
    },
    expect: (t, f) => {
      if (!f) return mean(t)
      const { nodes, weights } = HERMITE()
      const ts = times(t)
      return fromData(
        Float64Array.from(ts, (v) => {
          const m = meanAt(v)
          return nodes.reduce((acc, z, q) => acc + weights[q] * f(m + sd * z), 0)
        }),
        [ts.length],
      )
    },
    bayesRisk: noiseVariance,
    model,
  }
}

/** The truth a dataset can carry in `meta.truth`, one kind per task and generating process. */
export type Truth =
  | ClassificationTruth
  | RegressionTruth
  | ChangepointTruth
  | AdditiveTruth
  | Curve1dTruth
  | RegimeTruth
  | InverseTruth
  | SpectralTruth

/** Number of reference points drawn for Monte Carlo Bayes errors. */
export const REFERENCE_SIZE = 6000

// ── Batches ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The rows of an $n \times d$ matrix as a fresh row-major Float64Array, with $n$ and $d$ (any strides are read).
 * Throws `ShapeError` unless the tensor has rank 2.
 *
 * @param x The points, an $n \times d$ matrix.
 * @returns `data`, the $nd$ values row by row, and the shape `n`, `d`.
 */
export function points(x: Tensor): { data: Float64Array; n: Size; d: Size } {
  if (x.shape.length !== 2)
    throw new ShapeError('truth', `truth: points must be an [n, d] matrix, got rank ${x.shape.length}`)
  const [n, d] = x.shape
  const data = new Float64Array(n * d)
  const [s0, s1] = x.strides
  const src = x.data
  for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) data[i * d + c] = Number(src[x.offset + i * s0 + c * s1])
  return { data, n, d }
}

/**
 * An $n \times d$ float64 matrix from row-major data.
 *
 * @param data The $nd$ values, row by row (used as they are, not copied).
 * @param n The number of rows.
 * @param d The number of columns.
 * @returns The matrix.
 */
export function pointsFrom(data: Float64Array, n: Size, d: Size): Tensor {
  return fromData(data, [n, d])
}

/**
 * Row-major values of a tensor as a fresh Float64Array: a matrix through `points`, a vector through its stride, and
 * anything else as its first value only (a scalar).
 *
 * @param t A tensor of rank 0, 1 or 2.
 * @returns A copy of its values.
 */
function flat(t: Tensor): Float64Array {
  if (t.shape.length === 2) return points(t).data
  if (t.shape.length === 1)
    return Float64Array.from({ length: t.shape[0] }, (_, i) => Number(t.data[t.offset + i * t.strides[0]]))
  return Float64Array.of(Number(t.data[t.offset]))
}

/**
 * Per-row log densities of $k$ classes as one $n \times k$ matrix, as a `ClassModel`'s `logDensity` returns it.
 *
 * @param columns One vector of $n$ log densities per class, in class order.
 * @returns The $n \times k$ matrix.
 */
export function classColumns(columns: readonly Tensor[]): Tensor {
  return stack(columns, 1)
}

/**
 * A value computed on first use and cached.
 *
 * @param f Computes the value; called at most once.
 * @returns A function returning the value.
 */
function lazy<T>(f: () => T): () => T {
  let cached: { value: T } | undefined
  return () => (cached ??= { value: f() }).value
}

/**
 * Apply the label operations, in order, to an unnormalised vector over clean classes: a reweighting multiplies class
 * $j$ by its weight, a noise matrix maps the vector $\pvec$ to $\pvec^\top \Tmat$ (over observed classes).
 *
 * @param p The unnormalised probabilities (or prior-weighted densities) of the clean classes (not modified).
 * @param ops The label operations.
 * @returns The unnormalised vector over observed classes.
 */
function applyOps(p: number[], ops: readonly LabelOp[]): number[] {
  let v = p
  for (const op of ops) {
    if (op.kind === 'weights') v = v.map((a, j) => a * op.weights[j])
    else {
      const m = op.matrix
      const out = new Array<number>(m[0].length).fill(0)
      for (let i = 0; i < v.length; i++) if (v[i] !== 0) for (let j = 0; j < out.length; j++) out[j] += v[i] * m[i][j]
      v = out
    }
  }
  return v
}

/**
 * A vector scaled to sum to one.
 *
 * @param v Non-negative values with a positive sum (a zero sum gives NaN).
 * @returns The values divided by their sum.
 */
function normalise(v: number[]): number[] {
  const s = v.reduce((a, b) => a + b, 0)
  return v.map((a) => a / s)
}

/**
 * The clean log joint $\log \pi_j + \log p(\xvec \mid j)$, up to a term per row.
 *
 * @param model The class model.
 * @param x The points, $n \times d$.
 * @returns `a`, the $n \times k$ values row-major, with `n` and `k`.
 */
function logJoint(model: ClassModel, x: Tensor): { a: Float64Array; n: Size; k: Size } {
  const ld = model.logDensity(x)
  const k = model.classes
  const a = flat(ld)
  const n = a.length / k
  const logPrior = model.priors.map(Math.log)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) a[i * k + j] += logPrior[j]
  return { a, n, k }
}

/**
 * The observed-label posterior: the clean joint at each point, scaled, passed through the label operations and
 * normalised.
 *
 * @param model The class model.
 * @param x The points, $n \times d$.
 * @returns `p`, the $n \times k$ posterior row-major ($k$ the number of observed classes; NaN rows where every class
 *   has zero density), with `n` and `k`.
 */
function posteriorData(model: ClassModel, x: Tensor): { p: Float64Array; n: Size; k: Size } {
  const { a, n, k } = logJoint(model, x)
  const kOut = model.ops.reduce((c, op) => (op.kind === 'noise' ? op.matrix[0].length : c), k)
  const p = new Float64Array(n * kOut)
  for (let i = 0; i < n; i++) {
    const row = Array.from(a.subarray(i * k, (i + 1) * k))
    const m = Math.max(...row)
    if (!(m > -Infinity)) {
      p.fill(NaN, i * kOut, (i + 1) * kOut)
      continue
    }
    p.set(
      normalise(
        applyOps(
          row.map((v) => Math.exp(v - m)),
          model.ops,
        ),
      ),
      i * kOut,
    )
  }
  return { p, n, k: kOut }
}

/**
 * The closed-form Bayes error, when the model has one and its label operations allow it: any class reweightings
 * (which change the effective priors), optionally followed by one symmetric binary flip at rate $\rho \le 1/2$, which
 * maps an error $e$ to $\rho + (1 - 2\rho) e$. None after covariate shift.
 *
 * @param model The class model, with its `closedForm` when it has one.
 * @returns The Bayes error, or undefined when there is no closed form for this model and these operations.
 */
function closedFormError(model: ClassModel): number | undefined {
  if (!model.closedForm || model.shifted) return undefined
  let priors = [...model.priors]
  let flip: number | undefined
  for (const [i, op] of model.ops.entries()) {
    if (op.kind === 'weights') priors = priors.map((p, j) => p * op.weights[j])
    else {
      const m = op.matrix
      const symmetricBinary = m.length === 2 && i === model.ops.length - 1 && m[0][1] === m[1][0] && m[0][1] <= 0.5
      if (!symmetricBinary) return undefined
      flip = m[0][1]
    }
  }
  const e = model.closedForm(normalise(priors))
  return flip === undefined ? e : flip + (1 - 2 * flip) * e
}

/**
 * Build the truth model from a class model (see `ClassificationTruth`). The Bayes error is in closed form when
 * `closedFormError` allows it, else a weighted Monte Carlo mean of $1 - \max_j \Pr(\tilde y = j \mid \xvec)$ over
 * `model.reference()`, with its standard error; both, and the observed class proportions, are computed on first
 * access. `score` is the log odds of class 1, so it needs at least two classes.
 *
 * @param model The class densities, priors, label operations and reference sample.
 * @returns The truth.
 *
 * @example Two unit Gaussian classes at -1 and 1, then with 20% label noise
 * const model = {
 *   classes: 2,
 *   priors: [0.5, 0.5],
 *   logDensity: (x) => tensor(toArray(x).map(([v]) => [-((v + 1) ** 2) / 2, -((v - 1) ** 2) / 2])),
 *   ops: [],
 *   reference: () => ({ x: tensor([[0]]), weights: Float64Array.of(1) }),
 *   closedForm: (priors) => twoGaussianBayesError(2, priors),
 *   family: 'two unit Gaussians',
 * }
 * const x = tensor([[-1], [0.5], [2]])
 * const clean = classificationTruth(model)
 * print('posterior of class 1:', toArray(clean.posterior(x)).map((r) => r[1]))
 * print('Bayes rule:', clean.decide(x), ' Bayes error:', clean.bayesError, `(${clean.bayesErrorMethod})`)
 * const noisy = classificationTruth({ ...model, ops: [{ kind: 'noise', matrix: [[0.8, 0.2], [0.2, 0.8]] }] })
 * print('with 20% label noise (0.2 + 0.6 e):', noisy.bayesError)
 */
export function classificationTruth(model: ClassModel): ClassificationTruth {
  const monteCarlo = lazy(() => {
    const { x, weights } = model.reference()
    const { p, n, k } = posteriorData(model, x)
    let mean = 0
    const errs = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const e = 1 - Math.max(...p.subarray(i * k, (i + 1) * k))
      errs[i] = Number.isNaN(e) ? 0 : e
      mean += weights[i] * errs[i]
    }
    // Standard error of a self-normalised weighted mean: sqrt(Σ wᵢ² (eᵢ − ē)²).
    let v = 0
    errs.forEach((e, i) => (v += weights[i] * weights[i] * (e - mean) ** 2))
    return { error: mean, se: Math.sqrt(v) }
  })
  const exact = lazy(() => closedFormError(model))
  const prevalence = lazy(() => {
    if (!model.shifted) return normalise(applyOps([...model.priors], model.ops))
    const { x, weights } = model.reference()
    const { p, n, k } = posteriorData(model, x)
    const out = new Array<number>(k).fill(0)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < k; j++) out[j] += weights[i] * (Number.isNaN(p[i * k + j]) ? 0 : p[i * k + j])
    return normalise(out)
  })
  const posterior = (x: Tensor) => {
    const { p, n, k } = posteriorData(model, x)
    return fromData(p, [n, k])
  }
  return {
    kind: 'model',
    task: 'classification',
    name: model.family,
    classes: model.classes,
    priors: model.priors,
    get prevalence() {
      return prevalence()
    },
    logDensity: model.logDensity,
    cleanPosterior: (x) => {
      const { a, n, k } = logJoint(model, x)
      const z = flat(logsumexp(fromData(a, [n, k]), 1))
      for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) a[i * k + j] = Math.exp(a[i * k + j] - z[i])
      return fromData(a, [n, k])
    },
    posterior,
    predictive: (x) => {
      const { p, n, k } = posteriorData(model, x)
      for (let i = 0; i < n; i++) if (Number.isNaN(p[i * k])) p.fill(1 / k, i * k, (i + 1) * k)
      if (k === 2)
        return Bernoulli(
          fromData(
            Float64Array.from({ length: n }, (_, i) => p[2 * i + 1]),
            [n],
          ),
        )
      return Categorical(fromData(p, [n, k]))
    },
    decide: (x) => {
      const { p, n, k } = posteriorData(model, x)
      const out = new Int32Array(n)
      for (let i = 0; i < n; i++) {
        let best = -1
        for (let j = 0; j < k; j++)
          if (!Number.isNaN(p[i * k + j]) && (best < 0 || p[i * k + j] > p[i * k + best])) best = j
        out[i] = best
      }
      return fromData(out, [n])
    },
    expect: (x, f = (y) => y) => {
      const { p, n, k } = posteriorData(model, x)
      const fs = Array.from({ length: k }, (_, j) => f(j))
      const out = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[i] += p[i * k + j] * fs[j]
      return fromData(out, [n])
    },
    score: (x) => {
      const { a, n, k } = logJoint(model, x)
      const out = new Float64Array(n)
      if (model.ops.length === 0) {
        // log odds of class 1 against the rest: a₁ − log Σ_{j ≠ 1} exp aⱼ.
        const rest = Float64Array.from(a, (v, i) => (i % k === 1 ? -Infinity : v))
        const z = flat(logsumexp(fromData(rest, [n, k]), 1))
        for (let i = 0; i < n; i++) out[i] = a[i * k + 1] - z[i]
      } else {
        const { p, k: kOut } = posteriorData(model, x)
        for (let i = 0; i < n; i++) out[i] = Math.log(p[i * kOut + 1]) - Math.log(1 - p[i * kOut + 1])
      }
      return fromData(out, [n])
    },
    get bayesError() {
      return exact() ?? monteCarlo().error
    },
    get bayesRisk() {
      return exact() ?? monteCarlo().error
    },
    get bayesErrorMethod() {
      return exact() === undefined ? 'monte carlo' : 'closed form'
    },
    get bayesErrorSe() {
      return exact() === undefined ? monteCarlo().se : 0
    },
    model,
  }
}

// ── Class-conditional densities, from the registered distributions ──────────────────────────────────────────────────

/**
 * A float64 matrix from its rows.
 *
 * @param rows The rows, all of the same length (the first row's length is the column count).
 * @returns The matrix.
 */
const matrixOf = (rows: readonly (readonly number[])[]): Tensor =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])
/**
 * A float64 vector.
 *
 * @param v The values (copied).
 * @returns The vector.
 */
const vectorOf = (v: readonly number[]): Tensor => fromData(Float64Array.from(v), [v.length])

/**
 * The Gaussian classes $\Gauss(\muvec_j, \Sigmamat_j)$: $\log p(\xvec \mid j)$ for every row, $n \times k$, as a
 * `ClassModel`'s `logDensity`. Throws when a covariance is not positive definite.
 *
 * @param means The class means $\muvec_j$, $k$ rows of $d$ values.
 * @param covariances The class covariances $\Sigmamat_j$, $k$ matrices of $d \times d$ given as rows.
 * @returns The log density of each class at each row of an $n \times d$ matrix.
 */
export function gaussianClasses(
  means: readonly (readonly number[])[],
  covariances: readonly (readonly (readonly number[])[])[],
): (x: Tensor) => Tensor {
  const laws = means.map((m, j) => MultivariateNormal(vectorOf(m), { covariance: matrixOf(covariances[j]) }))
  return (x) => classColumns(laws.map((law) => law.logProb(x) as Tensor))
}

/**
 * The Bayes error of two Gaussian classes with a shared covariance at Mahalanobis distance $\Delta$ and priors
 * $(\pi_0, \pi_1)$. The log likelihood ratio $L$ is $\Gauss(\pm\Delta^2 / 2, \Delta^2)$ under each class and the
 * Bayes rule says 1 when $L > t = \log(\pi_0 / \pi_1)$, so the error is
 * $\pi_1 \Phi((t - \Delta^2 / 2) / \Delta) + \pi_0 \Phi((-t - \Delta^2 / 2) / \Delta)$ (e.g. Duda, Hart and Stork,
 * 2001, §2.8.3). It is 0 when a prior is 0, and $\min(\pi_0, \pi_1)$ when $\Delta = 0$.
 *
 * @param delta The Mahalanobis distance $\Delta \ge 0$ between the class means.
 * @param priors The class priors $(\pi_0, \pi_1)$, summing to one.
 * @returns The Bayes error.
 *
 * @example Equal priors give the error rate Phi(-Delta / 2)
 * print('Delta = 2, equal priors:', twoGaussianBayesError(2, [0.5, 0.5]))
 * print('Delta = 2, priors 0.9 and 0.1:', twoGaussianBayesError(2, [0.9, 0.1]))
 * print('Delta = 0:', twoGaussianBayesError(0, [0.7, 0.3]))
 */
export function twoGaussianBayesError(delta: number, priors: readonly number[]): number {
  const [p0, p1] = priors
  if (p0 === 0 || p1 === 0) return 0
  if (delta === 0) return Math.min(p0, p1)
  const t = Math.log(p0 / p1)
  return p1 * normalCdf((t - (delta * delta) / 2) / delta) + p0 * normalCdf((-t - (delta * delta) / 2) / delta)
}

/** Rows evaluated per block against the nodes of a curve density, so a block holds at most $2^{18}$ node terms. */
const BLOCK_TERMS = 1 << 18

/**
 * The log density of points uniform along a curve $\cvec(u)$, $u$ uniform on $[0, 1]$, blurred by isotropic Gaussian
 * noise of standard deviation $\sigma$ (`sd`): $\log \frac{1}{M} \sum_m \Gauss(\xvec; \cvec(u_m), \sigma^2 \Imat)$, a
 * midpoint rule over $M$ nodes, evaluated as a batch of bivariate normals. $M$ is chosen so that nodes are at most
 * $\sigma / 2$ apart along the curve (from its length measured on 256 chords), between 64 and 2000, which keeps the
 * rule accurate to well under a percent while $M$ is below its cap.
 *
 * @param curve The curve $\cvec(u)$ in the plane, for $u \in [0, 1]$.
 * @param sd The noise standard deviation $\sigma$.
 * @returns The log density of every row of an $n \times 2$ matrix of points ($n$ values).
 */
export function curveLogDensity(curve: (u: number) => [number, number], sd: number): (x: Tensor) => Tensor {
  let length = 0
  let prev = curve(0)
  for (let i = 1; i <= 256; i++) {
    const p = curve(i / 256)
    length += Math.hypot(p[0] - prev[0], p[1] - prev[1])
    prev = p
  }
  const m = Math.min(2000, Math.max(64, Math.ceil((2 * length) / sd)))
  const nodes = new Float64Array(2 * m)
  for (let i = 0; i < m; i++) [nodes[2 * i], nodes[2 * i + 1]] = curve((i + 0.5) / m)
  const law = MultivariateNormal(fromData(nodes, [m, 2]), {
    covariance: fromData(Float64Array.of(sd * sd, 0, 0, sd * sd), [2, 2]),
  })
  const logM = Math.log(m)
  return (x) => {
    const { data, n } = points(x)
    const out = new Float64Array(n)
    const block = Math.max(1, Math.floor(BLOCK_TERMS / m))
    for (let start = 0; start < n; start += block) {
      const rows = Math.min(block, n - start)
      const terms = law.logProb(fromData(data.slice(2 * start, 2 * (start + rows)), [rows, 1, 2])) as Tensor
      const z = flat(logsumexp(reshape(terms, [rows, m]), 1))
      for (let i = 0; i < rows; i++) out[start + i] = z[i] - logM
    }
    return fromData(out, [n])
  }
}

/**
 * A reference sample with equal weights, for points drawn from the population itself.
 *
 * @param x The $m$ points, an $m \times d$ matrix.
 * @returns The points with weights $1 / m$.
 */
export function equalReference(x: Tensor): Reference {
  const n = x.shape[0]
  return { x, weights: new Float64Array(n).fill(1 / n) }
}

// ── Regression ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The 32-point probabilists' Gauss–Hermite rule, its weights normalised to sum to one, so that
 * $\sum_q w_q f(z_q) \approx \expect[f(Z)]$ for $Z \sim \Gauss(0, 1)$. Made on first use.
 */
const HERMITE = lazy(() => {
  const rule = gaussHermite(32, { probabilists: true })
  const w = flat(rule.weights)
  const total = w.reduce((a, b) => a + b, 0)
  return { nodes: flat(rule.nodes), weights: w.map((v) => v / total) }
})

/**
 * Build the truth model from a regression model (see `RegressionTruth`); its name is the model's `family`.
 *
 * @param model The regression function, the noise level, the Bayes risk and the outlier fraction.
 * @returns The truth.
 */
export function regressionTruthOf(model: RegressionModel): RegressionTruth {
  const perRow = (x: Tensor, f: (row: Float64Array) => number) => {
    const { data, n, d } = points(x)
    return fromData(
      Float64Array.from({ length: n }, (_, i) => f(data.subarray(i * d, (i + 1) * d))),
      [n],
    )
  }
  const mean = (x: Tensor) => perRow(x, model.mean)
  return {
    kind: 'model',
    task: 'regression',
    name: model.family,
    mean,
    noiseSdAt: (x) => perRow(x, model.sdAt),
    predictive: (x) => Normal(mean(x), perRow(x, model.sdAt)),
    decide: mean,
    expect: (x, f) => {
      if (!f) return mean(x)
      const { nodes, weights } = HERMITE()
      return perRow(x, (row) => {
        const [m, s] = [model.mean(row), model.sdAt(row)]
        return nodes.reduce((acc, z, q) => acc + weights[q] * f(m + s * z), 0)
      })
    },
    noiseSd: model.noiseSd,
    bayesRisk: model.bayesRisk,
    outlierFraction: model.outlierFraction,
    model,
  }
}

/**
 * A regression truth $y = m(\xvec) + \varepsilon$ with homoscedastic noise unless `sdAt` is given, and no outliers.
 *
 * @param mean The regression function $m(\xvec)$ at one point.
 * @param noiseSd The noise standard deviation (with `sdAt`, the value reported as `noiseSd`).
 * @param options The noise level as a function, the Bayes risk and a name for captions.
 * @param options.sdAt The noise standard deviation $\sigma(\xvec)$ at one point. Default `noiseSd` everywhere.
 * @param options.bayesRisk The Bayes risk $\expect[\sigma(\xvec)^2]$. Default $\sigma^2$ = `noiseSd` squared, which is
 *   only right without `sdAt`.
 * @param options.family What the regression function is, for captions. Default `'regression function'`.
 * @returns The truth.
 *
 * @example A line with noise of standard deviation 0.5
 * const truth = regressionTruth((x) => 2 * x[0], 0.5)
 * const x = tensor([[0], [1]])
 * print('mean:', truth.mean(x), ' Bayes risk:', truth.bayesRisk)
 * print('E[y^2 | x] (m^2 + sigma^2):', truth.expect(x, (y) => y * y))
 */
export function regressionTruth(
  mean: (x: Row) => number,
  noiseSd: number,
  options: { sdAt?: (x: Row) => number; bayesRisk?: number; family?: string } = {},
): RegressionTruth {
  return regressionTruthOf({
    mean,
    sdAt: options.sdAt ?? (() => noiseSd),
    noiseSd,
    bayesRisk: options.bayesRisk ?? noiseSd * noiseSd,
    outlierFraction: 0,
    family: options.family ?? 'regression function',
  })
}

/**
 * A regression truth with some parts of its model replaced, as a modifier changes the noise or adds outliers.
 *
 * @param t The truth to start from (not modified).
 * @param edit The parts of its `model` to replace.
 * @returns A new truth built from the edited model.
 */
export function remodelRegression(t: RegressionTruth, edit: Partial<RegressionModel>): RegressionTruth {
  return regressionTruthOf({ ...t.model, ...edit })
}

/**
 * The truth of a piecewise series from its segments, which must be contiguous, start at 0 and be non-empty; they cover
 * $0, \dots, n - 1$ with $n$ the last segment's end. Throws `DomainError` when a segment does not continue the
 * series. The Bayes risk is the segments' `risk` weighted by their lengths.
 *
 * @param name What the series is, for captions.
 * @param family What changes between segments; `'poisson'` makes the predictive law Poisson.
 * @param segments The segments in order.
 * @returns The truth.
 *
 * @example A shift in mean at index 5
 * const segment = (start, end, mean) => ({ start, end, params: { mean, sd: 1 }, mean, variance: 1, risk: 1 })
 * const truth = changepointTruth('mean shift', 'mean', [segment(0, 5, 0), segment(5, 10, 3)])
 * print('changepoints:', truth.changepoints, ' length:', truth.n)
 * print('segment at t = 0, 4, 5, 9:', truth.decide(tensor([0, 4, 5, 9])))
 * print('mean there:', truth.expect(tensor([0, 4, 5, 9])))
 */
export function changepointTruth(
  name: string,
  family: ChangepointFamily,
  segments: readonly Segment[],
): ChangepointTruth {
  const n = segments.length ? segments[segments.length - 1].end : 0
  segments.forEach((g, i) => {
    if (g.start !== (i === 0 ? 0 : segments[i - 1].end) || g.end <= g.start)
      throw new DomainError(
        'changepointTruth',
        `changepointTruth: segment ${i} (${g.start}–${g.end}) does not continue the series`,
      )
  })
  const segmentAt = (t: number): Size => {
    const i = Math.max(0, Math.min(n - 1, Math.round(t)))
    let lo = 0
    let hi = segments.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (segments[mid].start <= i) lo = mid
      else hi = mid - 1
    }
    return lo
  }
  const at = (t: Tensor) => Array.from(t.data as ArrayLike<number>, (v) => segments[segmentAt(v)])
  const flat = (t: Tensor) => fromData(Float64Array.from(t.data as ArrayLike<number>))
  return {
    kind: 'model',
    task: 'changepoint',
    name,
    family,
    n,
    changepoints: segments.slice(1).map((g) => g.start),
    segments,
    segmentAt,
    decide: (t) => fromData(Int32Array.from(t.data as ArrayLike<number>, segmentAt)),
    predictive: (t) => {
      const g = at(flat(t))
      if (family === 'poisson') return Poisson(fromData(Float64Array.from(g, (s) => s.mean)))
      return Normal(
        fromData(Float64Array.from(g, (s) => s.mean)),
        fromData(Float64Array.from(g, (s) => Math.sqrt(s.variance))),
      )
    },
    expect: (t, f) => {
      const g = at(flat(t))
      if (!f) return fromData(Float64Array.from(g, (s) => s.mean))
      // E[f(y)] by Gauss–Hermite quadrature under the normal law (counts: the Poisson sum up to a far tail).
      const { nodes, weights } = HERMITE()
      return fromData(
        Float64Array.from(g, (s) => {
          if (family === 'poisson') {
            let acc = 0
            let p = Math.exp(-s.mean)
            const top = Math.ceil(s.mean + 12 * Math.sqrt(s.mean) + 20)
            for (let k = 0; k <= top; k++) {
              acc += p * f(k)
              p *= s.mean / (k + 1)
            }
            return acc
          }
          const sd = Math.sqrt(s.variance)
          return nodes.reduce((acc, z, q) => acc + weights[q] * f(s.mean + sd * z), 0)
        }),
      )
    },
    bayesRisk: n ? segments.reduce((acc, g) => acc + g.risk * (g.end - g.start), 0) / n : 0,
  }
}

// ── Regimes: a gate over x chooses which of K functions generated y ─────────────────────────────────────────────────

/**
 * The parts of a `RegimeTruth`: $K$ regimes, a gate $\Pr(\text{regime } k \mid \xvec)$, and each regime's function of
 * $\xvec$ (the mean of $y$ for regression, the log-odds of class 1 for classification), on a box of inputs.
 */
export interface RegimeModel {
  /** What the data are, for captions. */
  name: string
  /** Whether $y$ is a real target or a binary class. */
  task: 'regression' | 'classification'
  /** The number of regimes $K$. */
  regimes: Size
  /** $\Pr(\text{regime } k \mid \xvec)$ at one point, length $K$. */
  gate: (x: Row) => number[]
  /** Regime $k$'s function at one point: the mean of $y$ (regression) or the log-odds of $y = 1$ (classification). */
  fn: (x: Row, k: number) => number
  /** The noise standard deviation of $y$ around a regime's mean (regression; 0 for classification). */
  noiseSd: number
  /** The lower corner of the input box the population's $\xvec$ is uniform on. */
  lower: readonly number[]
  /** The upper corner of that box. */
  upper: readonly number[]
  /** Each regime's function as text, for captions. */
  formulas: readonly string[]
}

/**
 * The truth of data in which a gate over $\xvec$ picks one of $K$ regimes and the regime's function generates $y$: a
 * mixture of regressions or classifiers whose weights depend on $\xvec$ (the generative model of a mixture of experts;
 * Jacobs, Jordan, Nowlan and Hinton, 1991). When the gate is hard, the regimes partition the input space and the truth
 * is a piecewise function.
 */
export interface RegimeTruth extends TruthContract {
  /** Whether $y$ is a real target or a binary class. */
  readonly task: 'regression' | 'classification'
  /** What the data are, for captions. */
  readonly name: string
  /** The number of regimes $K$. */
  readonly regimes: Size
  /** $\Pr(\text{regime } k \mid \xvec)$ per row ($n \times K$). */
  gate(x: Tensor): Tensor
  /** The most probable regime per row (int32, $n$ values). */
  regime(x: Tensor): Tensor
  /**
   * Each regime's prediction per row ($n \times K$): its mean of $y$ (regression) or
   * $\Pr(y = 1 \mid \xvec, \text{regime})$ (classification).
   */
  regimeMean(x: Tensor): Tensor
  /** $\expect[y \mid \xvec]$ per row ($n$ values): the gate-weighted regime means. */
  mean(x: Tensor): Tensor
  /** $\log p(y \mid \xvec)$ per row ($n$ values): the log of the gate-weighted mixture of the regimes' laws. */
  logLikelihood(x: Tensor, y: Tensor): Tensor
  /**
   * Regression: the normal with $y$'s conditional mean and variance (exact where the gate is hard, moment-matched where
   * regimes overlap; `logLikelihood` is exact everywhere). Classification: the Bernoulli of class 1.
   */
  predictive(x: Tensor): Distribution
  /** The mean (regression) or the Bayes class (classification, int32). */
  decide(x: Tensor): Tensor
  /**
   * $\expect[f(y) \mid \xvec]$ per row: the mean without `f`; with `f`, the gate-weighted Gauss–Hermite expectations
   * of the regimes (regression) or $(1 - p) f(0) + p f(1)$ (classification).
   */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /**
   * $\expect[\var(y \mid \xvec)]$ (regression) or the Bayes error $\expect[\min(p, 1 - p)]$ (classification), over a
   * grid of the box (computed on first access).
   */
  readonly bayesRisk: number
  /** The model the truth is built from. */
  readonly model: RegimeModel
}

/**
 * The logistic function $1 / (1 + e^{-v})$.
 *
 * @param v A log-odds.
 * @returns The probability.
 */
const sigmoidOf = (v: number) => 1 / (1 + Math.exp(-v))

/**
 * Build a regime truth (see `RegimeTruth`). The Bayes risk is averaged over a regular midpoint grid of the input box:
 * 2000 points in one dimension, about 10000 in more (100 by 100 in two).
 *
 * @param model The regimes, the gate, each regime's function, the noise and the input box.
 * @returns The truth.
 *
 * @example A hard gate between two constant regimes
 * const truth = regimeTruth({
 *   name: 'step',
 *   task: 'regression',
 *   regimes: 2,
 *   gate: (x) => (x[0] < 0.5 ? [1, 0] : [0, 1]),
 *   fn: (x, k) => (k === 0 ? 0 : 1),
 *   noiseSd: 0.1,
 *   lower: [0],
 *   upper: [1],
 *   formulas: ['0', '1'],
 * })
 * const x = tensor([[0.2], [0.8]])
 * print('regime:', truth.regime(x), ' mean:', truth.mean(x))
 * print('Bayes risk (the noise variance):', truth.bayesRisk)
 */
export function regimeTruth(model: RegimeModel): RegimeTruth {
  const K = model.regimes
  const perRow = (x: Tensor, f: (row: Float64Array) => number[]) => {
    const { data, n, d } = points(x)
    const out: number[][] = []
    for (let i = 0; i < n; i++) out.push(f(data.subarray(i * d, (i + 1) * d)))
    return out
  }
  const regimeValue = (row: Row, k: number) =>
    model.task === 'regression' ? model.fn(row, k) : sigmoidOf(model.fn(row, k))
  const meanAt = (row: Row) => {
    const g = model.gate(row)
    return g.reduce((acc, w, k) => acc + w * regimeValue(row, k), 0)
  }
  const varianceAt = (row: Row) => {
    const g = model.gate(row)
    const m = meanAt(row)
    if (model.task === 'classification') return m * (1 - m)
    // Law of total variance: σ² + Σ gₖ (μₖ − μ)².
    return model.noiseSd ** 2 + g.reduce((acc, w, k) => acc + w * (model.fn(row, k) - m) ** 2, 0)
  }
  const mean = (x: Tensor) => fromData(Float64Array.from(perRow(x, (r) => [meanAt(r)]).flat()), [x.shape[0]])
  const risk = lazy(() => {
    // A regular grid over the box: 2000 points in 1-D, 100 × 100 in 2-D.
    const d = model.lower.length
    const per = d === 1 ? 2000 : Math.max(2, Math.round(Math.pow(10000, 1 / d)))
    const total = per ** d
    let acc = 0
    const row = new Float64Array(d)
    for (let i = 0; i < total; i++) {
      let rest = i
      for (let c = 0; c < d; c++) {
        const j = rest % per
        rest = Math.floor(rest / per)
        row[c] = model.lower[c] + ((j + 0.5) * (model.upper[c] - model.lower[c])) / per
      }
      if (model.task === 'regression') acc += varianceAt(row)
      else {
        const p = meanAt(row)
        acc += Math.min(p, 1 - p)
      }
    }
    return acc / total
  })
  const truth: RegimeTruth = {
    kind: 'model',
    task: model.task,
    name: model.name,
    regimes: K,
    gate: (x) => fromData(Float64Array.from(perRow(x, (r) => model.gate(r)).flat()), [x.shape[0], K]),
    regime: (x) =>
      fromData(
        Int32Array.from(
          perRow(x, (r) => {
            const g = model.gate(r)
            return [g.indexOf(Math.max(...g))]
          }).flat(),
        ),
        [x.shape[0]],
      ),
    regimeMean: (x) =>
      fromData(Float64Array.from(perRow(x, (r) => Array.from({ length: K }, (_, k) => regimeValue(r, k))).flat()), [
        x.shape[0],
        K,
      ]),
    mean,
    logLikelihood: (x, y) => {
      const ys = toFlatArray(y)
      let i = 0
      const out = perRow(x, (r) => {
        const g = model.gate(r)
        const v = ys[i++]
        let p = 0
        for (let k = 0; k < K; k++) {
          if (g[k] === 0) continue
          if (model.task === 'regression') {
            const s = model.noiseSd
            p += (g[k] * Math.exp(-0.5 * ((v - model.fn(r, k)) / s) ** 2)) / (s * Math.sqrt(2 * Math.PI))
          } else {
            const q = sigmoidOf(model.fn(r, k))
            p += g[k] * (v === 1 ? q : 1 - q)
          }
        }
        return [Math.log(p)]
      })
      return fromData(Float64Array.from(out.flat()), [x.shape[0]])
    },
    predictive: (x) => {
      const m = mean(x)
      if (model.task === 'classification') return Bernoulli(m)
      return Normal(m, fromData(Float64Array.from(perRow(x, (r) => [Math.sqrt(varianceAt(r))]).flat()), [x.shape[0]]))
    },
    decide: (x) => {
      const m = mean(x)
      if (model.task === 'regression') return m
      return fromData(
        Int32Array.from(toFlatArray(m), (p) => (p > 0.5 ? 1 : 0)),
        [x.shape[0]],
      )
    },
    expect: (x, f) => {
      if (!f) return mean(x)
      if (model.task === 'classification')
        return fromData(Float64Array.from(perRow(x, (r) => [(1 - meanAt(r)) * f(0) + meanAt(r) * f(1)]).flat()), [
          x.shape[0],
        ])
      const { nodes, weights } = HERMITE()
      return fromData(
        Float64Array.from(
          perRow(x, (r) => {
            const g = model.gate(r)
            let acc = 0
            for (let k = 0; k < K; k++)
              if (g[k] > 0) {
                const m = model.fn(r, k)
                acc += g[k] * nodes.reduce((a, z, q) => a + weights[q] * f(m + model.noiseSd * z), 0)
              }
            return [acc]
          }).flat(),
        ),
        [x.shape[0]],
      )
    },
    get bayesRisk() {
      return risk()
    },
    model,
  }
  return truth
}

// ── Inverse problems: y given x where x = f(y) + ε and f is many-to-one ─────────────────────────────────────────────

/**
 * One exact solution of an inverse problem: a target $\yvec$ with $f(\yvec) = \xvec$, and its share of
 * $p(\yvec \mid \xvec)$.
 */
export interface InverseSolution {
  /** The solution $\yvec^*$, $D$ values. */
  readonly value: number[]
  /**
   * The solution's probability in the small-noise limit, $p(\yvec^*) / \lvert \det f'(\yvec^*) \rvert$ normalised over
   * the solutions.
   */
  readonly weight: number
}

/**
 * The parts of an `InverseTruth`: targets $\yvec$ drawn from a uniform prior on a box, inputs
 * $\xvec = f(\yvec) + \varepsilonvec$ with $\varepsilonvec \sim \Gauss(\zeros, \sigma^2 \Imat)$, and the inverse of
 * $f$, which is multi-valued.
 */
export interface InverseModel {
  /** What the problem is, for captions. */
  name: string
  /** The dimension $D$ of the target $\yvec$. */
  outputs: Size
  /** The forward map $f(\yvec)$, noise-free. */
  forward: (y: Row) => number[]
  /**
   * Every $\yvec$ in the prior's box with $f(\yvec) = \xvec$, with its weight (empty where $\xvec$ is outside $f$'s
   * image).
   */
  solutions: (x: Row) => InverseSolution[]
  /**
   * The law of $\yvec$ given $\xvec$ as weighted atoms: a quadrature rule of the exact posterior, or the solutions. The
   * weights need not be normalised; all zero (or none) where $\xvec$ has no posterior.
   */
  atoms: (x: Row) => { values: number[][]; weights: number[] }
  /** $\log p(\yvec \mid \xvec)$, where the posterior has a closed form up to quadrature (one-dimensional targets). */
  logLikelihood?: (x: Row, y: Row) => number
  /** The noise standard deviation $\sigma$ of the inputs. */
  noise: number
  /** The lower corner of the prior's box of targets. */
  lower: readonly number[]
  /** The upper corner of that box. */
  upper: readonly number[]
  /** $f$ as text, for captions. */
  formula: string
}

/**
 * The truth of an inverse problem (Bishop, 1994, "Mixture density networks"): $\yvec$ is drawn uniformly on a box and
 * observed through $\xvec = f(\yvec) + \varepsilonvec$, and the task is to predict $\yvec$ from $\xvec$. Where $f$
 * folds over, $\yvec$ given $\xvec$ has a mode at every solution of $f(\yvec) = \xvec$, and the conditional mean
 * $\expect[\yvec \mid \xvec]$, the minimiser of the squared error, can fall between them on no solution at all.
 */
export interface InverseTruth extends TruthContract {
  /** The task, always regression. */
  readonly task: 'regression'
  /** What the problem is, for captions. */
  readonly name: string
  /** The dimension $D$ of $\yvec$. */
  readonly outputs: Size
  /** The noise-free solutions of $f(\yvec) = \xvec$ at one input, most probable first. */
  solutions(x: Row): InverseSolution[]
  /** $f(\yvec)$ at one target. */
  forward(y: Row): number[]
  /**
   * $\expect[\yvec \mid \xvec]$ per row: $n$ values for $D = 1$, else $n \times D$. NaN where $\xvec$ has no
   * posterior.
   */
  mean(x: Tensor): Tensor
  /** $\var(y_j \mid \xvec)$ per row, the shape of `mean`. */
  variance(x: Tensor): Tensor
  /**
   * $\log p(\yvec \mid \xvec)$ per row ($n$ values); NaN where the model has no density (a law concentrated on the
   * solutions).
   */
  logLikelihood(x: Tensor, y: Tensor): Tensor
  /** The normal with $\yvec$'s conditional mean and variance per row (moment-matched: the true law is multimodal). */
  predictive(x: Tensor): AnyUnivariate
  /** $\expect[\yvec \mid \xvec]$, the Bayes decision under squared loss. */
  decide(x: Tensor): Tensor
  /**
   * $\expect[f(y) \mid \xvec]$ ($n$ values) over the atoms; with `f`, $D = 1$ only (else `DomainError`). Without `f`,
   * the mean.
   */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /**
   * $\expect[\sum_j \var(y_j \mid \xvec)]$ over the prior (on noise-free inputs, a 2000-point grid in one dimension,
   * $60 \times 60$ in two), computed on first access.
   */
  readonly bayesRisk: number
  /** The model the truth is built from. */
  readonly model: InverseModel
}

/**
 * Build an inverse-problem truth (see `InverseTruth`): the moments come from the model's atoms, and the Bayes risk
 * skips grid points whose posterior is undefined.
 *
 * @param model The forward map, its solutions and atoms, the noise and the prior's box.
 * @returns The truth.
 *
 * @example y squared: the mean falls between the two solutions
 * // y = ±√x, equally likely, for x in [0, 1]; nothing outside f's image.
 * const roots = (x) => (x >= 0 && x <= 1 ? [Math.sqrt(x), -Math.sqrt(x)] : [])
 * const truth = inverseTruth({
 *   name: 'square',
 *   outputs: 1,
 *   forward: ([y]) => [y * y],
 *   solutions: ([x]) => roots(x).map((r) => ({ value: [r], weight: 0.5 })),
 *   atoms: ([x]) => ({ values: roots(x).map((r) => [r]), weights: roots(x).map(() => 0.5) }),
 *   noise: 0.01,
 *   lower: [-1],
 *   upper: [1],
 *   formula: 'y^2',
 * })
 * print('solutions at x = 0.25:', truth.solutions([0.25]).map((s) => s.value[0]))
 * print('mean and variance there:', truth.mean(tensor([[0.25]])), truth.variance(tensor([[0.25]])))
 * print('Bayes risk (E[y^2] = 1/3):', truth.bayesRisk)
 */
export function inverseTruth(model: InverseModel): InverseTruth {
  const D = model.outputs
  const moments = (r: Row) => {
    const { values, weights } = model.atoms(r)
    const total = weights.reduce((a, w) => a + w, 0)
    const m = new Array<number>(D).fill(0)
    const v = new Array<number>(D).fill(0)
    if (!(total > 0)) return { m: m.fill(NaN), v: v.fill(NaN) }
    values.forEach((y, i) => y.forEach((yj, j) => (m[j] += (weights[i] / total) * yj)))
    values.forEach((y, i) => y.forEach((yj, j) => (v[j] += (weights[i] / total) * (yj - m[j]) ** 2)))
    return { m, v }
  }
  const perRow = (x: Tensor, f: (r: Float64Array) => number[]) => {
    const { data, n, d } = points(x)
    const out = new Float64Array(n * D)
    for (let i = 0; i < n; i++) out.set(f(data.subarray(i * d, (i + 1) * d)), i * D)
    return fromData(out, D === 1 ? [n] : [n, D])
  }
  const mean = (x: Tensor) => perRow(x, (r) => moments(r).m)
  const variance = (x: Tensor) => perRow(x, (r) => moments(r).v)
  const risk = lazy(() => {
    const per = D === 1 ? 2000 : 60
    const total = per ** D
    const y = new Array<number>(D)
    let acc = 0
    let count = 0
    for (let i = 0; i < total; i++) {
      let rest = i
      for (let j = 0; j < D; j++) {
        y[j] = model.lower[j] + (((rest % per) + 0.5) * (model.upper[j] - model.lower[j])) / per
        rest = Math.floor(rest / per)
      }
      const { v } = moments(model.forward(y))
      if (v.every(Number.isFinite)) {
        acc += v.reduce((a, b) => a + b, 0)
        count++
      }
    }
    return acc / Math.max(1, count)
  })
  return {
    kind: 'model',
    task: 'regression',
    name: model.name,
    outputs: D,
    solutions: (x) => model.solutions(x),
    forward: (y) => model.forward(y),
    mean,
    variance,
    logLikelihood: (x, y) => {
      const { data, n, d } = points(x)
      const ys = toFlatArray(y)
      return fromData(
        Float64Array.from({ length: n }, (_, i) =>
          model.logLikelihood
            ? model.logLikelihood(data.subarray(i * d, (i + 1) * d), ys.subarray(i * D, (i + 1) * D))
            : NaN,
        ),
        [n],
      )
    },
    predictive: (x) => {
      const v = variance(x)
      return Normal(mean(x), fromData(Float64Array.from(toFlatArray(v), Math.sqrt), v.shape))
    },
    decide: mean,
    expect: (x, f) => {
      if (!f) return mean(x)
      if (D !== 1) throw new DomainError('inverseTruth', 'inverseTruth: expect with f needs a 1-d target')
      return perRow(x, (r) => {
        const { values, weights } = model.atoms(r)
        const total = weights.reduce((a, w) => a + w, 0)
        return [values.reduce((a, v, i) => a + (weights[i] / total) * f(v[0]), 0)]
      })
    },
    get bayesRisk() {
      return risk()
    },
    model,
  }
}
