/**
 * Known truth for synthetic datasets. When the generating process has a closed form, a dataset carries it in
 * `meta.truth` as a model (`kind: 'model'`) with the contract's capabilities: the Bayes rule (`decide`), the Bayes
 * posterior or the conditional law of y (`predictive`), the regression function (`expect`), and the lowest risk any
 * predictor can reach (`bayesRisk`). A figure draws the Bayes-optimal boundary and curves beside a fitted model's by
 * calling the same methods on both.
 *
 * A classification truth is built from class-conditional densities p(x | y = j), evaluated with the registered
 * distributions of `aifn-compute/probability/distributions`, and clean class priors πⱼ, followed by a list of label
 * operations: noise matrices T (T[i][j] = P(observed j | label i)) and class reweightings w (from resampling by
 * label). The joint of x and the observed label is then
 *
 *   p(x, ỹ) ∝ ((π ∘ p(x)) T₁ ∘ w₁ …)_ỹ,
 *
 * applied left to right, and the Bayes posterior is that vector normalised (Duda, Hart and Stork, 2001, "Pattern
 * Classification", §2.2). Modifiers compose by editing this model. Every method takes a batch of points, an [n, d]
 * float64 matrix.
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

const toFlatArray = (t: Tensor) => Float64Array.from(toFlat(t))

/** A point, one number per feature (used inside regression functions). */
export type Row = ArrayLike<number>

/** A label operation applied after the clean labels are drawn. */
export type LabelOp =
  | { kind: 'noise'; matrix: number[][] }
  | {
      kind: 'weights'
      /** Relative weight of each observed class (resampling by label). */
      weights: number[]
    }

/**
 * Points drawn from the population's marginal of x ([m, d]) with importance weights summing to one, for Monte Carlo
 * estimates (the Bayes error when no closed form applies).
 */
export interface Reference {
  x: Tensor
  weights: Float64Array
}

/** The parts a classification truth is built from; modifiers edit these. */
export interface ClassModel {
  classes: Size
  /** Clean class proportions πⱼ in the population. */
  priors: number[]
  /**
   * log p(x | clean y = j) for every row of x ([n, d]) and class j: an [n, k] matrix, up to a term per row shared by
   * all classes. −∞ outside a class's support.
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
 * The Bayes-optimal classifier of a synthetic classification problem, as a model. Methods take points ([n, d]) and
 * refer to the observed labels `y` of the dataset, after any label noise or resampling.
 */
export interface ClassificationTruth extends TruthContract, Scores<Tensor> {
  readonly task: 'classification'
  /** What the class densities are, e.g. "two moons". */
  readonly name: string
  readonly classes: Size
  /** Clean class proportions in the population. */
  readonly priors: readonly number[]
  /** Observed class proportions in the population. */
  readonly prevalence: readonly number[]
  /** log p(x | clean y = j) per row and class ([n, k]), up to a term shared by all classes. */
  logDensity(x: Tensor): Tensor
  /** P(clean y = j | x) per row and class ([n, k]). */
  cleanPosterior(x: Tensor): Tensor
  /** P(observed y = j | x) per row and class ([n, k]): the Bayes posterior. NaN where x has zero density under every class. */
  posterior(x: Tensor): Tensor
  /**
   * The Bayes posterior as a distribution over the batch: a Bernoulli over class 1 for two classes, else a
   * Categorical. Rows with zero density under every class get the uniform law.
   */
  predictive(x: Tensor): AnyUnivariate
  /** The Bayes rule: the most probable observed class per row (int32 [n]); −1 where the posterior is undefined. */
  decide(x: Tensor): Tensor
  /** E[f(ỹ) | x] under the posterior ([n]); the mean class index without `f` (P(ỹ = 1 | x) for two classes). */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** The log odds of class 1 against the rest ([n]): a Bayes-optimal score for ROC and precision–recall curves. */
  score(x: Tensor): Tensor
  /** The Bayes error E[1 − maxⱼ P(ỹ = j | x)], the lowest error any classifier can reach (computed on first access). */
  readonly bayesError: number
  /** How `bayesError` is computed: exactly, or by Monte Carlo over a reference sample of the population. */
  readonly bayesErrorMethod: 'closed form' | 'monte carlo'
  /** Standard error of a Monte Carlo `bayesError` (0 in closed form). */
  readonly bayesErrorSe: number
  /** The model the truth is built from (for modifiers). */
  readonly model: ClassModel
}

/** The parts a regression truth is built from: y = m(x) + ε, ε ~ N(0, σ(x)²). Modifiers edit these. */
export interface RegressionModel {
  /** The regression function m(x) = E[y | x] at one point. */
  mean: (x: Row) => number
  /** σ(x), the noise standard deviation at one point. */
  sdAt: (x: Row) => number
  /** The noise standard deviation (at the left end of the input range when it varies). */
  noiseSd: number
  /** The Bayes risk under squared loss, E[σ(x)²]. */
  bayesRisk: number
  /** Fraction of targets replaced by gross outliers (see `withOutliers`), 0 by default. */
  outlierFraction: number
  /** What the regression function is, for captions. */
  family: string
}

/** The truth of a synthetic regression problem y = m(x) + ε, ε ~ N(0, σ(x)²), as a model. */
export interface RegressionTruth extends TruthContract {
  readonly task: 'regression'
  readonly name: string
  /** The regression function m(x) per row ([n]); the same as `expect`. */
  mean(x: Tensor): Tensor
  /** σ(x) per row ([n]). */
  noiseSdAt(x: Tensor): Tensor
  /** The conditional law of y: Normal(m(x), σ(x)) over the batch. Outliers (if any) are not part of it. */
  predictive(x: Tensor): AnyUnivariate
  /** The point prediction that minimises squared loss: m(x) ([n]). */
  decide(x: Tensor): Tensor
  /** E[f(y) | x] ([n]); m(x) without `f`, else by 32-point Gauss–Hermite quadrature. */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** The noise standard deviation (at the left end of the input range when it varies). */
  readonly noiseSd: number
  /** Fraction of targets replaced by gross outliers. */
  readonly outlierFraction: number
  readonly model: RegressionModel
}

/** One segment of a piecewise series: indices start … end − 1, its parameters and the law of a value inside it. */
export interface Segment {
  /** First index of the segment. */
  readonly start: Size
  /** One past its last index. */
  readonly end: Size
  /** The generating parameters, e.g. `{ mean, sd }`, `{ rate }` or `{ coefficients, constant, sd }`. */
  readonly params: Readonly<Record<string, number | readonly number[]>>
  /** Mean and variance of a value in the segment (the stationary ones for an autoregression). */
  readonly mean: number
  readonly variance: number
  /** Mean squared error of predicting a value from the true parameters (the innovation variance for an autoregression). */
  readonly risk: number
}

/** What changes between segments. */
export type ChangepointFamily = 'mean' | 'variance' | 'poisson' | 'autoregressive'

/**
 * The truth of a piecewise series: its segments and changepoints, as a model whose inputs are times t (float64 [m],
 * integer indices). `decide` gives the segment index (int32), `predictive` the law of the value at t (normal, or
 * Poisson for counts; the stationary marginal for an autoregression), `expect` its mean.
 */
export interface ChangepointTruth extends TruthContract {
  readonly task: 'changepoint'
  readonly name: string
  readonly family: ChangepointFamily
  /** Length of the series. */
  readonly n: Size
  /** Indices where a new segment begins (0 excluded), ascending. */
  readonly changepoints: readonly Size[]
  readonly segments: readonly Segment[]
  /** The segment index at time t. */
  segmentAt(t: number): Size
  decide(t: Tensor): Tensor
  predictive(t: Tensor): AnyUnivariate
  expect(t: Tensor, f?: (y: number) => number): Tensor
}

/**
 * The truth of a generalised additive model g(E[y | x]) = α + Σⱼ fⱼ(xⱼ) with y from an exponential-dispersion family
 * (`additiveData`): the true partial effects fⱼ on the link scale, each centred to mean zero over xⱼ ~ U(0, 1), and the
 * conditional law of y.
 */
export interface AdditiveTruth extends TruthContract {
  readonly task: 'regression'
  readonly name: string
  readonly family: FamilyName
  readonly link: LinkName
  /** α: the mean of η over the population. */
  readonly intercept: number
  /** The dispersion φ (Gaussian: σ²; gamma: the squared coefficient of variation; 1 otherwise). */
  readonly dispersion: number
  /** Each feature's shape name. */
  readonly shapes: readonly string[]
  /**
   * The true partial effect of feature j on a grid [m], centred to mean zero over U(0, 1), or over the values
   * `centreOn` [n] when given (as a fitted GAM centres its smooths on the training data).
   */
  partial(j: Size, grid: Tensor, centreOn?: Tensor): Tensor
  /** η(x) = α + Σⱼ fⱼ(xⱼ), [n]. */
  linearPredictor(x: Tensor): Tensor
  /** μ(x) = g⁻¹(η(x)), [n]. */
  mean(x: Tensor): Tensor
  /** The family's law of y at μ(x) and φ. */
  predictive(x: Tensor): Distribution
  decide(x: Tensor): Tensor
  /** μ(x) without `f`; with `f`, Gaussian only (Gauss–Hermite). */
  expect(x: Tensor, f?: (y: number) => number): Tensor
}

/** The parts of an additive truth. */
export interface AdditiveModel {
  family: FamilyName
  link: LinkName
  intercept: number
  dispersion: number
  /** Each feature's effect on [0, 1], already centred over U(0, 1), with its name. */
  effects: readonly { name: string; f: (x: number) => number }[]
}

/** Build an additive truth (see `AdditiveTruth`). */
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
 * The truth of a one-dimensional smooth regression (`curve1d`) with x ~ U(0, 1): the mean μ(x) = g⁻¹(η(x)) and the
 * whole law of y, so its expectiles e_τ(x) are known. A location–scale law y = μ(x) + σ(x)Z with standardised noise Z
 * (normal, or a skewed shifted log-normal) has e_τ(x) = μ(x) + σ(x)e_τ(Z); a gamma law y = μ(x)G has e_τ(x) =
 * μ(x)e_τ(G); Poisson and Bernoulli expectiles come from their masses at each x. Expectiles are computed numerically
 * from the law (`expectiles` on its atoms: 4000 equally weighted quantiles, or the masses), not in closed form.
 */
export interface Curve1dTruth extends TruthContract {
  readonly task: 'regression'
  readonly name: string
  /** The family and link the data were drawn from (location–scale noise reports `gaussian`, `identity`). */
  readonly family: FamilyName
  readonly link: LinkName
  /** The law of y around its mean, for captions. */
  readonly law: string
  /** η(x) = g(μ(x)), [n]. */
  linearPredictor(x: Tensor): Tensor
  /** μ(x) = E[y | x], [n]. */
  mean(x: Tensor): Tensor
  /** The standard deviation of y at x, [n]. */
  sdAt(x: Tensor): Tensor
  /** The τ-expectile of y given x, [n], τ ∈ (0, 1). */
  expectile(x: Tensor, tau: number): Tensor
  /** P(y < e_τ(x)) averaged over x ~ U(0, 1): the share of the population below the true τ-expectile curve. */
  shareBelow(tau: number): number
  predictive(x: Tensor): Distribution
  decide(x: Tensor): Tensor
  /** μ(x) without `f`; with `f`, E[f(y) | x] over the law's atoms. */
  expect(x: Tensor, f?: (y: number) => number): Tensor
}

/** How y varies around μ(x) in a `Curve1dModel`. */
export type Curve1dLaw =
  /** y = μ(x) + σ(x)Z, Z standardised: normal, or a log-normal with log-scale sd `skew` (default 0.75), shifted. */
  | { kind: 'location-scale'; noise: 'normal' | 'skewed'; sd: (x: number) => number; skew?: number }
  /** y from the exponential family at μ(x) with dispersion φ (Poisson, Bernoulli, or gamma with CV² = φ). */
  | { kind: 'family'; dispersion: number }

/** The parts of a `Curve1dTruth`. */
export interface Curve1dModel {
  name: string
  family: FamilyName
  link: LinkName
  /** η(x) on [0, 1]; μ = g⁻¹(η). */
  eta: (x: number) => number
  law: Curve1dLaw
}

const ATOMS = 4000
const levels = lazy(() =>
  fromData(
    Float64Array.from({ length: ATOMS }, (_, i) => (i + 0.5) / ATOMS),
    [ATOMS],
  ),
)

/** Build a one-dimensional smooth regression truth (see `Curve1dTruth`). */
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

/** One deterministic sinusoid A sin(2πft + φ) of a spectral model; its power in a one-sided spectrum is A²/2. */
export interface SpectralLine {
  /** Frequency, in Hz (or cycles per unit of time). */
  readonly frequency: number
  readonly amplitude: number
  /** Phase φ, in radians. */
  readonly phase: number
}

/** An ARMA(p, q) process x_t = Σ φᵢ x_{t−i} + ε_t + Σ θⱼ ε_{t−j}, ε_t ~ N(0, σ²); white noise has no coefficients. */
export interface ArmaParts {
  readonly ar: readonly number[]
  readonly ma: readonly number[]
  readonly sigma2: number
}

/**
 * The parts of a `SpectralTruth`: sinusoids plus a stationary ARMA process, sampled at rate fs; optionally a second
 * series y = h ∗ x + v coupled to the first through a rational filter H = B/A and independent ARMA noise v.
 */
export interface SpectralModel {
  readonly name: string
  /** Sample rate (for uneven sampling, the mean rate n/T, which only scales the noise density). */
  readonly fs: number
  readonly lines: readonly SpectralLine[]
  readonly noise: ArmaParts
  readonly coupling?: { readonly b: readonly number[]; readonly a: readonly number[]; readonly noise: ArmaParts }
}

/** The true spectra of a coupled pair (x, y), with y = h ∗ x + v. */
export interface CoupledSpectra {
  /** The one-sided PSD of y, |H|² S_xx + S_vv. */
  psd(f: Tensor | ArrayLike<number>): Tensor
  /** The cross-spectral density S_xy = H S_xx (scipy's conj(X)·Y convention), complex128. */
  crossSpectrum(f: Tensor | ArrayLike<number>): Tensor
  /** The magnitude-squared coherence |H|² S_xx / (|H|² S_xx + S_vv). */
  coherence(f: Tensor | ArrayLike<number>): Tensor
  /** The phase of S_xy, arg H(f), in radians (unwrapped along f). */
  phase(f: Tensor | ArrayLike<number>): Tensor
}

/**
 * The truth of a series with a known power spectrum (task `spectrum`): a line spectrum (sinusoids of known frequency,
 * amplitude and phase) on top of the continuous spectrum of a stationary ARMA process, which is white noise when the
 * process has no coefficients. Methods over times (an [n, 1] or [n] tensor): `expect` is the sum of the sinusoids,
 * `predictive` the marginal normal law around it, `bayesRisk` the variance of the stochastic part. `psd(f)` is the
 * continuous part as a one-sided density (`aifn-compute/signal/statistical`'s `armaSpectrum`), and `lines` the line part,
 * each line of power A²/2: a periodogram of n samples shows a line as a peak of height ≈ (A²/2)·n/fs in density units
 * on top of `psd`.
 */
export interface SpectralTruth extends TruthContract {
  readonly task: 'spectrum'
  readonly name: string
  readonly fs: number
  readonly lines: readonly SpectralLine[]
  /** The continuous part's one-sided power spectral density at frequencies f. */
  psd(f: Tensor | ArrayLike<number>): Tensor
  /** The variance of the stochastic part (the integral of `psd` over [0, fs/2]). */
  readonly noiseVariance: number
  /** The total variance: `noiseVariance` + Σ A²/2. */
  readonly variance: number
  /** For a coupled pair, the second series' spectra and the cross-spectra. */
  readonly coupled?: CoupledSpectra
  readonly model: SpectralModel
}

const freqArray = (f: Tensor | ArrayLike<number>): Float64Array => Float64Array.from(isTensor(f) ? toFlat(f) : f)

/** The variance of an ARMA process from its impulse response, σ² Σ ψⱼ² (ψ₀ = 1, ψⱼ = θⱼ + Σ φᵢ ψ_{j−i}). */
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

/** A filter's frequency response H(e^{iω}) = B(e^{−iω}) / A(e^{−iω}) at ω = 2πf/fs, as [re, im]. */
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

/** Build a spectral truth (see `SpectralTruth`). */
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

/** The rows of an [n, d] matrix as a fresh row-major Float64Array, with n and d. */
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

/** An [n, d] float64 matrix from row-major data. */
export function pointsFrom(data: Float64Array, n: Size, d: Size): Tensor {
  return fromData(data, [n, d])
}

/** Row-major values of a tensor (any shape) as a fresh Float64Array. */
function flat(t: Tensor): Float64Array {
  if (t.shape.length === 2) return points(t).data
  if (t.shape.length === 1)
    return Float64Array.from({ length: t.shape[0] }, (_, i) => Number(t.data[t.offset + i * t.strides[0]]))
  return Float64Array.of(Number(t.data[t.offset]))
}

/** Per-row log densities of k classes (each [n]) as one [n, k] matrix. */
export function classColumns(columns: readonly Tensor[]): Tensor {
  return stack(columns, 1)
}

function lazy<T>(f: () => T): () => T {
  let cached: { value: T } | undefined
  return () => (cached ??= { value: f() }).value
}

/** Apply the label operations to an unnormalised vector over clean classes. */
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

function normalise(v: number[]): number[] {
  const s = v.reduce((a, b) => a + b, 0)
  return v.map((a) => a / s)
}

/** The clean log joint log πⱼ + log p(x | j), [n × k] row-major. */
function logJoint(model: ClassModel, x: Tensor): { a: Float64Array; n: Size; k: Size } {
  const ld = model.logDensity(x)
  const k = model.classes
  const a = flat(ld)
  const n = a.length / k
  const logPrior = model.priors.map(Math.log)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) a[i * k + j] += logPrior[j]
  return { a, n, k }
}

/** The observed-label posterior, [n × k] row-major (NaN rows where every class has zero density). */
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
 * (which change the effective priors), optionally followed by one symmetric binary flip at rate ρ ≤ 1/2, which maps an
 * error e to ρ + (1 − 2ρ) e.
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

/** Build the truth model from a class model. */
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

const matrixOf = (rows: readonly (readonly number[])[]): Tensor =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])
const vectorOf = (v: readonly number[]): Tensor => fromData(Float64Array.from(v), [v.length])

/** The Gaussian classes N(meanⱼ, Σⱼ): log p(x | j) for every row, [n, k]. Throws when a covariance is not positive definite. */
export function gaussianClasses(
  means: readonly (readonly number[])[],
  covariances: readonly (readonly (readonly number[])[])[],
): (x: Tensor) => Tensor {
  const laws = means.map((m, j) => MultivariateNormal(vectorOf(m), { covariance: matrixOf(covariances[j]) }))
  return (x) => classColumns(laws.map((law) => law.logProb(x) as Tensor))
}

/**
 * The Bayes error of two Gaussian classes with a shared covariance at Mahalanobis distance Δ and priors (π₀, π₁). The
 * log likelihood ratio L is N(±Δ²/2, Δ²) under each class and the Bayes rule says 1 when L > t = log(π₀/π₁), so
 * the error is π₁ Φ((t − Δ²/2)/Δ) + π₀ Φ((−t − Δ²/2)/Δ) (e.g. Duda, Hart and Stork, 2001, §2.8.3).
 */
export function twoGaussianBayesError(delta: number, priors: readonly number[]): number {
  const [p0, p1] = priors
  if (p0 === 0 || p1 === 0) return 0
  if (delta === 0) return Math.min(p0, p1)
  const t = Math.log(p0 / p1)
  return p1 * normalCdf((t - (delta * delta) / 2) / delta) + p0 * normalCdf((-t - (delta * delta) / 2) / delta)
}

/** Rows evaluated per block against the nodes of a curve density, so a block holds at most ~2¹⁸ node terms. */
const BLOCK_TERMS = 1 << 18

/**
 * The log density of points uniform along a curve c(u), u uniform on [0, 1], blurred by isotropic Gaussian noise of
 * standard deviation `sd`: log (1/M) Σₘ N(x; c(uₘ), sd² I), a midpoint rule over M nodes, evaluated as a batch of
 * bivariate normals. M is chosen so that nodes are at most sd/2 apart along the curve, which keeps the rule accurate
 * to well under a percent. Returns the density of every row of x ([n]).
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

/** A reference sample with equal weights. */
export function equalReference(x: Tensor): Reference {
  const n = x.shape[0]
  return { x, weights: new Float64Array(n).fill(1 / n) }
}

// ── Regression ───────────────────────────────────────────────────────────────────────────────────────────────────────

const HERMITE = lazy(() => {
  const rule = gaussHermite(32, { probabilists: true })
  const w = flat(rule.weights)
  const total = w.reduce((a, b) => a + b, 0)
  return { nodes: flat(rule.nodes), weights: w.map((v) => v / total) }
})

/** Build the truth model from a regression model. */
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

/** A regression truth with homoscedastic noise unless `sdAt` is given. */
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

/** A regression truth with some parts of its model replaced. */
export function remodelRegression(t: RegressionTruth, edit: Partial<RegressionModel>): RegressionTruth {
  return regressionTruthOf({ ...t.model, ...edit })
}

/** The truth of a piecewise series from its segments (contiguous, covering 0 … n − 1). */
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
 * The parts of a `RegimeTruth`: K regimes, a gate P(regime k | x), and each regime's function of x (the mean of y for
 * regression, the log-odds of class 1 for classification), on a box of inputs.
 */
export interface RegimeModel {
  name: string
  task: 'regression' | 'classification'
  regimes: Size
  /** P(regime k | x) at one point, length K. */
  gate: (x: Row) => number[]
  /** Regime k's function at one point: the mean of y (regression) or the log-odds of y = 1 (classification). */
  fn: (x: Row, k: number) => number
  /** The noise sd of y around a regime's mean (regression; 0 for classification). */
  noiseSd: number
  /** The input box the population's x is uniform on. */
  lower: readonly number[]
  upper: readonly number[]
  /** Each regime's function as text, for captions. */
  formulas: readonly string[]
}

/**
 * The truth of data in which a gate over x picks one of K regimes and the regime's function generates y: a mixture of
 * regressions or classifiers whose weights depend on x (the generative model of a mixture of experts; Jacobs, Jordan,
 * Nowlan and Hinton, 1991). When the gate is hard, the regimes partition the input space and the truth is a piecewise
 * function.
 */
export interface RegimeTruth extends TruthContract {
  readonly task: 'regression' | 'classification'
  readonly name: string
  readonly regimes: Size
  /** P(regime k | x) per row ([n, K]). */
  gate(x: Tensor): Tensor
  /** The most probable regime per row (int32 [n]). */
  regime(x: Tensor): Tensor
  /** Each regime's prediction per row ([n, K]): its mean of y (regression) or P(y = 1 | x, regime) (classification). */
  regimeMean(x: Tensor): Tensor
  /** E[y | x] per row ([n]): the gate-weighted regime means. */
  mean(x: Tensor): Tensor
  /** log p(y | x) per row ([n]): the log of the gate-weighted mixture of the regimes' laws. */
  logLikelihood(x: Tensor, y: Tensor): Tensor
  /**
   * Regression: the normal with y's conditional mean and variance (exact where the gate is hard, moment-matched where
   * regimes overlap; `logLikelihood` is exact everywhere). Classification: the Bernoulli of class 1.
   */
  predictive(x: Tensor): Distribution
  /** The mean (regression) or the Bayes class (classification, int32). */
  decide(x: Tensor): Tensor
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** E[Var(y | x)] (regression) or the Bayes error E[min(p, 1 − p)] (classification), over a grid of the box. */
  readonly bayesRisk: number
  readonly model: RegimeModel
}

const sigmoidOf = (v: number) => 1 / (1 + Math.exp(-v))

/** Build a regime truth (see `RegimeTruth`). */
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

/** One exact solution of an inverse problem: a target y with f(y) = x, and its share of p(y | x). */
export interface InverseSolution {
  readonly value: number[]
  /** The solution's probability in the small-noise limit, p(y*)/|det f′(y*)| normalised over the solutions. */
  readonly weight: number
}

/**
 * The parts of an `InverseTruth`: targets y drawn from a uniform prior on a box, inputs x = f(y) + ε with
 * ε ~ N(0, σ²I), and the inverse of f, which is multi-valued.
 */
export interface InverseModel {
  name: string
  /** The dimension D of the target y. */
  outputs: Size
  /** The forward map f(y), noise-free. */
  forward: (y: Row) => number[]
  /** Every y in the prior's box with f(y) = x, with its weight (empty where x is outside f's image). */
  solutions: (x: Row) => InverseSolution[]
  /** The law of y given x as weighted atoms: a quadrature rule of the exact posterior, or the solutions. */
  atoms: (x: Row) => { values: number[][]; weights: number[] }
  /** log p(y | x), where the posterior has a closed form up to quadrature (1-d targets). */
  logLikelihood?: (x: Row, y: Row) => number
  /** The noise sd σ of the inputs. */
  noise: number
  /** The prior's box of targets. */
  lower: readonly number[]
  upper: readonly number[]
  /** f as text, for captions. */
  formula: string
}

/**
 * The truth of an inverse problem (Bishop, 1994, "Mixture density networks"): y is drawn uniformly on a box and
 * observed through x = f(y) + ε, and the task is to predict y from x. Where f folds over, y given x has a mode at every
 * solution of f(y) = x, and the conditional mean E[y | x], the minimiser of the squared error, can fall between them on
 * no solution at all.
 */
export interface InverseTruth extends TruthContract {
  readonly task: 'regression'
  readonly name: string
  /** The dimension D of y. */
  readonly outputs: Size
  /** The noise-free solutions of f(y) = x at one input, most probable first. */
  solutions(x: Row): InverseSolution[]
  /** f(y) at one target. */
  forward(y: Row): number[]
  /** E[y | x] per row: [n] for D = 1, else [n, D]. */
  mean(x: Tensor): Tensor
  /** Var(yⱼ | x) per row, the shape of `mean`. */
  variance(x: Tensor): Tensor
  /** log p(y | x) per row ([n]); NaN where the model has no density (a law concentrated on the solutions). */
  logLikelihood(x: Tensor, y: Tensor): Tensor
  /** The normal with y's conditional mean and variance per row (moment-matched: the true law is multimodal). */
  predictive(x: Tensor): AnyUnivariate
  /** E[y | x], the Bayes decision under squared loss. */
  decide(x: Tensor): Tensor
  /** E[f(y) | x] ([n]; D = 1 only with `f`). */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** E[Σⱼ Var(yⱼ | x)] over the prior (on noise-free inputs, a 2000-point grid in 1-d, 60 × 60 in 2-d). */
  readonly bayesRisk: number
  readonly model: InverseModel
}

/** Build an inverse-problem truth (see `InverseTruth`). */
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
