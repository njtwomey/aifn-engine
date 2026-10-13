/** The functions of `aifn-compute/learning/selection`, registered with the sources they follow. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bootstrap from './bootstrap'
import * as criteria from './criteria'
import * as predictive from './predictive'
import * as rules from './rules'
import * as smoothers from './smoothers'

const fn = definer<FunctionInfo>('function', 'learning/selection')

fn(
  {
    key: 'linearSmootherCv',
    name: 'LOO and GCV of a linear smoother',
    summary: 'Leave-one-out residuals (y - fit) / (1 - S_ii) and generalised cross-validation from a smoother matrix.',
    role: 'estimator',
    cite: ['allen1974', 'craven1979', 'hastie2009'],
  },
  smoothers.linearSmootherCv,
)
fn(
  {
    key: 'ridgeSmoother',
    name: 'Ridge smoother matrix',
    summary: 'The hat matrix of ridge regression with an unpenalised intercept, from one SVD of the centred design.',
    role: 'construction',
    cite: ['hoerl1970', 'hastie2009'],
  },
  smoothers.ridgeSmoother,
)
fn(
  {
    key: 'ridgeCv',
    name: 'Ridge cross-validation',
    summary: 'Exact leave-one-out and GCV of ridge regression over a grid of penalties from one SVD, as RidgeCV.',
    role: 'estimator',
    cite: ['allen1974', 'craven1979', 'hastie2009'],
  },
  smoothers.ridgeCv,
)
fn(
  {
    key: 'oneStandardError',
    name: 'One-standard-error rule',
    summary: 'The simplest candidate whose cross-validated mean is within one standard error of the best.',
    role: 'solver',
    cite: ['breiman1984', 'hastie2009'],
  },
  rules.oneStandardError,
)
fn(
  {
    key: 'bootstrapPredictionError',
    name: 'Bootstrap prediction error',
    tex: '\\widehat{\\mathrm{Err}}^{(.632+)}',
    summary: 'Leave-one-out bootstrap, .632 and .632+ estimates of prediction error, with the no-information rate.',
    role: 'estimator',
    random: true,
    cite: ['efron1997', 'hastie2009'],
  },
  bootstrap.bootstrapPredictionError,
)
fn(
  {
    key: 'informationCriteria',
    name: 'Information criteria',
    summary: 'AIC, AICc and BIC from maximised log-likelihoods, elementwise over candidates.',
    role: 'estimator',
    cite: ['akaike1974', 'hurvich1989', 'schwarz1978', 'burnham2002'],
  },
  criteria.informationCriteria,
)
fn(
  {
    key: 'mallowsCp',
    name: "Mallows' Cp",
    tex: 'C_p',
    summary: 'RSS / n + 2 d sigma^2 / n, and the classical RSS / sigma^2 - n + 2d, elementwise over candidates.',
    role: 'estimator',
    cite: ['mallows1973', 'hastie2009'],
  },
  criteria.mallowsCp,
)
fn(
  {
    key: 'sure',
    name: "Stein's unbiased risk estimate",
    summary: 'An unbiased estimate of the squared-error risk of an estimate of a normal mean, from its divergence.',
    role: 'estimator',
    cite: ['stein1981'],
  },
  criteria.sure,
)
fn(
  {
    key: 'softThresholdSure',
    name: 'SURE for soft thresholding',
    summary: 'SURE of the soft-threshold estimate in closed form at each threshold, as in SureShrink.',
    role: 'estimator',
    cite: ['donoho1995', 'stein1981'],
  },
  criteria.softThresholdSure,
)
fn(
  {
    key: 'lassoSure',
    name: 'SURE for the lasso',
    summary: 'SURE of lasso fits with the number of nonzero coefficients as the degrees of freedom.',
    role: 'estimator',
    cite: ['zou2007', 'stein1981'],
  },
  criteria.lassoSure,
)
fn(
  {
    key: 'waic',
    name: 'WAIC',
    summary: 'The widely applicable information criterion from an S x n matrix of pointwise log-likelihood draws.',
    role: 'estimator',
    cite: ['watanabe2010', 'vehtari2017'],
  },
  predictive.waic,
)
fn(
  {
    key: 'psisLoo',
    name: 'PSIS-LOO',
    summary: 'Leave-one-out elpd by Pareto-smoothed importance sampling, with the k-hat diagnostic, as arviz.',
    role: 'estimator',
    cite: ['vehtari2017', 'vehtari2024'],
  },
  predictive.psisLoo,
)
fn(
  {
    key: 'psisWeights',
    name: 'Pareto-smoothed importance weights',
    summary: 'Normalised importance weights whose largest ratios are replaced by a fitted generalised Pareto tail.',
    role: 'transform',
    cite: ['vehtari2024', 'zhang2009'],
  },
  predictive.psisWeights,
)
fn(
  {
    key: 'dic',
    name: 'Deviance information criterion',
    summary: 'DIC = mean deviance + p_D, with p_D and the variance-based p_V.',
    role: 'estimator',
    cite: ['spiegelhalter2002', 'gelman2013'],
  },
  predictive.dic,
)

/** The functions of the module, keyed by name. */
export const selectionFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', smoothers, rules, bootstrap, criteria, predictive) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
