/**
 * The algorithms and functions of `aifn-compute/signal/statistical`, registered with their metadata. An algorithm is
 * registered with what its factory takes (`problem`) and the roles of its state's fields (`state`: iterate,
 * objective, and the `Status` flags it sets), so a generic trace view picks default series and a worker can address
 * an algorithm by key (design S §2.3); a function with its role, the notes it serves and its citations.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as autoregression from './autoregression'
import * as parametric from './parametric'
import * as adaptive from './adaptive'

const algorithm = definer<AlgorithmInfo>('algorithm', 'signal/statistical')

algorithm(
  {
    key: 'lms',
    stability: 'stable',
    name: 'Least mean squares (LMS)',
    summary: 'An adaptive FIR filter that takes one stochastic-gradient step on the squared error per sample.',
    problem: 'signal',
    state: { iterate: 'w', objective: 'squaredError', flags: ['terminated'] },
    notes: ['least-mean-squares-filter'],
    cite: ['widrow1976', 'sayed2008'],
  },
  adaptive.lms,
)
algorithm(
  {
    key: 'nlms',
    stability: 'stable',
    name: 'Normalised LMS',
    summary:
      "LMS with each step divided by the regressor's energy, so the step size does not depend on the input scale.",
    problem: 'signal',
    state: { iterate: 'w', objective: 'squaredError', flags: ['terminated'] },
    notes: ['least-mean-squares-filter'],
    cite: ['sayed2008'],
  },
  adaptive.nlms,
)
algorithm(
  {
    key: 'rls',
    stability: 'stable',
    name: 'Recursive least squares (RLS)',
    summary:
      'An adaptive FIR filter that keeps the exact exponentially weighted least-squares taps, updated per sample.',
    problem: 'signal',
    state: { iterate: 'w', objective: 'squaredError', flags: ['terminated'] },
    notes: ['recursive-least-squares-filter'],
    cite: ['sayed2008'],
  },
  adaptive.rls,
)

/** Every algorithm of the module, keyed by factory name. */
export const statisticalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', adaptive) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'signal/statistical')
const AR = ['parametric-spectral-estimation', 'linear-prediction', 'autoregressive-model']

fn(
  {
    key: 'yuleWalker',
    name: 'Yule–Walker AR estimate',
    summary: 'AR coefficients from the sample autocorrelation by the Levinson–Durbin recursion.',
    role: 'estimator',
    notes: AR,
    cite: ['yule1927', 'walker1931'],
  },
  autoregression.yuleWalker,
)
fn(
  {
    key: 'burg',
    name: "Burg's AR estimate",
    summary: 'AR coefficients minimising forward and backward prediction errors, one reflection coefficient per order.',
    role: 'estimator',
    notes: AR,
  },
  autoregression.burg,
)
fn(
  {
    key: 'leastSquaresAr',
    name: 'Least-squares AR estimate',
    summary: 'AR coefficients minimising forward (and backward) prediction errors without zero padding.',
    role: 'estimator',
    notes: AR,
    cite: ['stoica2005'],
  },
  parametric.leastSquaresAr,
)
fn(
  {
    key: 'armaSpectrum',
    name: 'ARMA power spectral density',
    summary: 'σ² |B(e^{−iω})|² / |A(e^{−iω})|²: the spectrum of white noise through a rational filter.',
    role: 'property',
    returns: 'spectrum',
    notes: [
      'autoregressive-moving-average-model',
      'parametric-spectral-estimation',
      'autocorrelation-and-wiener-khinchin',
    ],
  },
  parametric.armaSpectrum,
)
fn(
  {
    key: 'arPsd',
    name: 'Autoregressive spectral estimate',
    summary: 'The spectrum of an AR(p) fitted by Burg, Yule–Walker or least squares.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['parametric-spectral-estimation', 'autoregressive-model'],
    cite: ['stoica2005'],
  },
  parametric.arPsd,
)
fn(
  {
    key: 'music',
    name: 'MUSIC pseudospectrum',
    summary: 'Peaks where the steering vector is orthogonal to the noise subspace of the correlation matrix.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['subspace-frequency-estimation', 'parametric-spectral-estimation'],
    cite: ['schmidt1986'],
  },
  parametric.music,
)
fn(
  {
    key: 'esprit',
    name: 'ESPRIT frequency estimates',
    summary: 'Frequencies from the rotational invariance of the signal subspace, without a grid search.',
    role: 'estimator',
    notes: ['subspace-frequency-estimation', 'parametric-spectral-estimation'],
    cite: ['roy1989'],
  },
  parametric.esprit,
)
fn(
  {
    key: 'sinusoidFit',
    name: 'Least-squares sinusoid fit',
    summary: 'Amplitudes and phases of sinusoids at known frequencies, a linear least-squares problem.',
    role: 'fit',
    notes: ['subspace-frequency-estimation', 'parametric-spectral-estimation'],
    cite: ['stoica2005'],
  },
  parametric.sinusoidFit,
)

/** The functions of the module, keyed by name. */
export const statisticalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', autoregression, parametric) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
