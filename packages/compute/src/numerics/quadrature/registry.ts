/**
 * The algorithms of `aifn-compute/numerics/quadrature`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as adaptive from './adaptive'
import * as gauss from './gauss'
import * as multivariate from './multivariate'
import * as rules from './rules'

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/quadrature')

algorithm(
  {
    key: 'adaptiveSimpson',
    stability: 'stable',
    name: 'Adaptive Simpson',
    problem: 'integral',
    state: { iterate: 'value', objective: 'error', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['numerical-integration'],
  },
  adaptive.adaptiveSimpson,
)
algorithm(
  {
    key: 'gaussKronrod',
    stability: 'stable',
    name: 'Adaptive Gauss–Kronrod',
    problem: 'integral',
    state: { iterate: 'value', objective: 'error', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['numerical-integration'],
  },
  adaptive.gaussKronrod,
)
algorithm(
  {
    key: 'romberg',
    stability: 'stable',
    name: 'Romberg',
    problem: 'integral',
    state: { iterate: 'value', objective: 'error', flags: ['converged', 'diverged'] },
    notes: ['numerical-integration'],
  },
  rules.romberg,
)
algorithm(
  {
    key: 'monteCarlo',
    stability: 'stable',
    name: 'Monte Carlo integration',
    problem: 'integral',
    state: { iterate: 'value', objective: 'standardError', flags: ['diverged'] },
    random: true,
    notes: ['monte-carlo-integration'],
  },
  multivariate.monteCarlo,
)

/** Every algorithm of the module, keyed by factory name. */
export const quadratureAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', adaptive, multivariate, rules) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'numerics/quadrature')
const NI = ['numerical-integration']
const MC = ['monte-carlo-integration']

fn({ key: 'trapezoid', name: 'Composite trapezoid rule', role: 'solver', notes: NI }, rules.trapezoid)
fn({ key: 'simpson', name: "Composite Simpson's rule", role: 'solver', notes: NI }, rules.simpson)
fn({ key: 'trapezoidSamples', name: 'Trapezoid rule on samples', role: 'solver', notes: NI }, rules.trapezoidSamples)
fn({ key: 'kronrod15', name: 'Gauss–Kronrod 15-point rule', role: 'solver', notes: NI }, adaptive.kronrod15)
fn({ key: 'integrate', name: 'Integrate (adaptive)', role: 'solver', notes: NI }, adaptive.integrate)
fn(
  {
    key: 'gaussLegendre',
    name: 'Gauss–Legendre nodes and weights',
    role: 'construction',
    notes: NI,
    cite: ['golub1969'],
  },
  gauss.gaussLegendre,
)
fn(
  {
    key: 'gaussHermite',
    name: 'Gauss–Hermite nodes and weights',
    role: 'construction',
    notes: [...NI, 'gaussian-integral'],
    cite: ['golub1969'],
  },
  gauss.gaussHermite,
)
fn(
  {
    key: 'gaussLaguerre',
    name: 'Gauss–Laguerre nodes and weights',
    role: 'construction',
    notes: NI,
    cite: ['golub1969'],
  },
  gauss.gaussLaguerre,
)
fn(
  {
    key: 'normalExpectation',
    name: 'Gaussian expectation by Gauss–Hermite',
    tex: '\\mathbb{E}[f(X)],\\ X \\sim \\mathcal{N}(\\mu, \\sigma^2)',
    role: 'solver',
    notes: [...NI, 'expectation'],
  },
  gauss.normalExpectation,
)
fn(
  { key: 'integrateGauss', name: 'Gaussian quadrature on an interval', role: 'solver', notes: NI },
  gauss.integrateGauss,
)
fn(
  { key: 'productRule', name: 'Tensor-product rule', role: 'construction', notes: ['multiple-integrals', ...NI] },
  multivariate.productRule,
)
fn(
  { key: 'integrate2d', name: 'Double integral', role: 'solver', notes: ['multiple-integrals', ...NI] },
  multivariate.integrate2d,
)
fn(
  { key: 'integrateMonteCarlo', name: 'Monte Carlo integral', role: 'estimator', random: true, notes: MC },
  multivariate.integrateMonteCarlo,
)
fn({ key: 'halton', name: 'Halton sequence', role: 'construction', notes: MC }, multivariate.halton)
fn({ key: 'sobol', name: 'Sobol sequence', role: 'construction', notes: MC }, multivariate.sobol)
fn(
  { key: 'quasiMonteCarlo', name: 'Quasi-Monte Carlo integral', role: 'estimator', notes: MC },
  multivariate.quasiMonteCarlo,
)

/** The functions of the module, keyed by name. */
export const quadratureFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', rules, adaptive, gauss, multivariate) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
