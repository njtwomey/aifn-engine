/**
 * The samplers of `aifn-compute/probability/samplers`, registered as functions that draw (`random: true`) with the notes they
 * serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as samplers from './samplers'

const fn = definer<FunctionInfo>('function', 'probability/samplers')
const REPARAM = ['reparameterisations-of-common-distributions']

fn(
  {
    key: 'logGammaVariate',
    name: 'Log-gamma draws',
    role: 'simulation',
    random: true,
    notes: REPARAM,
    cite: ['marsaglia2000'],
  },
  samplers.logGammaVariate,
)
fn(
  {
    key: 'gammaVariate',
    name: 'Gamma draws (Marsaglia–Tsang)',
    role: 'simulation',
    random: true,
    notes: ['gamma-distribution', ...REPARAM, 'rejection-sampling'],
    cite: ['marsaglia2000'],
  },
  samplers.gammaVariate,
)
fn(
  { key: 'beta', name: 'Beta draws', role: 'simulation', random: true, notes: ['beta-distribution', ...REPARAM] },
  samplers.beta,
)
fn(
  {
    key: 'chiSquare',
    name: 'Chi-square draws',
    role: 'simulation',
    random: true,
    notes: ['chi-squared-distribution', ...REPARAM],
  },
  samplers.chiSquare,
)
fn(
  {
    key: 'studentT',
    name: 'Student t draws',
    role: 'simulation',
    random: true,
    notes: ['student-t-distribution', ...REPARAM],
  },
  samplers.studentT,
)
fn(
  {
    key: 'dirichlet',
    name: 'Dirichlet draws',
    role: 'simulation',
    random: true,
    notes: ['dirichlet-distribution', 'sampling-the-simplex', ...REPARAM],
  },
  samplers.dirichlet,
)
fn(
  {
    key: 'poisson',
    name: 'Poisson draws',
    role: 'simulation',
    random: true,
    notes: ['poisson-distribution', 'poisson-process'],
    cite: ['devroye1986'],
  },
  samplers.poisson,
)
fn(
  {
    key: 'binomial',
    name: 'Binomial draws',
    role: 'simulation',
    random: true,
    notes: ['binomial-distribution'],
    cite: ['devroye1986'],
  },
  samplers.binomial,
)
fn(
  {
    key: 'multinomial',
    name: 'Multinomial draws',
    role: 'simulation',
    random: true,
    notes: ['multinomial-distribution'],
  },
  samplers.multinomial,
)
fn(
  {
    key: 'multivariateNormal',
    name: 'Multivariate normal draws',
    summary: 'μ + Lz with L a Cholesky factor of the covariance and z standard normal.',
    role: 'simulation',
    random: true,
    notes: ['multivariate-normal-distribution', ...REPARAM],
  },
  samplers.multivariateNormal,
)

/** The samplers of the module, keyed by name. */
export const samplerFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', samplers) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
