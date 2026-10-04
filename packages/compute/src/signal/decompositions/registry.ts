/**
 * The algorithms of `aifn-compute/signal/decompositions`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as emd from './emd'
import * as vmd from './vmd'

const algorithm = definer<AlgorithmInfo>('algorithm', 'signal/decompositions')

algorithm(
  {
    key: 'siftSteps',
    name: 'EMD sifting',
    summary: 'The sifting iterations that extract one intrinsic mode function.',
    problem: 'signal',
    state: { iterate: 'h', flags: ['converged', 'terminated'] },
    notes: ['empirical-mode-decomposition'],
    cite: ['huang1998'],
  },
  emd.siftSteps,
)

algorithm(
  {
    key: 'vmdSteps',
    name: 'Variational mode decomposition',
    summary: 'ADMM sweeps that fit K narrow-band modes and their centre frequencies together, in the Fourier domain.',
    problem: 'signal',
    state: { iterate: 'omega', objective: 'change', flags: ['converged'] },
    notes: ['variational-mode-decomposition'],
    cite: ['dragomiretskiy2014'],
  },
  vmd.vmdSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const decompositionsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', emd, vmd) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'signal/decompositions')
const EMD = ['empirical-mode-decomposition', 'hilbert-huang-transform']

fn(
  {
    key: 'emd',
    name: 'Empirical mode decomposition',
    role: 'transform',
    returns: 'decomposition',
    notes: EMD,
    cite: ['huang1998'],
  },
  emd.emd,
)
fn(
  { key: 'extrema', name: 'Extrema and zero crossings', role: 'property', notes: ['empirical-mode-decomposition'] },
  emd.extrema,
)
fn(
  {
    key: 'siftImf',
    name: 'Sift one IMF',
    role: 'transform',
    notes: ['empirical-mode-decomposition'],
    cite: ['huang1998'],
  },
  emd.siftImf,
)
fn(
  {
    key: 'eemd',
    name: 'Ensemble EMD',
    summary: 'EMD averaged over noise-added copies of the signal.',
    role: 'transform',
    returns: 'decomposition',
    random: true,
    notes: ['ensemble-empirical-mode-decomposition'],
    cite: ['wu2009'],
  },
  emd.eemd,
)
fn(
  {
    key: 'ceemdan',
    name: 'CEEMDAN',
    summary:
      'Complete ensemble EMD with adaptive noise: each mode from the ensemble of first IMFs of noise-added residues.',
    role: 'transform',
    returns: 'decomposition',
    random: true,
    notes: ['ensemble-empirical-mode-decomposition'],
    cite: ['torres2011'],
  },
  emd.ceemdan,
)
fn(
  {
    key: 'vmd',
    name: 'Variational mode decomposition',
    role: 'transform',
    returns: 'decomposition',
    notes: ['variational-mode-decomposition'],
    cite: ['dragomiretskiy2014'],
  },
  vmd.vmd,
)
fn(
  { key: 'vmdModes', name: 'VMD modes of a state', role: 'transform', notes: ['variational-mode-decomposition'] },
  vmd.vmdModes,
)

/** The functions of the module, keyed by name. */
export const decompositionsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', emd, vmd) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
