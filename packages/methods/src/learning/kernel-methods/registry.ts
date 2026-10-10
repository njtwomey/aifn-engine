/** The registry of `aifn-methods/learning/kernel-methods`: SVM solvers as traceable algorithms. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as crammerSinger from './crammerSinger'
import * as svm from './svm'

/** A table of registry entries of one kind, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'learning/kernel-methods')
const fn = definer<FunctionInfo>('function', 'learning/kernel-methods')
const SVM = ['solving-support-vector-machines', 'support-vector-machine']
const state = { iterate: 'weights', objective: 'primalObjective', flags: ['converged'] } as const

algorithm(
  {
    key: 'smoSteps',
    name: 'Sequential minimal optimisation',
    summary: 'Update two dual variables per step in closed form.',
    problem: 'quadratic-program',
    state,
    notes: [...SVM, 'kernel-support-vector-machine'],
    cite: ['platt1998'],
  },
  svm.smoSteps,
)
algorithm(
  {
    key: 'dualCoordinateSteps',
    name: 'Dual coordinate descent (linear SVM)',
    problem: 'quadratic-program',
    state,
    notes: SVM,
  },
  svm.dualCoordinateSteps,
)
algorithm(
  {
    key: 'pegasosSteps',
    name: 'Pegasos',
    summary: 'Stochastic subgradient descent on the primal SVM objective.',
    problem: 'objective',
    state,
    random: true,
    notes: [...SVM, 'subgradients'],
  },
  svm.pegasosSteps,
)
algorithm(
  {
    key: 'crammerSingerSteps',
    name: 'Crammer–Singer multiclass SVM',
    problem: 'quadratic-program',
    state,
    notes: ['multiclass-support-vector-machines'],
    cite: ['crammer2001'],
  },
  crammerSinger.crammerSingerSteps,
)
fn(
  {
    key: 'dualDecision',
    name: 'SVM decision from dual variables',
    role: 'inference',
    notes: ['support-vector-machine', 'representer-theorem'],
  },
  svm.dualDecision,
)

/** The algorithms of the module, keyed by factory name. */
export const kernelMethodsAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  svm,
  crammerSinger,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const kernelMethodsFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', svm) as Table<FunctionInfo>
