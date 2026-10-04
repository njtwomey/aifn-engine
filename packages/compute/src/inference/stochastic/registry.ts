/**
 * The algorithms of `aifn-compute/inference/stochastic`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as chains from './chains'
import * as diagnostics from './diagnostics'
import * as factorGibbs from './factorGibbs'
import * as gibbs from './gibbs'
import * as hamiltonian from './hamiltonian'
import * as langevin from './langevin'
import * as metropolis from './metropolis'
import * as smc from './smc'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/stochastic')

algorithm(
  {
    key: 'metropolisHastings',
    name: 'Metropolis–Hastings',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['metropolis-hastings', 'markov-chain-monte-carlo'],
    cite: ['metropolis1953', 'hastings1970'],
  },
  metropolis.metropolisHastings,
)
algorithm(
  {
    key: 'randomWalkMetropolis',
    name: 'Random-walk Metropolis',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['metropolis-hastings', 'markov-chain-monte-carlo'],
    cite: ['metropolis1953'],
  },
  metropolis.randomWalkMetropolis,
)
algorithm(
  {
    key: 'independenceMetropolis',
    name: 'Independence Metropolis',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['metropolis-hastings'],
    cite: ['tierney1994'],
  },
  metropolis.independenceMetropolis,
)
algorithm(
  {
    key: 'gibbs',
    name: 'Gibbs sampling',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['gibbs-sampling', 'markov-chain-monte-carlo'],
    cite: ['geman1984'],
  },
  gibbs.gibbs,
)
algorithm(
  {
    key: 'sliceSampler',
    name: 'Slice sampling',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['markov-chain-monte-carlo'],
  },
  gibbs.sliceSampler,
)
algorithm(
  {
    key: 'hmc',
    name: 'Hamiltonian Monte Carlo',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    glossary: 'hmc',
    notes: ['hamiltonian-monte-carlo', 'markov-chain-monte-carlo'],
    cite: ['neal2011'],
  },
  hamiltonian.hmc,
)
algorithm(
  {
    key: 'nuts',
    name: 'No-U-Turn sampler',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    glossary: 'nuts',
    notes: ['hamiltonian-monte-carlo'],
    cite: ['hoffman2014', 'betancourt2017'],
  },
  hamiltonian.nuts,
)
algorithm(
  {
    key: 'unadjustedLangevin',
    name: 'Unadjusted Langevin',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['langevin-dynamics'],
    cite: ['roberts1996'],
  },
  langevin.unadjustedLangevin,
)
algorithm(
  {
    key: 'mala',
    name: 'Metropolis-adjusted Langevin',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['langevin-dynamics'],
    cite: ['roberts1996'],
  },
  langevin.mala,
)
algorithm(
  {
    key: 'sgld',
    name: 'Stochastic gradient Langevin dynamics',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'gradEstimate', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['stochastic-gradient-langevin-dynamics'],
    cite: ['welling2011'],
  },
  langevin.sgld,
)
algorithm(
  {
    key: 'langevinParticles',
    name: 'Langevin on a batch of particles',
    problem: 'log-density',
    state: { iterate: 'x', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['langevin-dynamics', 'energy-based-models'],
    cite: ['du2019', 'nijkamp2019'],
  },
  langevin.langevinParticles,
)
algorithm(
  {
    key: 'particleFilter',
    name: 'Particle filter',
    problem: 'sequence',
    state: { iterate: 'mean', objective: 'logEvidence', flags: ['diverged'] },
    random: true,
    notes: ['particle-filter'],
    cite: ['gordon1993'],
  },
  smc.particleFilter,
)
algorithm(
  {
    key: 'temperedSmc',
    name: 'Tempered sequential Monte Carlo',
    problem: 'log-density',
    state: { iterate: 'particles', objective: 'logEvidence', flags: ['diverged'] },
    random: true,
  },
  smc.temperedSmc,
)
algorithm(
  {
    key: 'factorGraphGibbs',
    name: 'Gibbs sampling on a factor graph',
    problem: 'factor-graph',
    state: { iterate: 'assignment', flags: [] },
    random: true,
    notes: ['gibbs-sampling', 'factor-graph'],
    cite: ['geman1984'],
  },
  factorGibbs.factorGraphGibbs,
)
algorithm(
  {
    key: 'modelGibbs',
    name: 'Gibbs sampling on a model',
    problem: 'factor-graph',
    state: { iterate: 'values', objective: 'logJoint', flags: [] },
    random: true,
    notes: ['gibbs-sampling'],
    cite: ['geman1984'],
  },
  factorGibbs.modelGibbs,
)

/** Every algorithm of the module, keyed by factory name. */
export const stochasticAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', factorGibbs, gibbs, hamiltonian, langevin, metropolis, smc) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'inference/stochastic')
const DIAG = ['markov-chain-monte-carlo-diagnostics']

fn(
  {
    key: 'effectiveSampleSize',
    name: 'Effective sample size',
    role: 'estimator',
    notes: DIAG,
    cite: ['vehtari2021', 'geyer1992'],
  },
  diagnostics.effectiveSampleSize,
)
fn(
  {
    key: 'integratedAutocorrelationTime',
    name: 'Integrated autocorrelation time',
    role: 'estimator',
    notes: DIAG,
    cite: ['geyer1992'],
  },
  diagnostics.integratedAutocorrelationTime,
)
fn(
  {
    key: 'splitRhat',
    name: 'Split R̂',
    tex: '\\hat R',
    role: 'estimator',
    notes: DIAG,
    cite: ['vehtari2021', 'gelman2013'],
  },
  diagnostics.splitRhat,
)
fn(
  { key: 'monteCarloStandardError', name: 'Monte Carlo standard error', role: 'estimator', notes: DIAG },
  diagnostics.monteCarloStandardError,
)
fn({ key: 'summarise', name: 'Chain summary', role: 'estimator', notes: DIAG }, diagnostics.summarise)
fn(
  {
    key: 'sampleChains',
    name: 'Run several chains',
    role: 'simulation',
    random: true,
    notes: ['markov-chain-monte-carlo', ...DIAG],
  },
  chains.sampleChains,
)
fn(
  { key: 'raoBlackwell', name: 'Rao–Blackwellised estimate', role: 'estimator', notes: ['gibbs-sampling'] },
  chains.raoBlackwell,
)
fn(
  { key: 'gibbsMarginals', name: 'Gibbs marginals', role: 'estimator', notes: ['gibbs-sampling'] },
  factorGibbs.gibbsMarginals,
)
fn(
  {
    key: 'gaussianConditionals',
    name: 'Gaussian full conditionals',
    role: 'construction',
    notes: ['gibbs-sampling', 'multivariate-normal-distribution'],
  },
  gibbs.gaussianConditionals,
)
fn(
  {
    key: 'conditionalMean',
    name: 'Gaussian conditional mean',
    role: 'property',
    notes: ['multivariate-normal-distribution', 'schur-complement'],
  },
  gibbs.conditionalMean,
)
fn(
  {
    key: 'leapfrog',
    name: 'Leapfrog integrator',
    role: 'simulation',
    notes: ['hamiltonian-monte-carlo'],
    cite: ['neal2011'],
  },
  hamiltonian.leapfrog,
)
fn(
  {
    key: 'persistentLangevin',
    name: 'Persistent Langevin with a replay buffer',
    role: 'simulation',
    random: true,
    notes: ['energy-based-models', 'joint-energy-models'],
    cite: ['du2019', 'grathwohl2019', 'tieleman2008'],
  },
  langevin.persistentLangevin,
)
fn(
  {
    key: 'chainBuffer',
    name: 'Replay buffer of persistent chains',
    role: 'construction',
    random: true,
    notes: ['energy-based-models'],
    cite: ['du2019'],
  },
  langevin.chainBuffer,
)
fn(
  { key: 'resample', name: 'Resample particles', role: 'simulation', random: true, notes: ['particle-filter'] },
  smc.resample,
)

/** The functions of the module, keyed by name. */
export const stochasticFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', diagnostics, chains, factorGibbs, gibbs, hamiltonian, langevin, smc) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
