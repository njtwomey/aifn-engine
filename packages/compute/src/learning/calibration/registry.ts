/**
 * The registry of `aifn-compute/learning/calibration`: pool adjacent violators as a step-through algorithm (the roles
 * of its state's fields in `state`), and the fits and maps as functions with the notes they serve.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as isotonic from './isotonic'
import * as maps from './maps'
import * as platt from './platt'

const NOTES = ['isotonic-regression', 'isotonic-calibration']

definer<AlgorithmInfo>('algorithm', 'learning/calibration')(
  {
    key: 'poolAdjacentViolatorsSteps',
    name: 'Pool adjacent violators',
    summary: 'Add points one at a time and pool the last two blocks while they violate the order.',
    problem: 'least-squares',
    state: { iterate: 'fit', objective: 'sse', flags: [] },
    notes: NOTES,
    cite: ['ayer1955', 'best1990'],
  },
  isotonic.poolAdjacentViolatorsSteps,
)
definer<FunctionInfo>('function', 'learning/calibration')(
  {
    key: 'isotonicRegression',
    name: 'Isotonic regression',
    summary: 'The monotone least-squares fit, with ties in x pooled first.',
    role: 'fit',
    notes: NOTES,
    cite: ['ayer1955', 'zadrozny2002'],
  },
  isotonic.isotonicRegression,
)

const fn = definer<FunctionInfo>('function', 'learning/calibration')
const CAL = ['classifier-calibration', 'calibration-definitions']

fn(
  {
    key: 'plattScaling',
    name: 'Platt scaling',
    tex: 'P(y = 1 \\mid f) = 1/(1 + e^{Af + B})',
    summary: 'A sigmoid fitted to real-valued scores by maximum likelihood, with Platt’s smoothed targets.',
    role: 'fit',
    notes: ['platt-scaling', ...CAL, 'support-vector-machine'],
    cite: ['platt1999', 'niculescumizil2005'],
  },
  platt.plattScaling,
)
fn(
  {
    key: 'temperatureScaling',
    name: 'Temperature scaling',
    tex: '\\operatorname{softmax}(z/T)',
    summary: 'Divide every logit by one fitted T > 0; the predicted class never changes.',
    role: 'fit',
    notes: ['temperature-scaling', ...CAL],
    cite: ['guo2017'],
  },
  maps.temperatureScaling,
)
fn(
  {
    key: 'betaCalibration',
    name: 'Beta calibration',
    tex: '\\sigma(a \\ln s - b \\ln(1 - s) + c)',
    summary: 'A logistic regression on ln s and −ln(1 − s): exact when each class’s scores are beta-distributed.',
    role: 'fit',
    notes: ['beta-calibration', ...CAL],
    cite: ['kull2017', 'kull2017b'],
  },
  maps.betaCalibration,
)
fn(
  {
    key: 'dirichletCalibration',
    name: 'Dirichlet calibration',
    tex: '\\operatorname{softmax}(W \\ln q + b)',
    summary: 'A multinomial logistic regression on the log-probabilities, with the ODIR penalty.',
    role: 'fit',
    notes: ['dirichlet-calibration', ...CAL],
    cite: ['kull2019'],
  },
  maps.dirichletCalibration,
)
fn(
  {
    key: 'histogramBinning',
    name: 'Histogram binning',
    summary: 'Replace each bin’s scores by its fraction of positives.',
    role: 'fit',
    notes: ['classifier-calibration', 'reliability-diagrams-and-consistency-bars'],
    cite: ['zadrozny2001'],
  },
  maps.histogramBinning,
)
fn(
  {
    key: 'isotonicCalibration',
    name: 'Isotonic calibration',
    summary: 'The monotone step map of scores to probabilities by pool adjacent violators.',
    role: 'fit',
    notes: NOTES,
    cite: ['zadrozny2002'],
  },
  maps.isotonicCalibration,
)
fn(
  {
    key: 'topLabelConfidence',
    name: 'Top-label confidence',
    summary: 'Each case’s largest probability and whether its top class is right, for confidence reliability diagrams.',
    role: 'transform',
    notes: ['reliability-diagrams-and-consistency-bars', 'calibration-error', 'calibration-definitions'],
    cite: ['guo2017'],
  },
  maps.topLabelConfidence,
)

/** The algorithms of the module, keyed by factory name. */
export const calibrationAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', isotonic) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const calibrationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', isotonic, maps, platt) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
