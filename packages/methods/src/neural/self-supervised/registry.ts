/**
 * The registry of `aifn-methods/neural/self-supervised`: self-distillation, and the linear objectives and codes of
 * self-supervision.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as codes from './linear-codes'
import * as predictive from './predictive-code'
import * as distillation from './self-distillation'

const MODULE = 'neural/self-supervised'
const fn = definer<FunctionInfo>('function', MODULE)
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)

algorithm(
  {
    key: 'selfDistillationTraining',
    name: 'Self-distillation with no labels (DINO)',
    summary:
      'A student matches a centred, sharpened EMA teacher across two views; centring and sharpening keep it from collapsing.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', flags: ['diverged'] },
    random: true,
    notes: ['self-distillation-and-dino', 'dimensional-collapse'],
    cite: ['caron2021'],
  },
  distillation.selfDistillationTraining,
)

const SSL = ['self-supervised-learning', 'comparing-self-supervised-methods']

fn(
  {
    key: 'linearCodeObjective',
    name: 'Self-supervised objectives of a linear code',
    summary:
      'Reconstruction, InfoNCE, view agreement and VICReg’s variance hinge for the encoder z = wᵀx, to draw their landscapes.',
    role: 'property',
    notes: SSL,
    cite: ['oord2018', 'chen2020', 'bardes2022'],
  },
  codes.linearCodeObjective,
)
fn(
  {
    key: 'linearCodeLandscape',
    name: 'Objective landscape over linear encoders',
    role: 'property',
    notes: SSL,
    cite: ['oord2018', 'bardes2022'],
  },
  codes.linearCodeLandscape,
)
fn({ key: 'inputCovariance', name: 'Input covariance', role: 'estimator', notes: SSL }, codes.inputCovariance)
fn(
  {
    key: 'thresholdProbeAccuracy',
    name: 'Threshold probe accuracy',
    summary: 'The best accuracy of one threshold on a one-number code: a linear probe of two classes.',
    role: 'estimator',
    notes: SSL,
  },
  codes.thresholdProbeAccuracy,
)

const CPC = ['contrastive-predictive-coding']

fn(
  {
    key: 'linearPredictiveCode',
    name: 'Most informative linear predictive code',
    tex: '\\Gamma w = \\gamma_k,\\quad I(c_t; x_{t+k}) = -\\tfrac12 \\log(1 - \\rho^2)',
    summary: 'The window code of a Gaussian series that keeps most information about the value k steps ahead.',
    role: 'solver',
    notes: CPC,
    cite: ['oord2018'],
  },
  predictive.linearPredictiveCode,
)
fn(
  { key: 'codeCorrelation', name: 'Code correlation with a component', role: 'property', notes: CPC },
  predictive.codeCorrelation,
)
fn(
  { key: 'applyPredictiveCode', name: 'Apply a predictive code', role: 'transform', notes: CPC },
  predictive.applyPredictiveCode,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const sources = [codes, predictive, distillation]
/** The algorithms of the module. */
export const selfSupervisedAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  ...sources,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const selfSupervisedFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  ...sources,
) as Table<FunctionInfo>
