/** The registry of `aifn-methods/neural/privacy`: a streamed DP-SGD study of a small MLP across noise multipliers. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as study from './study'

const fn = definer<FunctionInfo>('function', 'neural/privacy')
const notes = ['differentially-private-stochastic-gradient-descent', 'differential-privacy']

fn(
  {
    key: 'privateStudyModel',
    name: 'DP-SGD study MLP',
    summary: 'The one-hidden-layer tanh MLP the DP-SGD study trains, and the map from flat θ to its parameters.',
    role: 'construction',
    notes,
  },
  study.privateStudyModel,
)
fn(
  {
    key: 'privateTrainingStudy',
    name: 'Streamed DP-SGD study',
    summary:
      'The same MLP trained by DP-SGD once per noise multiplier, recording loss, test accuracy and the ε spent at checkpoints.',
    role: 'simulation',
    random: true,
    notes,
    cite: ['abadi2016dp', 'mironov2017'],
  },
  study.privateTrainingStudy,
)

/** The functions of the module, keyed by name. */
export const privacyStudyFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', study) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
