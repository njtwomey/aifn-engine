/** The registry of `aifn-methods/theory/adaptive-holdout`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as attack from './attack'
import * as ladder from './ladder'
import * as maximum from './maximum'
import * as thresholdout from './thresholdout'

const fn = definer<FunctionInfo>('function', 'theory/adaptive-holdout')

fn(
  {
    key: 'ladder',
    name: 'Ladder leaderboard',
    summary: 'Report a holdout loss only when it beats the best so far by more than eta, rounded to a multiple of eta.',
    role: 'estimator',
    cite: ['blum2015ladder'],
  },
  ladder.ladder,
)
fn(
  {
    key: 'ladderParameterFree',
    name: 'Parameter-free Ladder',
    summary: 'The Ladder with its step set by a one-sided paired t-test against the best submission so far.',
    role: 'estimator',
    cite: ['blum2015ladder'],
  },
  ladder.ladderParameterFree,
)
fn(
  {
    key: 'thresholdout',
    name: 'Thresholdout',
    summary: 'Answer from the training set unless the holdout disagrees beyond a noisy threshold, then add noise.',
    role: 'estimator',
    random: true,
    cite: ['dwork2015', 'dwork2015holdout'],
  },
  thresholdout.thresholdout,
)
fn(
  {
    key: 'boostingAttack',
    name: 'Boosting attack on a leaderboard',
    summary: 'Random submissions, then the majority vote of those that score well: holdout against true loss.',
    role: 'simulation',
    random: true,
    cite: ['blum2015ladder'],
  },
  attack.boostingAttack,
)
fn(
  {
    key: 'expectedGaussianMaximum',
    name: 'Expected maximum of J normals',
    summary: 'E max of J independent standard normals by quadrature: the winner’s curse of picking the best of J.',
    role: 'property',
  },
  maximum.expectedGaussianMaximum,
)

/** The functions of the module, keyed by name. */
export const adaptiveHoldoutFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', ladder, thresholdout, attack, maximum) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
