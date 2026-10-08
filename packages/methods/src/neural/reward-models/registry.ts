/** The registry of `aifn-methods/neural/reward-models`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as preferences from './preferences'

const fn = definer<FunctionInfo>('function', 'neural/reward-models')
const notes = ['reward-models']

fn(
  {
    key: 'syntheticPreferences',
    name: 'Synthetic preference pairs',
    summary: 'Random response pairs labelled by a gold reward under the Bradley–Terry model, with Gumbel noise.',
    role: 'simulation',
    notes,
    cite: ['bradley1952'],
  },
  preferences.syntheticPreferences,
)
fn(
  {
    key: 'fitBradleyTerry',
    name: 'Bradley–Terry reward model',
    tex: '\\min_w \\sum_i -\\log\\sigma\\big(w^\\top(\\phi(x_{w,i}) - \\phi(x_{l,i}))\\big) + \\tfrac{\\lambda}{2}\\|w\\|^2',
    summary:
      'A linear reward fitted to preference pairs: logistic regression without an intercept on feature differences.',
    role: 'fit',
    notes,
    cite: ['bradley1952'],
  },
  preferences.fitBradleyTerry,
)
fn(
  {
    key: 'bestOfNKl',
    name: 'Best-of-n KL bound',
    tex: '\\mathrm{KL} \\le \\log n - \\tfrac{n - 1}{n}',
    summary: 'The usual best-of-n KL formula, an upper bound on the divergence from the base policy.',
    role: 'property',
    notes,
    cite: ['beirami2025'],
  },
  preferences.bestOfNKl,
)
fn(
  {
    key: 'bestOfNWeights',
    name: 'Best-of-n rank probabilities',
    role: 'property',
    notes,
  },
  preferences.bestOfNWeights,
)
fn(
  {
    key: 'bestOfNCurve',
    name: 'Best-of-n over-optimisation curve',
    summary: 'Exact expected proxy and gold rewards of the best of n from a pool, against the KL bound.',
    role: 'property',
    notes,
    cite: ['gao2023overoptimisation', 'beirami2025'],
  },
  preferences.bestOfNCurve,
)

type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
/** The functions of the module, keyed by name. */
export const rewardModelFunctions: Table = entries<FunctionInfo>('function', preferences) as Table
