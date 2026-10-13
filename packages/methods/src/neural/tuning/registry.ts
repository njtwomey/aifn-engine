/** The registry of `aifn-methods/neural/tuning`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as sweep from './learning-rate-sweep'

const fn = definer<FunctionInfo>('function', 'neural/tuning')

fn(
  {
    key: 'learningRateSweep',
    name: 'Learning-rate sweep',
    summary: 'Train a small network once per learning rate and seed; the validation loss of every run.',
    role: 'simulation',
    random: true,
    cite: ['bengio2012practical', 'smith2017clr'],
  },
  sweep.learningRateSweep,
)

/** The functions of the module, keyed by name. */
export const tuningFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', sweep) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
