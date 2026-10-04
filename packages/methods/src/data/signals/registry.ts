/** The signal generators of `aifn-methods/data/signals`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as signals from './signals'

const fn = definer<FunctionInfo>('function', 'data/signals')

fn(
  { key: 'uniformTimes', name: 'Uniform sample times', role: 'construction', notes: ['sampling-theorem'] },
  signals.uniformTimes,
)
fn(
  {
    key: 'chirp',
    name: 'Chirp',
    summary: 'A sinusoid whose frequency sweeps linearly, quadratically or logarithmically.',
    role: 'construction',
    notes: ['short-time-fourier-transform', 'instantaneous-frequency'],
  },
  signals.chirp,
)
fn(
  {
    key: 'tones',
    name: 'Sum of tones',
    role: 'construction',
    notes: ['discrete-fourier-transform', 'spectral-leakage-and-windows'],
  },
  signals.tones,
)

/** The functions of the module, keyed by name. */
export const signalGeneratorFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', signals) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
