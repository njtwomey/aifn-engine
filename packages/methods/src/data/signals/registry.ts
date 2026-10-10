/**
 * The registry entries of the deterministic signals of `signals.ts` (kind `function`, area `data/signals`): each
 * function with its display name, its role and the notes that use it, collected in `signalGeneratorFunctions`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as signals from './signals'

/** Registers a function of `signals.ts` (kind `function`, area `data/signals`). */
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

/** The registered functions of `signals.ts` (`uniformTimes`, `chirp`, `tones`), keyed by name. */
export const signalGeneratorFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', signals) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
