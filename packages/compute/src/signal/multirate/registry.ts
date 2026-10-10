/**
 * The functions of `aifn-compute/signal/multirate`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as multirate from './multirate'

/** Registers a function of `signal/multirate` with its metadata. */
const fn = definer<FunctionInfo>('function', 'signal/multirate')

fn(
  {
    key: 'resamplePoly',
    name: 'Polyphase resampling',
    summary: 'Upsample by L, low-pass, downsample by M, computed in polyphase form.',
    role: 'transform',
    returns: 'signal',
    notes: ['sample-rate-conversion', 'polyphase-decomposition', 'decimation-and-interpolation'],
    cite: ['vaidyanathan1993'],
  },
  multirate.resamplePoly,
)
fn(
  {
    key: 'decimateSignal',
    name: 'Decimate',
    role: 'transform',
    returns: 'signal',
    notes: ['decimation-and-interpolation', 'aliasing'],
  },
  multirate.decimateSignal,
)
fn(
  {
    key: 'polyphase',
    name: 'Polyphase components',
    role: 'transform',
    notes: ['polyphase-decomposition'],
    cite: ['vaidyanathan1993'],
  },
  multirate.polyphase,
)
fn(
  {
    key: 'dftFilterBank',
    name: 'DFT filter bank',
    role: 'transform',
    notes: ['filter-banks', 'polyphase-decomposition'],
    cite: ['vaidyanathan1993'],
  },
  multirate.dftFilterBank,
)

fn(
  {
    key: 'sincInterpolate',
    name: 'Sinc (Whittaker–Shannon) interpolation',
    tex: 'x(t) = \\sum_n x[n] \\operatorname{sinc}(f_s t - n)',
    role: 'transform',
    notes: ['sampling-theorem', 'aliasing'],
    cite: ['shannon1949'],
  },
  multirate.sincInterpolate,
)

/**
 * The functions of the module, keyed by name, each with its registry `info` (name, role, notes, citations).
 *
 * @example The registered functions
 * print('keys =', Object.keys(multirateFunctions))
 * print('resamplePoly =', multirateFunctions.resamplePoly.info.name)
 */
export const multirateFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', multirate) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
