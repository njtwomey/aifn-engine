/**
 * The functions of `aifn-compute/signal/time-frequency`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cqt from './cqt'
import * as hilbert from './hilbert'
import * as reassign from './reassign'
import * as wigner from './wigner'

const fn = definer<FunctionInfo>('function', 'signal/time-frequency')

fn(
  {
    key: 'hilbert',
    name: 'Analytic signal (Hilbert transform)',
    summary: 'x + iH{x}, by zeroing the negative frequencies of the DFT.',
    role: 'transform',
    notes: ['hilbert-transform', 'instantaneous-frequency'],
  },
  hilbert.hilbert,
)
fn(
  {
    key: 'instantaneous',
    name: 'Instantaneous amplitude, phase and frequency',
    role: 'transform',
    notes: ['instantaneous-frequency', 'hilbert-transform'],
    cite: ['boashash1992'],
  },
  hilbert.instantaneous,
)
fn(
  { key: 'envelope', name: 'Envelope', role: 'transform', notes: ['hilbert-transform', 'instantaneous-frequency'] },
  hilbert.envelope,
)
fn(
  {
    key: 'hilbertSpectrum',
    name: 'Hilbert spectrum',
    summary: 'The instantaneous frequencies and amplitudes of a decomposition’s modes on a time–frequency grid.',
    role: 'transform',
    returns: 'time-frequency',
    notes: ['hilbert-huang-transform'],
    cite: ['huang1998'],
  },
  hilbert.hilbertSpectrum,
)
fn(
  {
    key: 'cqt',
    name: 'Constant-Q transform',
    role: 'transform',
    returns: 'time-frequency',
    notes: ['constant-q-transform', 'chroma-features'],
    cite: ['brown1991'],
  },
  cqt.cqt,
)

const WVD = ['wigner-ville-distribution', 'time-frequency-uncertainty']
fn(
  {
    key: 'wignerVille',
    name: 'Wigner–Ville distribution',
    tex: 'W_z(t, f) = \\sum_\\tau z[t + \\tau]\\, z^*[t - \\tau]\\, e^{-i 4 \\pi f \\tau}',
    summary: 'A quadratic distribution with perfect concentration on chirps and cross-terms between components.',
    role: 'transform',
    returns: 'time-frequency',
    notes: WVD,
    cite: ['cohen1989'],
  },
  wigner.wignerVille,
)
fn(
  {
    key: 'pseudoWignerVille',
    name: 'Pseudo Wigner–Ville distribution',
    summary: 'The Wigner–Ville distribution with a lag window: smoothed along frequency.',
    role: 'transform',
    returns: 'time-frequency',
    notes: WVD,
    cite: ['cohen1989'],
  },
  wigner.pseudoWignerVille,
)
fn(
  {
    key: 'smoothedPseudoWignerVille',
    name: 'Smoothed pseudo Wigner–Ville distribution',
    summary: 'Separable lag and time smoothing: a Cohen-class kernel that suppresses most cross-terms.',
    role: 'transform',
    returns: 'time-frequency',
    notes: WVD,
    cite: ['cohen1989'],
  },
  wigner.smoothedPseudoWignerVille,
)
fn(
  {
    key: 'reassignedSpectrogram',
    name: 'Reassigned spectrogram',
    summary: 'Spectrogram values moved to the local centre of gravity of the energy.',
    role: 'transform',
    returns: 'time-frequency',
    notes: ['reassignment-and-synchrosqueezing', 'short-time-fourier-transform'],
    cite: ['auger1995'],
  },
  reassign.reassignedSpectrogram,
)
fn(
  {
    key: 'synchrosqueeze',
    name: 'Synchrosqueezed STFT',
    summary: 'STFT coefficients moved along frequency to their instantaneous frequency; invertible.',
    role: 'transform',
    returns: 'time-frequency',
    notes: ['reassignment-and-synchrosqueezing', 'instantaneous-frequency'],
    cite: ['daubechies2011'],
  },
  reassign.synchrosqueeze,
)

/** The functions of the module, keyed by name. */
export const timeFrequencyFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', hilbert, cqt, wigner, reassign) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
