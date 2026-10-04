/** The functions of `aifn-compute/signal/audio`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as audio from './audio'

const fn = definer<FunctionInfo>('function', 'signal/audio')

fn(
  {
    key: 'hzToMel',
    name: 'Hertz to mel',
    role: 'transform',
    notes: ['auditory-scales', 'mel-filter-bank'],
    cite: ['stevens1937', 'slaney1998'],
  },
  audio.hzToMel,
)
fn({ key: 'melToHz', name: 'Mel to hertz', role: 'transform', notes: ['auditory-scales'] }, audio.melToHz)
fn(
  {
    key: 'melFilterbank',
    name: 'Mel filter bank',
    role: 'construction',
    notes: ['mel-filter-bank', 'mel-spectrogram'],
    cite: ['slaney1998'],
  },
  audio.melFilterbank,
)
fn(
  {
    key: 'mfcc',
    name: 'Mel-frequency cepstral coefficients',
    summary: 'The DCT of log mel-band energies of each frame.',
    role: 'transform',
    notes: ['mel-frequency-cepstral-coefficients', 'cepstrum'],
    cite: ['davis1980'],
  },
  audio.mfcc,
)

/** The functions of the module, keyed by name. */
export const audioFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', audio) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
