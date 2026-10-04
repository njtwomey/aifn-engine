/**
 * The functions of `aifn-compute/signal/cepstrum`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cepstrum from './cepstrum'
import * as yin from './yin'

const fn = definer<FunctionInfo>('function', 'signal/cepstrum')
const CEPSTRUM = ['cepstrum']
const PITCH = ['pitch-estimation']

fn(
  {
    key: 'realCepstrum',
    name: 'Real cepstrum',
    tex: 'c[q] = \\mathcal{F}^{-1}\\{\\log |\\mathcal{F}\\{x\\}|\\}[q]',
    summary: 'The inverse DFT of the log magnitude spectrum: convolution becomes addition.',
    role: 'transform',
    notes: [...CEPSTRUM, 'mel-frequency-cepstral-coefficients'],
    cite: ['childers1977'],
  },
  cepstrum.realCepstrum,
)
fn(
  {
    key: 'complexCepstrum',
    name: 'Complex cepstrum',
    summary: 'The inverse DFT of the complex log spectrum (unwrapped phase, linear phase removed); invertible.',
    role: 'transform',
    notes: CEPSTRUM,
    cite: ['childers1977'],
  },
  cepstrum.complexCepstrum,
)
fn(
  { key: 'inverseComplexCepstrum', name: 'Inverse complex cepstrum', role: 'transform', notes: CEPSTRUM },
  cepstrum.inverseComplexCepstrum,
)
fn(
  {
    key: 'cepstralEnvelope',
    name: 'Cepstral (liftered) spectral envelope',
    summary: 'The log spectrum smoothed by keeping only the low-quefrency cepstrum.',
    role: 'transform',
    notes: [...CEPSTRUM, 'mel-frequency-cepstral-coefficients'],
    cite: ['childers1977'],
  },
  cepstrum.cepstralEnvelope,
)
fn(
  {
    key: 'cepstralPitch',
    name: 'Cepstral pitch',
    summary: 'f₀ from the largest real-cepstrum peak in a quefrency range.',
    role: 'property',
    notes: [...PITCH, ...CEPSTRUM],
    cite: ['noll1967'],
  },
  cepstrum.cepstralPitch,
)
fn(
  {
    key: 'yinDifference',
    name: 'YIN difference function',
    tex: "d'(\\tau) = d(\\tau)\\, \\tau \\big/ \\sum_{u=1}^{\\tau} d(u)",
    role: 'transform',
    notes: PITCH,
    cite: ['decheveigne2002'],
  },
  yin.yinDifference,
)
fn(
  {
    key: 'yinPitch',
    name: 'YIN pitch (one frame)',
    summary: 'The first dip of the cumulative-mean-normalised difference below a threshold, refined by a parabola.',
    role: 'property',
    notes: PITCH,
    cite: ['decheveigne2002'],
  },
  yin.yinPitch,
)
fn({ key: 'yin', name: 'YIN pitch track', role: 'transform', notes: PITCH, cite: ['decheveigne2002'] }, yin.yin)

/** The functions of the module, keyed by name. */
export const cepstrumFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', cepstrum, yin) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
