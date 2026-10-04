/**
 * The functions of `aifn-compute/foundation/fourier` that are not primitives (`fft`, `ifft`, `rfft`, `irfft` are listed in
 * the primitive table), registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as complex from './complex'
import * as dct from './dct'
import * as fft from './fft'

const fn = definer<FunctionInfo>('function', 'foundation/fourier')
const DFT = ['discrete-fourier-transform']

fn(
  {
    key: 'dft',
    name: 'DFT by its matrix',
    summary: 'The DFT as the product with the DFT matrix: the definition, O(n²).',
    role: 'transform',
    notes: [...DFT, 'fast-fourier-transform'],
  },
  fft.dft,
)
fn(
  { key: 'dftMatrix', name: 'DFT matrix', tex: 'F_{kt} = e^{-2\\pi i kt/n}', role: 'construction', notes: DFT },
  fft.dftMatrix,
)
fn(
  {
    key: 'fft2',
    name: 'Two-dimensional FFT',
    role: 'transform',
    notes: ['fast-fourier-transform', 'fourier-transform'],
  },
  fft.fft2,
)
fn(
  { key: 'ifft2', name: 'Two-dimensional inverse FFT', role: 'transform', notes: ['fast-fourier-transform'] },
  fft.ifft2,
)
fn(
  {
    key: 'fftfreq',
    name: 'DFT sample frequencies',
    role: 'construction',
    notes: [...DFT, 'zero-padding-and-resolution'],
  },
  fft.fftfreq,
)
fn({ key: 'rfftfreq', name: 'Real-DFT sample frequencies', role: 'construction', notes: DFT }, fft.rfftfreq)
fn({ key: 'fftshift', name: 'Shift zero frequency to the centre', role: 'transform', notes: DFT }, fft.fftshift)
fn({ key: 'ifftshift', name: 'Undo fftshift', role: 'transform', notes: DFT }, fft.ifftshift)
fn(
  {
    key: 'dct',
    name: 'Discrete cosine transform (DCT-II)',
    role: 'transform',
    notes: ['mel-frequency-cepstral-coefficients', 'perceptual-audio-coding'],
    cite: ['ahmed1974'],
  },
  dct.dct,
)
fn({ key: 'idct', name: 'Inverse DCT', role: 'transform', notes: ['mel-frequency-cepstral-coefficients'] }, dct.idct)
fn({ key: 'dctMatrix', name: 'DCT matrix', role: 'construction', cite: ['ahmed1974'] }, dct.dctMatrix)
fn(
  { key: 'decibels', name: 'Decibels', tex: '10 \\log_{10} P', role: 'transform', notes: ['frequency-response'] },
  complex.decibels,
)

/** The functions of the module, keyed by name. */
export const fourierFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', fft, dct, complex) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
