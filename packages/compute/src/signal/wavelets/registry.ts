/**
 * The registry of wavelets (design S §2.13): the orthogonal Daubechies family (Haar = db1 to db10), each as a function
 * returning its four filters with its vanishing moments and filter length, and the continuous Morlet wavelet with its
 * centre frequency. The "wavelet families" note and the lab's wavelet picker enumerate this table.
 */

import { definer, entries, type Entry, type WaveletInfo, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as wavelets from './wavelets'
import { real, space } from 'aifn-compute/foundation/space'
import { morlet, waveletFilters, type WaveletName } from './wavelets'

const define = definer<WaveletInfo>('wavelet', 'signal/wavelets')
const none = space({})
const orthogonalNotes = ['wavelet-families', 'discrete-wavelet-transform']

/**
 * The registry entry of the Daubechies wavelet `dbN`, whose function returns its filters (`waveletFilters`).
 *
 * @param n The number of vanishing moments $N$, 1 to 10; the filters have $2N$ taps.
 * @returns The entry, keyed `dbN`.
 */
const daubechies = (n: number): Entry<() => ReturnType<typeof waveletFilters>, WaveletInfo> => {
  const key = `db${n}` as WaveletName
  return define(
    {
      key,
      name: `Daubechies ${n}`,
      family: 'daubechies',
      continuous: false,
      orthogonal: true,
      vanishingMoments: n,
      taps: 2 * n,
      params: none,
      cite: ['daubechies1988', 'mallat1989'],
      notes: orthogonalNotes,
    },
    () => waveletFilters(key),
  )
}

/** Every wavelet, keyed by the name the transforms take (`haar`, `db1` … `db10`, `morlet`). */
export const waveletRegistry: Readonly<Record<string, Entry<(...args: never[]) => unknown, WaveletInfo>>> =
  entries<WaveletInfo>('wavelet', {
    haar: define(
      {
        key: 'haar',
        name: 'Haar',
        family: 'haar',
        continuous: false,
        orthogonal: true,
        vanishingMoments: 1,
        taps: 2,
        params: none,
        cite: ['mallat1989'],
        notes: orthogonalNotes,
      },
      () => waveletFilters('haar'),
    ),
    ...Object.fromEntries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => [`db${n}`, daubechies(n)])),
    morlet: define(
      {
        key: 'morlet',
        name: 'Morlet',
        family: 'morlet',
        continuous: true,
        orthogonal: false,
        params: space({
          omega0: real(4, 12, { default: 6, label: '\\omega_0', doc: 'centre frequency (radians per unit scale)' }),
        }),
        cite: ['torrence1998'],
        notes: ['continuous-wavelet-transform', 'wavelet-families'],
      },
      morlet,
    ),
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, WaveletInfo>>>

const fn = definer<FunctionInfo>('function', 'signal/wavelets')
const DWT = ['discrete-wavelet-transform', 'multiresolution-analysis']

fn(
  {
    key: 'waveletFilters',
    name: 'Wavelet filter bank',
    role: 'construction',
    notes: ['wavelet-families', 'perfect-reconstruction-filter-banks'],
  },
  wavelets.waveletFilters,
)
fn(
  { key: 'dwt', name: 'Discrete wavelet transform (one level)', role: 'transform', notes: DWT, cite: ['mallat1989'] },
  wavelets.dwt,
)
fn({ key: 'idwt', name: 'Inverse DWT (one level)', role: 'transform', notes: DWT }, wavelets.idwt)
fn(
  {
    key: 'wavedec',
    name: 'Multilevel wavelet decomposition',
    role: 'transform',
    notes: [...DWT, 'wavelet-denoising'],
    cite: ['mallat1989'],
  },
  wavelets.wavedec,
)
fn({ key: 'waverec', name: 'Multilevel wavelet reconstruction', role: 'transform', notes: DWT }, wavelets.waverec)
fn(
  {
    key: 'wavefun',
    name: 'Scaling and wavelet functions',
    summary: 'φ and ψ by the cascade algorithm.',
    role: 'construction',
    notes: ['wavelet-families', 'multiresolution-analysis'],
    cite: ['daubechies1992'],
  },
  wavelets.wavefun,
)
fn(
  {
    key: 'cwt',
    name: 'Continuous wavelet transform (Morlet)',
    role: 'transform',
    notes: ['continuous-wavelet-transform'],
    cite: ['torrence1998'],
  },
  wavelets.cwt,
)

fn(
  {
    key: 'waveletThreshold',
    name: 'Soft and hard thresholding',
    role: 'transform',
    notes: ['wavelet-denoising'],
    cite: ['donoho1994'],
  },
  wavelets.waveletThreshold,
)
fn(
  {
    key: 'waveletDenoise',
    name: 'Wavelet shrinkage denoising',
    summary: 'Threshold the detail coefficients at σ̂√(2 ln n) and reconstruct.',
    role: 'transform',
    notes: ['wavelet-denoising', 'discrete-wavelet-transform'],
    cite: ['donoho1994'],
  },
  wavelets.waveletDenoise,
)

/** The functions of the module, keyed by name. */
export const waveletsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', wavelets) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
