/**
 * The registry of window functions (design S §2.13): each window as a function of its length (`periodic: true` for
 * spectral analysis), with its parameters and its figures of merit at the default parameters (Harris, 1978): the
 * main-lobe width between the first nulls, in DFT bins, and the peak side-lobe level in dB, both measured on the
 * periodic window. The "spectral leakage and windows" note and the lab's window picker enumerate this table.
 */

import { definer, entries, type Entry, type WindowInfo } from 'aifn-compute/foundation/registry'
import { real, space } from 'aifn-compute/foundation/space'
import type { Size } from 'aifn-compute/foundation/contracts'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { getWindow, type WindowName } from './windows'

/**
 * A window of length $n$ (in samples), as a tensor. The options hold `periodic` (`true` for the periodic window of
 * spectral analysis, otherwise the symmetric one) and a parameterised window's parameter by name (`beta`, `std` or
 * `alpha`, defaulting to the registry's default when left out).
 */
export type WindowFunction = (n: Size, options?: { periodic?: boolean } & Record<string, number | boolean>) => Tensor

/** Registers a window function under `signal/windows`. */
const define = definer<WindowInfo>('window', 'signal/windows')
const cite = ['harris1978']
const notes = ['spectral-leakage-and-windows']
const none = space({})

/**
 * Registers a window that takes no parameter, as `getWindow(key, n)`.
 *
 * @param key The name `getWindow` takes, which is also the registry key.
 * @param name The display name.
 * @param mainLobeWidth The main-lobe width between the first nulls, in DFT bins.
 * @param sideLobeDb The peak side-lobe level, in dB (negative).
 * @returns The registry entry.
 */
const plain = (key: WindowName, name: string, mainLobeWidth: number, sideLobeDb: number) =>
  define({ key, name, params: none, mainLobeWidth, sideLobeDb, cite, notes }, ((n, o = {}) =>
    getWindow(key, n, { periodic: o.periodic === true })) as WindowFunction)

/**
 * Every window, keyed by the name `getWindow` takes (`boxcar` is not listed apart from `rectangular`). Each entry is
 * the window as a function of its length, with its `info`: parameters, and the main-lobe width (DFT bins) and peak
 * side-lobe level (dB) at the default parameters, where known (the Gaussian has none recorded).
 *
 * @example A window and its figures of merit
 * const hann = windowRegistry.hann
 * print('hann(5) =', hann(5))
 * print('main lobe (bins) =', hann.info.mainLobeWidth)
 * print('side lobe (dB) =', hann.info.sideLobeDb)
 */
export const windowRegistry: Readonly<Record<string, Entry<WindowFunction, WindowInfo>>> = entries<WindowInfo>(
  'window',
  {
    rectangular: plain('rectangular', 'Rectangular', 2, -13.3),
    hann: plain('hann', 'Hann', 4, -31.5),
    hamming: plain('hamming', 'Hamming', 4, -42.6),
    blackman: plain('blackman', 'Blackman', 6, -58.1),
    blackmanharris: plain('blackmanharris', 'Blackman–Harris', 8, -92),
    nuttall: plain('nuttall', 'Nuttall', 8, -96.5),
    flattop: plain('flattop', 'Flat top', 10, -91.5),
    bartlett: plain('bartlett', 'Bartlett', 4, -26.5),
    triangular: plain('triangular', 'Triangular', 3.9, -26.5),
    cosine: define(
      { key: 'cosine', name: 'Cosine (sine)', params: none, mainLobeWidth: 3, sideLobeDb: -23, cite, notes },
      ((n, o = {}) => getWindow({ name: 'cosine' }, n, { periodic: o.periodic === true })) as WindowFunction,
    ),
    kaiser: define(
      {
        key: 'kaiser',
        name: 'Kaiser',
        params: space({
          beta: real(0, 20, { default: 8.6, label: '\\beta', doc: 'shape: larger trades width for lower side lobes' }),
        }),
        mainLobeWidth: 5.8,
        sideLobeDb: -63.3,
        cite,
        notes,
      },
      ((n, o = {}) =>
        getWindow({ name: 'kaiser', beta: Number(o.beta ?? 8.6) }, n, {
          periodic: o.periodic === true,
        })) as WindowFunction,
    ),
    gaussian: define(
      {
        key: 'gaussian',
        name: 'Gaussian',
        params: space({
          std: real(0.5, 100, { default: 16, scale: 'log', label: '\\sigma', doc: 'standard deviation, in samples' }),
        }),
        cite,
        notes,
      },
      ((n, o = {}) =>
        getWindow({ name: 'gaussian', std: Number(o.std ?? 16) }, n, {
          periodic: o.periodic === true,
        })) as WindowFunction,
    ),
    tukey: define(
      {
        key: 'tukey',
        name: 'Tukey (tapered cosine)',
        params: space({ alpha: real(0, 1, { default: 0.5, label: '\\alpha', doc: 'tapered fraction' }) }),
        mainLobeWidth: 2.66,
        sideLobeDb: -15.1,
        cite,
        notes,
      },
      ((n, o = {}) =>
        getWindow({ name: 'tukey', alpha: Number(o.alpha ?? 0.5) }, n, {
          periodic: o.periodic === true,
        })) as WindowFunction,
    ),
  },
) as Readonly<Record<string, Entry<WindowFunction, WindowInfo>>>
