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

/** A window of length n; parameterised windows take their parameter in the options. */
export type WindowFunction = (n: Size, options?: { periodic?: boolean } & Record<string, number | boolean>) => Tensor

const define = definer<WindowInfo>('window', 'signal/windows')
const cite = ['harris1978']
const notes = ['spectral-leakage-and-windows']
const none = space({})

const plain = (key: WindowName, name: string, mainLobeWidth: number, sideLobeDb: number) =>
  define({ key, name, params: none, mainLobeWidth, sideLobeDb, cite, notes }, ((n, o = {}) =>
    getWindow(key, n, { periodic: o.periodic === true })) as WindowFunction)

/** Every window, keyed by the name `getWindow` takes. */
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
