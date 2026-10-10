/**
 * `aifn-compute/signal/windows`: window functions for filter design and spectral analysis, as
 * `scipy.signal.windows`.
 *
 * - Building a window: `getWindow` (by name, such as hann, hamming, blackman, flattop, or with a parameter, such as
 *   kaiser with $\beta$, gaussian, tukey) and `windowValues` (a spec built, or explicit values checked for length, as
 *   the spectral functions take their `window` option).
 * - Choosing one: `windowRegistry`, every window as a function of its length, with its main-lobe width (DFT bins) and
 *   peak side-lobe level (dB).
 *
 * Windows are symmetric by default, for filter design; `periodic: true` gives the DFT-even form for spectral analysis.
 */

export { getWindow, windowValues, type WindowInput, type WindowName, type WindowSpec } from './windows'
export { windowRegistry, type WindowFunction } from './registry'
