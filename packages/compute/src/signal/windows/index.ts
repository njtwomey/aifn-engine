/**
 * `aifn-compute/signal/windows`: window functions (`getWindow`: hann, hamming, blackman, kaiser, gaussian, tukey, …), and
 * `windowRegistry`: every window as a function of its length, with its main-lobe width and peak side-lobe level.
 */

export { getWindow, windowValues, type WindowInput, type WindowName, type WindowSpec } from './windows'
export { windowRegistry, type WindowFunction } from './registry'
