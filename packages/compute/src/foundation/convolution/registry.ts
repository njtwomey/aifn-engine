/**
 * The functions of `aifn-compute/foundation/convolution` that are not primitives (`conv`, `convTranspose`, `pad` and
 * `linearFilter` are listed in the primitive table), registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as conv from './conv'
import * as convolution from './convolution'
import * as image from './image'

const fn = definer<FunctionInfo>('function', 'foundation/convolution')

fn(
  {
    key: 'convolve',
    name: 'Convolution',
    tex: '(x * h)[n]',
    role: 'transform',
    notes: ['convolution', 'linear-time-invariant-systems'],
    cite: ['oppenheim2010'],
  },
  convolution.convolve,
)
fn(
  {
    key: 'fftConvolve',
    name: 'Convolution by FFT',
    summary: 'Linear convolution as the inverse DFT of the product of zero-padded DFTs.',
    role: 'transform',
    notes: ['convolution-theorem', 'convolution', 'overlap-add-and-overlap-save'],
  },
  convolution.fftConvolve,
)
fn(
  { key: 'correlate', name: 'Cross-correlation', role: 'transform', notes: ['convolution', 'matched-filter'] },
  convolution.correlate,
)
fn(
  { key: 'correlationLags', name: 'Correlation lags', role: 'construction', notes: ['convolution'] },
  convolution.correlationLags,
)
fn(
  {
    key: 'upfirdn',
    name: 'Upsample, FIR filter, downsample',
    role: 'transform',
    notes: ['polyphase-decomposition', 'sample-rate-conversion'],
    cite: ['vaidyanathan1993'],
  },
  convolution.upfirdn,
)
fn(
  { key: 'convOutputSize', name: 'Convolution output size', role: 'property', notes: ['convolution'] },
  conv.convOutputSize,
)
fn(
  { key: 'correlate2d', name: 'Two-dimensional correlation', role: 'transform', notes: ['convolution'] },
  image.correlate2d,
)
fn(
  { key: 'convolve2d', name: 'Two-dimensional convolution', role: 'transform', notes: ['convolution'] },
  image.convolve2d,
)
fn(
  { key: 'separableFilter', name: 'Separable two-dimensional filter', role: 'transform', notes: ['convolution'] },
  image.separableFilter,
)

/** The functions of the module, keyed by name. */
export const convolutionFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', conv, convolution, image) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
