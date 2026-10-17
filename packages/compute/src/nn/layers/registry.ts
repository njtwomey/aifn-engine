/**
 * The layers and functional forms of `aifn-compute/nn/layers` that have notes of their own, registered with them:
 * layer constructors as constructions, functional forms as transforms.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as layers from './layers'
import * as spectral from './spectral'

const fn = definer<FunctionInfo>('function', 'nn/layers')
const NORM = ['normalisation-in-convolutional-networks']

fn(
  {
    key: 'groupNorm',
    name: 'Group normalisation',
    summary: 'Each example’s channels split into G groups, each standardised over its channels and positions.',
    tex: '(x - \\mu_g)/\\sqrt{\\sigma_g^2 + \\varepsilon}\\cdot\\gamma_c + \\beta_c',
    role: 'transform',
    notes: NORM,
    cite: ['wu2018groupnorm'],
  },
  layers.groupNorm,
)
fn(
  {
    key: 'instanceNorm',
    name: 'Instance normalisation',
    summary: 'Each channel of each example standardised over its positions: group normalisation with G = C.',
    role: 'transform',
    notes: NORM,
    cite: ['ulyanov2016instance'],
  },
  layers.instanceNorm,
)
fn(
  {
    key: 'GroupNorm',
    name: 'Group normalisation layer',
    summary: 'Group normalisation with a learned per-channel scale and shift, as torch.nn.GroupNorm.',
    role: 'construction',
    notes: NORM,
    cite: ['wu2018groupnorm'],
  },
  layers.GroupNorm,
)
fn(
  {
    key: 'InstanceNorm',
    name: 'Instance normalisation layer',
    summary: 'Instance normalisation with a learned per-channel scale and shift, without running statistics.',
    role: 'construction',
    notes: NORM,
    cite: ['ulyanov2016instance'],
  },
  layers.InstanceNorm,
)

fn(
  {
    key: 'SpectralConv1d',
    name: 'Spectral convolution (Fourier layer)',
    summary: 'The lowest Fourier modes of each channel mixed by learned complex matrices, the rest set to zero.',
    tex: 'v \\mapsto \\mathcal{F}^{-1}(R_k (\\mathcal{F} v)_k)',
    role: 'construction',
    notes: ['fourier-neural-operator'],
    cite: ['li2021fno'],
  },
  spectral.SpectralConv1d,
)

/** The registered functions and layers of the module, keyed by name. */
export const layerFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', layers, spectral) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
