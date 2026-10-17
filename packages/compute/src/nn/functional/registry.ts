/**
 * The functions of `aifn-compute/nn/functional` that have notes of their own, registered with them: the cost and
 * receptive-field arithmetic of convolutional networks, im2col and deformable convolution.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cost from './cost'
import * as deformable from './deformable'
import * as ops from './ops'
import * as receptive from './receptive'

const fn = definer<FunctionInfo>('function', 'nn/functional')
const ARCHITECTURES = [
  'classic-convolutional-network-architectures',
  'inception-networks',
  'densely-connected-network',
  'efficient-convolutional-networks',
  'modern-convolutional-networks',
  'convolution-versus-attention-in-vision',
  'residual-network',
]

fn(
  {
    key: 'convolutionCost',
    name: 'Cost of a convolution layer',
    summary:
      'Parameters, multiply-accumulates and output shape of an ordinary, grouped, depthwise or transposed convolution.',
    tex: "\\lfloor (n + 2p - d(k - 1) - 1)/s \\rfloor + 1,\\ C C' K / g",
    role: 'property',
    notes: [
      'one-by-one-convolution',
      'depthwise-and-grouped-convolution',
      'video-and-three-dimensional-convolution',
      'fully-convolutional-network-and-u-net',
    ],
    cite: ['dumoulin2016', 'howard2017mobilenets'],
  },
  cost.convolutionCost,
)
fn(
  {
    key: 'convolutionStackCost',
    name: 'Cost of a chain of convolutions',
    summary: 'Each layer’s cost with its input and output, and the totals, for convolutions fed one into the next.',
    role: 'property',
    notes: ['one-by-one-convolution'],
    cite: ['dumoulin2016'],
  },
  cost.convolutionStackCost,
)
fn(
  {
    key: 'layerCost',
    name: 'Cost of a layer or block',
    summary: 'Parameters, multiply-accumulates, output shape, depth and receptive field of one layer spec or block.',
    role: 'property',
    notes: ARCHITECTURES,
    cite: ['dumoulin2016', 'araujo2019'],
  },
  cost.layerCost,
)
fn(
  {
    key: 'networkCost',
    name: 'Cost of a convolutional network',
    summary:
      'Shape propagation through a tree of layers with sums, concatenations and gates: parameters, MACs, depth, field.',
    role: 'property',
    notes: ARCHITECTURES,
    cite: ['dumoulin2016', 'araujo2019'],
  },
  cost.networkCost,
)
fn(
  {
    key: 'receptiveField',
    name: 'Receptive field of a chain of layers',
    summary: 'Size, jump and start of the receptive field after each convolution or pooling layer, along one axis.',
    tex: 'r_l = r_{l-1} + (e_l - 1) j_{l-1},\\ j_l = j_{l-1} s_l',
    role: 'property',
    notes: ['receptive-field-and-dilated-convolution', 'wavenet-and-causal-convolution', 'segmentation-decoders'],
    cite: ['araujo2019', 'dumoulin2016', 'yu2016'],
  },
  receptive.receptiveField,
)
fn(
  {
    key: 'effectiveReceptiveField',
    name: 'Effective receptive field',
    summary: 'The gradient of one output unit with respect to the input, by reverse mode.',
    tex: '\\partial y_{i} / \\partial x',
    role: 'transform',
    notes: ['receptive-field-and-dilated-convolution'],
    cite: ['luo2016'],
  },
  receptive.effectiveReceptiveField,
)
fn(
  {
    key: 'dilatedTapCoverage',
    name: 'Dilated tap coverage',
    summary: 'The share of positions of a padded map at which each number of a dilated kernel’s taps falls inside.',
    role: 'property',
    notes: ['segmentation-decoders'],
    cite: ['chen2017deeplab'],
  },
  receptive.dilatedTapCoverage,
)
fn(
  {
    key: 'meanValidTaps',
    name: 'Mean taps inside the map',
    summary: 'The mean number of a dilated kernel’s taps that land inside a zero-padded map.',
    role: 'property',
    notes: ['segmentation-decoders'],
    cite: ['chen2017deeplab'],
  },
  receptive.meanValidTaps,
)
fn(
  {
    key: 'im2col',
    name: 'im2col',
    summary: 'Every convolution window copied into one column, so the layer becomes one matrix product.',
    role: 'transform',
    notes: ['fast-convolution-algorithms'],
    cite: ['chellapilla2006'],
  },
  ops.im2col,
)
fn(
  {
    key: 'deformableConv2d',
    name: 'Deformable convolution',
    summary: 'A 2-D convolution whose taps read at learned offsets by bilinear interpolation, optionally modulated.',
    tex: 'y(p_0) = \\sum_n w_n m_n\\, x(p_0 + p_n + \\Delta p_n)',
    role: 'transform',
    notes: ['deformable-convolution'],
    cite: ['dai2017deformable', 'zhu2019deformable'],
  },
  deformable.deformableConv2d,
)

/** The registered functions of the module, keyed by name. */
export const functionalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', cost, receptive, { im2col: ops.im2col }, deformable) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
