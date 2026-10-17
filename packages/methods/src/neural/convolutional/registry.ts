/**
 * The registry of `aifn-methods/neural/convolutional`: the Winograd transforms and tiles, the cost model of fast
 * convolution, the deformable convolution at one position and capsule routing, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as capsules from './capsules'
import * as deformable from './deformable'
import * as fast from './fast'

const fn = definer<FunctionInfo>('function', 'neural/convolutional')
const FAST = 'fast-convolution-algorithms'

fn(
  {
    key: 'winogradPoints',
    name: 'Winograd interpolation points',
    summary: 'The default Toom-Cook points 0, ±1, ±2, ±1/2, ±3, ±1/3, …, Lavin’s choice for F(4, 3) and F(6, 3).',
    role: 'construction',
    notes: [FAST],
    cite: ['lavin2016'],
  },
  fast.winogradPoints,
)
fn(
  {
    key: 'winogradTransforms',
    name: 'Winograd minimal filtering transforms',
    summary:
      'Aᵀ, G and Bᵀ of F(m, r) by the Toom-Cook construction: m outputs of an r-tap filter from m + r − 1 products.',
    tex: 'y = A^\\top[(G g) \\odot (B^\\top d)]',
    role: 'construction',
    notes: [FAST],
    cite: ['winograd1980', 'lavin2016'],
  },
  fast.winogradTransforms,
)
fn(
  {
    key: 'winogradTile',
    name: 'Winograd 2-D tile',
    summary: 'The m × m outputs of one data tile by F(m × m, r × r), in float64 or emulated float32.',
    tex: 'Y = A^\\top[(G g G^\\top) \\odot (B^\\top d B)]A',
    role: 'transform',
    notes: [FAST],
    cite: ['lavin2016'],
  },
  fast.winogradTile,
)
fn(
  {
    key: 'directTile',
    name: 'Direct correlation of a tile',
    summary: 'The m × m outputs of the direct cross-correlation of a tile, the reference for the Winograd tile.',
    role: 'transform',
    notes: [FAST],
  },
  fast.directTile,
)
fn(
  {
    key: 'winogradRelativeError',
    name: 'Float32 error of Winograd convolution',
    summary: 'Mean relative error of float32 Winograd and float32 direct tiles against float64, on random tiles.',
    role: 'simulation',
    notes: [FAST],
    cite: ['lavin2016'],
  },
  fast.winogradRelativeError,
)
fn(
  {
    key: 'convolutionAlgorithmCosts',
    name: 'Cost of fast convolution algorithms',
    summary:
      'Real multiplications and workspace of one layer by the direct sum, im2col, FFT and Winograd, as a stated cost model.',
    role: 'property',
    notes: [FAST],
    cite: ['chellapilla2006', 'lavin2016'],
  },
  fast.convolutionAlgorithmCosts,
)
fn(
  {
    key: 'deformableConvolutionAt',
    name: 'Deformable convolution at one position',
    summary:
      'One output of a (modulated) deformable convolution, tap by tap, with each bilinear sample and the offset gradient.',
    tex: 'y(p_0) = \\sum_n w_n m_n\\, x(p_0 + d\\,p_n + \\Delta p_n)',
    role: 'transform',
    notes: ['deformable-convolution'],
    cite: ['dai2017deformable', 'zhu2019deformable'],
  },
  deformable.deformableConvolutionAt,
)
fn(
  {
    key: 'squash',
    name: 'Capsule squash',
    summary: 'A capsule input scaled to length |s|²/(1 + |s|²) in its own direction.',
    tex: 'v = \\frac{\\lVert s \\rVert^2}{1 + \\lVert s \\rVert^2} \\frac{s}{\\lVert s \\rVert}',
    role: 'transform',
    notes: ['capsule-network'],
    cite: ['sabour2017capsules'],
  },
  capsules.squash,
)
fn(
  {
    key: 'capsulePredictions',
    name: 'Capsule predictions',
    summary: 'Each lower capsule’s prediction of each upper capsule, û(j|i) = W(ij) u(i).',
    role: 'transform',
    notes: ['capsule-network'],
    cite: ['sabour2017capsules'],
  },
  capsules.capsulePredictions,
)
fn(
  {
    key: 'dynamicRoutingTrace',
    name: 'Routing by agreement',
    summary:
      'Dynamic routing between capsules, every iteration recorded: couplings, inputs, outputs, agreements, logits.',
    tex: 'c_{ij} = \\operatorname{softmax}_j(b_{ij}),\\ b_{ij} \\mathrel{+}= \\hat u_{j|i} \\cdot v_j',
    role: 'simulation',
    notes: ['capsule-network'],
    cite: ['sabour2017capsules'],
  },
  capsules.dynamicRoutingTrace,
)

/** The registered functions of `aifn-methods/neural/convolutional`, keyed by `info.key`. */
export const convolutionalFunctions: Readonly<Record<string, Entry<unknown, FunctionInfo>>> = entries(
  'function',
  fast,
  deformable,
  capsules,
)
