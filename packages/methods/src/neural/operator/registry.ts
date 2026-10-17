/**
 * The registry of `aifn-methods/neural/operator`: the linear operator fits, DeepONet, the Fourier neural operator and
 * the mesh message passing, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as deeponet from './deeponet'
import * as fno from './fno'
import * as linear from './linear'
import * as mesh from './mesh'

/** A registry table: each registered export of the module by name, with its description. */
type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

const fn = definer<FunctionInfo>('function', 'neural/operator')
const PINO = 'physics-informed-neural-operator'

fn(
  {
    key: 'fitDenseOperator',
    name: 'Dense linear operator fit',
    tex: '\\min_G \\sum_i \\lVert a_i G - u_i \\rVert^2 + \\lambda \\lVert G \\rVert_F^2',
    role: 'fit',
    summary: 'An unrestricted linear map on grid values by ridge least squares, the structure-free baseline.',
    notes: ['operator-learning'],
  },
  linear.fitDenseOperator,
)
fn(
  {
    key: 'fitFourierMultiplier',
    name: 'Fourier multiplier fit',
    tex: 'm_k = \\sum_i \\overline{A_{ik}} B_{ik} / (\\sum_i |A_{ik}|^2 + \\lambda)',
    role: 'fit',
    summary: 'A translation-equivariant linear operator, one complex multiplier per mode, by least squares.',
    notes: ['operator-learning', 'fourier-neural-operator'],
    cite: ['li2021fno'],
  },
  linear.fitFourierMultiplier,
)
fn(
  {
    key: 'applyFourierMultiplier',
    name: 'Apply a Fourier multiplier',
    role: 'transform',
    notes: ['operator-learning'],
    cite: ['li2021fno'],
  },
  linear.applyFourierMultiplier,
)
fn(
  {
    key: 'fitPhysicsInformedMultiplier',
    name: 'Physics-informed multiplier fit',
    role: 'fit',
    summary: 'A space-time heat multiplier fitted to solved pairs and to the equation’s residual on unsolved inputs.',
    notes: [PINO],
    cite: ['li2024pino'],
  },
  linear.fitPhysicsInformedMultiplier,
)
fn(
  { key: 'applyModeMultiplier', name: 'Apply a mode multiplier to a series', role: 'transform', notes: [PINO] },
  linear.applyModeMultiplier,
)
fn(
  {
    key: 'deepOnet',
    name: 'DeepONet',
    tex: 'G(u)(y) \\approx \\sum_k b_k(u) t_k(y) + b_0',
    role: 'construction',
    summary: 'A branch network on sensor values times a trunk network on the query point, the unstacked DeepONet.',
    notes: ['deeponet'],
    cite: ['lu2021deeponet'],
  },
  deeponet.deepOnet,
)
fn(
  { key: 'deepOnetLoss', name: 'DeepONet loss on aligned data', role: 'construction', notes: ['deeponet'] },
  deeponet.deepOnetLoss,
)
fn(
  {
    key: 'FourierNeuralOperator1d',
    name: 'Fourier neural operator (1-D)',
    tex: 'v \\leftarrow \\sigma(\\mathcal{K} v + W v + b)',
    role: 'construction',
    summary: 'A lift, Fourier layers mixing the lowest modes globally, and a pointwise projection; resolution-free.',
    notes: ['fourier-neural-operator'],
    cite: ['li2021fno'],
  },
  fno.FourierNeuralOperator1d,
)
fn(
  {
    key: 'messageWeights',
    name: 'Mesh message weights',
    tex: 'w_{i,i\\pm1} = 2 / (h_\\pm (h_- + h_+))',
    role: 'construction',
    notes: ['graph-and-mesh-neural-operators'],
    cite: ['leveque2007'],
  },
  mesh.messageWeights,
)
fn(
  {
    key: 'meshHeatRollout',
    name: 'Heat equation by mesh message passing',
    role: 'simulation',
    summary: 'Explicit message passing of the heat equation on an irregular mesh, with geometric or uniform weights.',
    notes: ['graph-and-mesh-neural-operators'],
    cite: ['leveque2007'],
  },
  mesh.meshHeatRollout,
)

/** The functions of the module, keyed by name. */
export const operatorFunctions: Table = entries<FunctionInfo>('function', deeponet, fno, linear, mesh) as Table
