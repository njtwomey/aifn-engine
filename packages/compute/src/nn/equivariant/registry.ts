/** The functions of `aifn-compute/nn/equivariant`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as p4 from './p4'

const fn = definer<FunctionInfo>('function', 'nn/equivariant')
const GCNN = ['group-equivariant-convolutional-network']

fn({ key: 'rot90', name: 'Quarter-turn rotation', role: 'transform', notes: GCNN }, p4.rot90)
fn(
  {
    key: 'p4Act',
    name: 'Rotation acting on a p4 map',
    tex: '(r \\cdot f)(s) = \\mathrm{rot}^k f(s - k)',
    summary: 'Rotate every orientation slice and shift the orientations cyclically.',
    role: 'transform',
    notes: GCNN,
    cite: ['cohen2016gcnn'],
  },
  p4.p4Act,
)
fn(
  {
    key: 'p4Lift',
    name: 'Lifting correlation',
    tex: '[f \\star \\psi](g) = \\sum_y f(y)\\, \\psi(g^{-1} y)',
    summary: 'Correlate an image with the four rotated copies of each filter: a map over translations and rotations.',
    role: 'transform',
    notes: GCNN,
    cite: ['cohen2016gcnn'],
  },
  p4.p4Lift,
)
fn(
  {
    key: 'p4GroupConv',
    name: 'p4 group correlation',
    tex: '[f \\star \\psi](g) = \\sum_{h \\in p4} f(h)\\, \\psi(g^{-1} h)',
    summary: 'Correlate a map on the group with filters on the group: equivariant to rotations and translations.',
    role: 'transform',
    notes: GCNN,
    cite: ['cohen2016gcnn'],
  },
  p4.p4GroupConv,
)
fn(
  {
    key: 'orientationPool',
    name: 'Orientation pooling',
    summary: 'The maximum over orientations: a map that rotates with the input.',
    role: 'transform',
    notes: GCNN,
    cite: ['cohen2016gcnn'],
  },
  p4.orientationPool,
)

/** The functions of the module, keyed by name. */
export const equivariantFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', p4) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
