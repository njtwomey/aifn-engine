/** The functions of `aifn-compute/nn/sets`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as sets from './sets'

const fn = definer<FunctionInfo>('function', 'nn/sets')
const SETS = ['deep-sets', 'point-cloud-networks']

fn(
  {
    key: 'padSets',
    name: 'Padded batch of sets',
    summary: 'Sets of different sizes stacked into one tensor with a mask of the real elements.',
    role: 'construction',
    notes: SETS,
  },
  sets.padSets,
)
fn(
  {
    key: 'setPool',
    name: 'Set pooling',
    tex: 'r_b = \\bigoplus_{k \\in S_b} h_{bk}',
    summary: 'A masked sum, mean or maximum over each set: invariant to the order of its elements.',
    role: 'transform',
    notes: SETS,
    cite: ['zaheer2017', 'qi2017pointnet'],
  },
  sets.setPool,
)
fn(
  {
    key: 'criticalSet',
    name: 'Critical point set',
    summary: 'The elements that attain a max-pool in at least one channel: they alone fix the pooled output.',
    role: 'property',
    notes: ['point-cloud-networks'],
    cite: ['qi2017pointnet'],
  },
  sets.criticalSet,
)

/** The functions of the module, keyed by name. */
export const setsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', sets) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
