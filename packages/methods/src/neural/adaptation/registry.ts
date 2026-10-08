/** The registry of `aifn-methods/neural/adaptation`: the low-rank adaptation target and fit. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as lowRank from './low-rank'

const fn = definer<FunctionInfo>('function', 'neural/adaptation')
const notes = ['low-rank-adaptation']

fn(
  {
    key: 'lowRankTarget',
    name: 'Low-rank adaptation target',
    summary: 'A square weight change with random singular vectors and a power-law or low-rank-plus-noise spectrum.',
    role: 'construction',
    random: true,
    notes,
    cite: ['hu2022'],
  },
  lowRank.lowRankTarget,
)
fn(
  {
    key: 'lowRankLoss',
    name: 'Low-rank adapter loss and gradients',
    role: 'property',
    notes,
    cite: ['hu2022'],
  },
  lowRank.lowRankLoss,
)
fn(
  {
    key: 'lowRankFit',
    name: 'Low-rank adaptation fit (LoRA, rsLoRA, PiSSA)',
    tex: '\\min_{B, A} \\tfrac{1}{2}\\|s\\,BA - \\Delta W\\|_F^2',
    summary: 'Fit a weight change with s·BA from LoRA’s zero start or PiSSA’s principal start, by SGD or Adam.',
    role: 'fit',
    random: true,
    notes,
    cite: ['hu2022', 'kalajdzievski2023', 'meng2024pissa'],
  },
  lowRank.lowRankFit,
)

/** The functions of the module, keyed by name. */
export const adaptationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', lowRank) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
