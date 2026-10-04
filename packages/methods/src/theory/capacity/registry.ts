/** The registry of `aifn-methods/theory/capacity`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as c from './capacity'

const fn = definer<FunctionInfo>('function', 'theory/capacity')
const VC = ['vapnik-chervonenkis-dimension']

fn(
  {
    key: 'realisable',
    name: 'Realisable labelling',
    summary: 'Whether half-planes (by a feasibility LP), rectangles or intervals realise a ±1 labelling.',
    role: 'property',
    notes: VC,
    cite: ['vapnik1971'],
  },
  c.realisable,
)
fn(
  {
    key: 'shatteringTable',
    name: 'Shattering table',
    summary: 'All 2ⁿ labellings of a point set and which ones a class realises.',
    role: 'property',
    notes: VC,
    cite: ['vapnik1971', 'shalevshwartz2014'],
  },
  c.shatteringTable,
)
fn(
  {
    key: 'empiricalRademacher',
    name: 'Empirical Rademacher complexity',
    summary: 'E_σ max_h (1/n) Σ σᵢ h(xᵢ) over random signs, with Massart’s finite-class bound.',
    role: 'estimator',
    random: true,
    notes: ['rademacher-complexity'],
    cite: ['bartlett2002', 'mohri2018'],
  },
  c.empiricalRademacher,
)

/** The functions of the module, keyed by name. */
export const capacityFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', c) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
