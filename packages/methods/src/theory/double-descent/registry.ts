/** The registry of `aifn-methods/theory/double-descent`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as dd from './double-descent'

const fn = definer<FunctionInfo>('function', 'theory/double-descent')
const notes = ['double-descent']

fn(
  {
    key: 'doubleDescent',
    name: 'Double descent with random features',
    summary:
      'Test error, training error and weight norm of min-norm random-features regression against the feature count.',
    role: 'simulation',
    random: true,
    notes: [...notes, 'bias-variance-decomposition'],
    cite: ['belkin2019', 'nakkiran2020'],
  },
  dd.doubleDescent,
)
fn(
  {
    key: 'randomFeatures',
    name: 'Random features',
    summary: 'Random ReLU units max(0, a·x/√d) or random Fourier features cos(a·x/√d + b), scaled by 1/√p.',
    role: 'transform',
    notes,
  },
  dd.randomFeatures,
)
fn(
  {
    key: 'randomFeatureMap',
    name: 'Random feature map',
    summary: 'Random directions aⱼ ~ N(0, I) and phases for ReLU or Fourier features.',
    role: 'construction',
    random: true,
    notes,
  },
  dd.randomFeatureMap,
)
fn({ key: 'featureCounts', name: 'Feature counts', role: 'construction', notes }, dd.featureCounts)

/** The functions of the module, keyed by name. */
export const doubleDescentFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', dd) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
