/** The registry of `aifn-methods/theory/concentration`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as c from './concentration'

const fn = definer<FunctionInfo>('function', 'theory/concentration')
const BOUNDS = ['hoeffdings-inequality', 'chernoff-bounds']

fn(
  {
    key: 'concentrationStudy',
    name: 'Tail bounds against simulation',
    summary: 'Monte Carlo P(|X̄ − μ| ≥ t) for a bounded law against Chebyshev, Hoeffding, Bernstein and Chernoff.',
    role: 'simulation',
    random: true,
    notes: BOUNDS,
    cite: ['hoeffding1963', 'chernoff1952', 'boucheron2013'],
  },
  c.concentrationStudy,
)
fn(
  {
    key: 'tailBounds',
    name: 'Concentration bounds',
    summary: 'Chebyshev, Hoeffding, Bernstein and relative-entropy Chernoff bounds for a mean of [0, 1] variables.',
    role: 'property',
    notes: BOUNDS,
    cite: ['hoeffding1963', 'boucheron2013'],
  },
  c.tailBounds,
)
fn(
  {
    key: 'mcdiarmidStudy',
    name: 'McDiarmid on empty bins',
    summary: 'The fraction of empty bins after n throws: its simulated tail against the bounded-differences bound.',
    role: 'simulation',
    random: true,
    notes: ['mcdiarmids-inequality'],
    cite: ['mcdiarmid1989'],
  },
  c.mcdiarmidStudy,
)
fn(
  {
    key: 'runningMeans',
    name: 'Running means',
    summary: 'Sample means along independent sequences, converging to μ.',
    role: 'simulation',
    random: true,
    notes: ['law-of-large-numbers'],
  },
  c.runningMeans,
)
fn(
  {
    key: 'standardisedSums',
    name: 'Standardised sums',
    summary: '√n (X̄ − μ)/σ over many samples, approaching N(0, 1) when the variance is finite.',
    role: 'simulation',
    random: true,
    notes: ['central-limit-theorem'],
  },
  c.standardisedSums,
)
fn(
  { key: 'sampleMeans', name: 'Sample means', role: 'simulation', random: true, notes: ['law-of-large-numbers'] },
  c.sampleMeans,
)
fn({ key: 'lawMoments', name: 'Summand moments', role: 'property', notes: ['central-limit-theorem'] }, c.lawMoments)
fn({ key: 'bernoulliKl', name: 'Bernoulli KL', role: 'property', notes: ['chernoff-bounds'] }, c.bernoulliKl)

/** The functions of the module, keyed by name. */
export const concentrationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', c) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
