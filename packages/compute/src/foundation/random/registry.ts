/**
 * The draws of `aifn-compute/foundation/random`, registered as functions with the notes they serve. (Stream plumbing
 * and the block helpers are infrastructure and are not registered.)
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as samplers from './samplers'

const fn = definer<FunctionInfo>('function', 'foundation/random')

fn(
  { key: 'uniform', name: 'Uniform draws', role: 'simulation', random: true, notes: ['inverse-transform-sampling'] },
  samplers.uniform,
)
fn(
  {
    key: 'normal',
    name: 'Normal draws (Box–Muller)',
    role: 'simulation',
    random: true,
    notes: ['box-muller-transform', 'gaussian-distribution'],
    cite: ['box1958'],
  },
  samplers.normal,
)
fn(
  {
    key: 'normals',
    name: 'Standard normal draws',
    role: 'simulation',
    random: true,
    notes: ['box-muller-transform'],
    cite: ['box1958'],
  },
  samplers.normals,
)
fn(
  {
    key: 'exponential',
    name: 'Exponential draws (inverse transform)',
    role: 'simulation',
    random: true,
    notes: ['inverse-transform-sampling', 'exponential-distribution'],
  },
  samplers.exponential,
)
fn(
  { key: 'bernoulli', name: 'Bernoulli draws', role: 'simulation', random: true, notes: ['bernoulli-distribution'] },
  samplers.bernoulli,
)
fn({ key: 'integers', name: 'Uniform integers', role: 'simulation', random: true }, samplers.integers)
fn(
  {
    key: 'categorical',
    name: 'Categorical draws',
    role: 'simulation',
    random: true,
    notes: ['categorical-distribution', 'inverse-transform-sampling'],
  },
  samplers.categorical,
)
fn(
  {
    key: 'aliasTable',
    name: 'Alias table',
    summary: "Walker's alias table for O(1) categorical draws.",
    role: 'construction',
    notes: ['categorical-distribution'],
  },
  samplers.aliasTable,
)
fn(
  {
    key: 'aliasSample',
    name: 'Alias-method draws',
    role: 'simulation',
    random: true,
    notes: ['categorical-distribution'],
  },
  samplers.aliasSample,
)
fn({ key: 'shuffle', name: 'Fisher–Yates shuffle', role: 'simulation', random: true }, samplers.shuffle)
fn({ key: 'permutation', name: 'Random permutation', role: 'simulation', random: true }, samplers.permutation)
fn({ key: 'choice', name: 'Random choice', role: 'simulation', random: true }, samplers.choice)

/** The draws of the module, keyed by name. */
export const randomFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', samplers) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
