/** The registry of `aifn-methods/generative/flows`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as realnvp from './realnvp'
import * as run from './run'

const fn = definer<FunctionInfo>('function', 'generative/flows')
const notes = ['normalising-flow', 'change-of-variables']

fn(
  {
    key: 'realNvp',
    name: 'RealNVP flow',
    summary: 'Alternating affine coupling layers with MLP conditioners and bounded log-scales.',
    role: 'construction',
    notes,
    cite: ['dinh2017'],
  },
  realnvp.realNvp,
)
fn(
  {
    key: 'flowLogDensity',
    name: 'Flow log-density',
    summary: 'log N(f(x); 0, I) plus the coupling layers’ log-determinants.',
    role: 'property',
    notes,
    cite: ['dinh2017'],
  },
  realnvp.flowLogDensity,
)
fn({ key: 'flowSample', name: 'Flow samples', role: 'simulation', notes }, realnvp.flowSample)
fn({ key: 'flowForward', name: 'Flow layers', role: 'transform', notes }, realnvp.flowForward)
fn({ key: 'couplingLayer', name: 'RealNVP coupling layer', role: 'construction', notes }, realnvp.couplingLayer)
fn(
  { key: 'initRealNvp', name: 'Initial RealNVP parameters', role: 'construction', random: true, notes },
  realnvp.initRealNvp,
)
fn(
  { key: 'flowLogDensityValues', name: 'Flow log-density values', role: 'property', notes },
  realnvp.flowLogDensityValues,
)
fn(
  {
    key: 'realNvpRun',
    name: 'Streamed RealNVP training',
    summary: 'Maximum likelihood by Adam; density on a grid, samples and the data through each layer over training.',
    role: 'simulation',
    random: true,
    notes,
    cite: ['dinh2017'],
  },
  run.realNvpRun,
)

/** The functions of the module, keyed by name. */
export const flowFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', realnvp, run) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
