/**
 * The registry of `aifn-methods/information/channels`: the capacity and rate–distortion Blahut–Arimoto iterations as
 * traceable algorithms, and the solvers that run them to convergence as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as channel from './channel'

/** A table of the module's registry entries, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'information/channels')
const fn = definer<FunctionInfo>('function', 'information/channels')
const MI = ['mutual-information']

algorithm(
  {
    key: 'blahutArimotoCapacity',
    name: 'Blahut–Arimoto (capacity)',
    summary: 'Alternating maximisation of I(X; Y) over the input distribution, with upper and lower bounds.',
    problem: 'objective',
    state: { iterate: 'input', objective: 'information', flags: ['converged'] },
    notes: MI,
    cite: ['cover2006'],
  },
  channel.blahutArimotoCapacity,
)
algorithm(
  {
    key: 'blahutArimotoRateDistortion',
    name: 'Blahut–Arimoto (rate–distortion)',
    problem: 'objective',
    state: { iterate: 'conditional', objective: 'rate', flags: ['converged'] },
    notes: MI,
    cite: ['cover2006'],
  },
  channel.blahutArimotoRateDistortion,
)
fn(
  { key: 'channelCapacity', name: 'Channel capacity', role: 'solver', notes: MI, cite: ['shannon1948'] },
  channel.channelCapacity,
)
fn({ key: 'rateDistortion', name: 'Rate–distortion at a slope', role: 'solver', notes: MI }, channel.rateDistortion)
fn(
  { key: 'rateDistortionCurve', name: 'Rate–distortion curve', role: 'solver', notes: MI },
  channel.rateDistortionCurve,
)

/** The algorithms of the module, keyed by factory name. */
export const channelsAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  channel,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const channelsFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', channel) as Table<FunctionInfo>
