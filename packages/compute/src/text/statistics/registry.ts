/** The functions of `aifn-compute/text/statistics`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as s from './statistics'

const fn = definer<FunctionInfo>('function', 'text/statistics')
const notes = ['tokenisation', 'byte-pair-encoding']

fn(
  {
    key: 'tokenisationStatistics',
    name: 'Tokenisation statistics',
    role: 'estimator',
    summary: 'Fertility, bytes and characters per token, unknown rate, word coverage and vocabulary usage.',
    notes,
  },
  s.tokenisationStatistics,
)
fn(
  {
    key: 'tokeniserStatistics',
    name: 'Tokeniser statistics on a corpus',
    role: 'estimator',
    summary: 'Encode each text and measure fertility, compression, unknowns, coverage and vocabulary usage.',
    notes,
  },
  s.tokeniserStatistics,
)

/** The functions of the module, keyed by name. */
export const statisticsFunctions = entries<FunctionInfo>('function', s) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
