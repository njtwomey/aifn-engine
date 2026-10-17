/** The functions of `aifn-methods/algorithms/program-synthesis`: string programs synthesised from examples. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as strings from './strings'

const fn = definer<FunctionInfo>('function', 'algorithms/program-synthesis')
const notes = ['program-synthesis-from-examples']
const cite = ['gulwani2011flashfill', 'lau2003vsa']

fn(
  {
    key: 'versionSpace',
    name: 'Version space of string programs',
    role: 'solver',
    summary:
      'Every concatenation of constants, tokens and initials that maps each input to its output, as a graph over tuples of output positions (FlashFill).',
    notes,
    cite,
  },
  strings.versionSpace,
)
fn(
  {
    key: 'exampleVersionSpace',
    name: 'Version space of one example',
    role: 'construction',
    summary: 'The graph over output positions whose edges carry every atom producing that piece of the output.',
    notes,
    cite,
  },
  strings.exampleVersionSpace,
)
fn(
  {
    key: 'intersectVersionSpace',
    name: 'Intersect a version space with an example',
    role: 'transform',
    summary: 'The product graph of a version space and one more example, pruned to the nodes on a complete path.',
    notes,
    cite,
  },
  strings.intersectVersionSpace,
)
fn(
  {
    key: 'countByLength',
    name: 'Count consistent programs',
    role: 'property',
    summary: 'The number of programs in a version space of each length, by dynamic programming over the graph.',
    notes,
  },
  strings.countByLength,
)
fn(
  {
    key: 'topPrograms',
    name: 'Cheapest consistent programs',
    role: 'solver',
    summary: 'The k cheapest programs of a version space under an additive cost per atom.',
    notes,
  },
  strings.topPrograms,
)
fn(
  {
    key: 'runStringProgram',
    name: 'Run a string program',
    role: 'transform',
    summary: "A program's output on an input, or null when one of its atoms is undefined there.",
    notes,
  },
  strings.runStringProgram,
)

/** The functions of the module, keyed by name. */
export const programSynthesisFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', strings) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
