/** The functions of `aifn-compute/text/normalise`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as n from './normalise'

const fn = definer<FunctionInfo>('function', 'text/normalise')
const notes = ['text-normalisation']

fn(
  {
    key: 'normalise',
    name: 'Text normalisation',
    role: 'transform',
    summary: 'Unicode normal form, case folding, accent stripping and white space, in that order.',
    notes,
    cite: ['whistler2026uax15', 'unicode2026casefolding'],
  },
  n.normalise,
)
fn(
  {
    key: 'caseFold',
    name: 'Unicode case folding',
    role: 'transform',
    summary: 'Full Unicode case folding for caseless matching (ß → ss, ς → σ).',
    notes,
    cite: ['unicode2026casefolding'],
  },
  n.caseFold,
)
fn(
  {
    key: 'stripAccents',
    name: 'Accent stripping',
    role: 'transform',
    summary: 'NFKD, then drop combining marks.',
    notes,
  },
  n.stripAccents,
)
fn({ key: 'collapseWhitespace', name: 'White space collapsing', role: 'transform', notes }, n.collapseWhitespace)

/** The functions of the module, keyed by name. */
export const normaliseFunctions = entries<FunctionInfo>('function', n) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
