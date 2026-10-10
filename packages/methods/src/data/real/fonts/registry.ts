/**
 * The registry entries of the functions of `aifn-methods/data/real/fonts` besides its dataset: `fontVectors`,
 * `glyphContours` and `fontTable`, each linked to the manifold-of-fonts and GPLVM notes.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as fonts from './fonts'

/** Registers a function of the module under `data/real/fonts`. */
const fn = definer<FunctionInfo>('function', 'data/real/fonts')
/** The notes every entry links to. */
const NOTES = ['manifold-of-fonts', 'gaussian-process-latent-variable-model']

fn(
  {
    key: 'fontVectors',
    name: 'Font vectors',
    summary: 'Each font as one vector of glyph outline coordinates.',
    role: 'construction',
    notes: NOTES,
  },
  fonts.fontVectors,
)
fn(
  { key: 'glyphContours', name: 'Glyph contours of a font vector', role: 'transform', notes: NOTES },
  fonts.glyphContours,
)
fn({ key: 'fontTable', name: 'Font table', role: 'construction', notes: NOTES }, fonts.fontTable)

/** The functions of the module, keyed by name. */
export const fontFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', fonts) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
