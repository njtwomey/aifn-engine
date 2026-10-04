/**
 * The functions of `aifn-compute/numerics/polynomial`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as polynomial from './polynomial'

const fn = definer<FunctionInfo>('function', 'numerics/polynomial')

fn(
  { key: 'polyval', name: 'Evaluate a polynomial (Horner)', role: 'transform', notes: ['taylor-series'] },
  polynomial.polyval,
)
fn(
  { key: 'polyDerivative', name: 'Polynomial derivative', role: 'transform', notes: ['derivative'] },
  polynomial.polyDerivative,
)
fn({ key: 'polyMul', name: 'Polynomial product', role: 'transform', notes: ['convolution'] }, polynomial.polyMul)
fn({ key: 'polyDivide', name: 'Polynomial division', role: 'transform' }, polynomial.polyDivide)
fn(
  { key: 'polyFromRoots', name: 'Polynomial from its roots', role: 'construction', notes: ['poles-and-zeros'] },
  polynomial.polyFromRoots,
)
fn(
  { key: 'companionMatrix', name: 'Companion matrix', role: 'construction', notes: ['eigendecomposition'] },
  polynomial.companionMatrix,
)
fn(
  {
    key: 'polynomialRoots',
    name: 'Polynomial roots',
    summary: 'The eigenvalues of the companion matrix by Francis QR.',
    role: 'solver',
    notes: ['root-finding', 'poles-and-zeros'],
  },
  polynomial.polynomialRoots,
)
fn(
  { key: 'roots', name: 'Roots (NumPy order)', role: 'solver', notes: ['root-finding', 'poles-and-zeros'] },
  polynomial.roots,
)
fn(
  {
    key: 'residue',
    name: 'Partial fractions (s-domain)',
    role: 'transform',
    notes: ['transfer-function-and-block-diagrams', 'poles-and-zeros'],
  },
  polynomial.residue,
)
fn(
  {
    key: 'residuez',
    name: 'Partial fractions (z-domain)',
    role: 'transform',
    notes: ['z-transform', 'poles-and-zeros'],
  },
  polynomial.residuez,
)

/** The functions of the module, keyed by name. */
export const polynomialFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', polynomial) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
