/**
 * The functions of `aifn-compute/numerics/implicit`: implicit differentiation of fixed points and roots.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as implicit from './implicit'

const fn = definer<FunctionInfo>('function', 'numerics/implicit')

fn(
  {
    key: 'implicitFixedPoint',
    name: 'Differentiable fixed point',
    summary: 'x* = T(x*, θ) with dx*/dθ = (I − ∂T/∂x)⁻¹ ∂T/∂θ by the implicit function theorem.',
    role: 'transform',
    notes: ['jacobian-vector-products-and-higher-order-derivatives'],
  },
  implicit.implicitFixedPoint,
)
fn(
  {
    key: 'implicitRoot',
    name: 'Differentiable root',
    summary: 'F(x*, θ) = 0 with dx*/dθ = −(∂F/∂x)⁻¹ ∂F/∂θ by the implicit function theorem.',
    role: 'transform',
    notes: ['root-finding'],
  },
  implicit.implicitRoot,
)

/** The functions of the module, keyed by name. */
export const implicitFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', implicit) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
