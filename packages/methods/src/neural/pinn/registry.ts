/**
 * The registry of `aifn-methods/neural/pinn`: the problems, Burgers' exact solution and the streamed training run, as
 * functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as problems from './problems'
import * as run from './run'

const fn = definer<FunctionInfo>('function', 'neural/pinn')
const PINN = 'physics-informed-neural-networks'

fn(
  {
    key: 'burgersSolution',
    name: "Burgers' equation by Cole–Hopf",
    summary: 'The exact solution of viscous Burgers from −sin(πx), as a ratio of Gaussian integrals.',
    tex: 'u = -2\\nu \\, \\phi_x / \\phi',
    role: 'property',
    notes: [PINN, 'partial-differential-equations'],
    cite: ['hopf1950', 'cole1951'],
  },
  problems.burgersSolution,
)
fn(
  {
    key: 'burgers',
    name: "Burgers' equation problem",
    summary: 'u_t + u u_x = ν u_xx on [−1, 1] × [0, 1] from −sin(πx) with zero ends, the PINN benchmark.',
    role: 'construction',
    notes: [PINN, 'partial-differential-equations'],
    cite: ['raissi2019'],
  },
  problems.burgers,
)
fn(
  {
    key: 'heat',
    name: 'Heat equation problem',
    summary: 'u_t = κ u_xx on [−1, 1] × [0, 1] from sin(πx) with zero ends, solved by a decaying sine.',
    role: 'construction',
    notes: [PINN, 'partial-differential-equations'],
  },
  problems.heat,
)
fn(
  {
    key: 'pinnRun',
    name: 'Streamed PINN training run',
    summary:
      'Fit a network to a PDE by Adam then L-BFGS on its residual, initial and boundary losses, with the error on a grid.',
    role: 'simulation',
    notes: [PINN, 'automatic-differentiation'],
    cite: ['raissi2019', 'liu1989', 'kingma2015'],
  },
  run.pinnRun,
)

/** The functions of the module, keyed by name. */
export const pinnFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', problems, run) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
