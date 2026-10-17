/**
 * The registry of `aifn-methods/neural/pinn`: the problems, Burgers' exact solution and the streamed training runs, as
 * functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as problems from './problems'
import * as run from './run'
import * as variants from './variants'

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
    notes: [
      PINN,
      'partial-differential-equations',
      'hard-constraints-for-physics-informed-networks',
      'inverse-problems-with-physics-informed-networks',
    ],
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

fn(
  {
    key: 'poissonProblem',
    name: 'Poisson problem with an imbalanced residual',
    summary: '−u″ = f on [0, 1] with u = sin(kπx) + x: a residual of size (kπ)² against boundary values of size 1.',
    role: 'construction',
    notes: ['loss-balancing-for-physics-informed-networks', 'hard-constraints-for-physics-informed-networks'],
    cite: ['wang2023expert'],
  },
  problems.poissonProblem,
)
fn(
  {
    key: 'layerProblem',
    name: 'Interior layer problem',
    summary: '−u″ = f on [0, 1] with u = tanh((x − ½)/ε): a layer that uniform collocation points rarely hit.',
    role: 'construction',
    notes: ['loss-balancing-for-physics-informed-networks'],
    cite: ['wu2023rad'],
  },
  problems.layerProblem,
)
fn(
  {
    key: 'convectionProblem',
    name: 'Convection problem',
    summary: 'u_t + βu_x = 0 from sin x: a plain physics-informed network fails above β ≈ 10.',
    role: 'construction',
    notes: ['failure-modes-of-physics-informed-networks'],
    cite: ['krishnapriyan2021'],
  },
  problems.convectionProblem,
)
fn(
  {
    key: 'pinnVariantRun',
    name: 'Streamed PINN training with variants',
    summary:
      'PINN training with hard conditions, inverse coefficients, gradient-norm weights, resampling and causal weights.',
    role: 'simulation',
    notes: [
      'hard-constraints-for-physics-informed-networks',
      'loss-balancing-for-physics-informed-networks',
      'failure-modes-of-physics-informed-networks',
      'inverse-problems-with-physics-informed-networks',
    ],
    cite: ['raissi2019', 'lagaris1998', 'sukumar2022', 'wang2023expert', 'wu2023rad', 'krishnapriyan2021'],
  },
  variants.pinnVariantRun,
)

/** The functions of the module, keyed by name. */
export const pinnFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', problems, run, variants) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
