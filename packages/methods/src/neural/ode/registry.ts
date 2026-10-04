/**
 * The registry of `aifn-methods/neural/ode`: the neural ODE family, the continuous normalising flow, the latent ODE and
 * their streamed runs, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cnf from './cnf'
import * as latent from './latent'
import * as models from './models'
import * as run from './run'

const fn = definer<FunctionInfo>('function', 'neural/ode')
const NODE = 'neural-ordinary-differential-equations'

fn(
  {
    key: 'odeModel',
    name: 'Neural ODE family',
    summary:
      'A neural ODE, augmented NODE, second-order NODE or the ResNet it discretises, with a classifier or regressor readout.',
    role: 'construction',
    notes: [NODE, 'residual-network', 'residual-connections'],
    cite: ['chen2018', 'dupont2019', 'he2016'],
  },
  models.odeModel,
)
fn(
  {
    key: 'reflectionData',
    name: 'Reflection g(x) = −x',
    summary: 'Points of g(x) = −x on [−1, 1], which no 1-d neural ODE can fit because its trajectories cannot cross.',
    role: 'construction',
    notes: [NODE, 'vector-fields-and-flows'],
    cite: ['dupont2019'],
  },
  run.reflectionData,
)
fn(
  {
    key: 'discInRing',
    name: 'Disc inside a ring',
    summary:
      'Points uniform on a disc (class 0) and on a surrounding annulus (class 1): no flow of the plane separates them.',
    role: 'construction',
    random: true,
    notes: [NODE, 'vector-fields-and-flows'],
    cite: ['dupont2019'],
  },
  run.discInRing,
)
fn(
  {
    key: 'odeRun',
    name: 'Streamed neural ODE training run',
    summary:
      'Train a member of the neural ODE family and yield trajectories, fields, work per iteration and adjoint-vs-backprop gradients.',
    role: 'simulation',
    random: true,
    notes: [NODE, 'numerical-ode-solvers', 'backpropagation'],
    cite: ['chen2018', 'dupont2019', 'kidger2022'],
  },
  run.odeRun,
)
fn(
  {
    key: 'cnf',
    name: 'Continuous normalising flow',
    summary: 'A time-dependent neural ODE whose log density follows the instantaneous change of variables (FFJORD).',
    role: 'construction',
    notes: ['normalising-flow', NODE, 'probability-flow-ode'],
    cite: ['chen2018', 'grathwohl2019', 'hutchinson1989'],
  },
  cnf.cnf,
)
fn(
  {
    key: 'cnfRun',
    name: 'Streamed CNF training run',
    summary:
      'Train a CNF on 2-d points and yield samples moving through the flow, the density at each time and the NLL.',
    role: 'simulation',
    random: true,
    notes: ['normalising-flow', NODE],
    cite: ['grathwohl2019'],
  },
  cnf.cnfRun,
)
fn(
  {
    key: 'trajectories',
    name: 'Irregularly sampled trajectories',
    summary: 'Sines or 2-d spirals on a time grid, observed at random grid points in the first half.',
    role: 'construction',
    random: true,
    notes: [NODE],
    cite: ['chen2018'],
  },
  latent.trajectories,
)
fn(
  {
    key: 'latentOde',
    name: 'Latent ODE',
    summary: 'A GRU encoder to q(z₀), a neural ODE in the latent space and a linear decoder.',
    role: 'construction',
    notes: [NODE, 'variational-autoencoder'],
    cite: ['chen2018'],
  },
  latent.latentOde,
)
fn(
  {
    key: 'latentOdeRun',
    name: 'Streamed latent ODE training run',
    summary: 'Train a latent ODE on irregular trajectories and yield interpolation, extrapolation and latent paths.',
    role: 'simulation',
    random: true,
    notes: [NODE],
    cite: ['chen2018'],
  },
  latent.latentOdeRun,
)

/** The functions of the module, keyed by name. */
export const neuralOdeFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', models, run, cnf, latent) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
