/**
 * The algorithms of `aifn-compute/dynamics/ode`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as adjoint from './adjoint'
import * as events from './events'
import * as linear from './linear'
import * as neural from './neural'
import * as rows from './rows'
import * as solve from './solve'
import * as stability from './stability'
import * as adaptive from './adaptive'
import * as explicit from './explicit'
import * as implicit from './implicit'
import * as symplectic from './symplectic'
import * as variableBdf from './variable-bdf'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/ode')

algorithm(
  {
    key: 'rungeKutta',
    name: 'Explicit Runge–Kutta',
    summary: 'A fixed-step explicit Runge–Kutta method from a Butcher tableau (Euler, Heun, midpoint, RK4).',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  explicit.rungeKutta,
)
algorithm(
  {
    key: 'dormandPrince',
    name: 'Dormand–Prince',
    summary: 'The adaptive 5(4) embedded Runge–Kutta pair with error-controlled step sizes.',
    problem: 'ode',
    state: { iterate: 'x', objective: 'error', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['dormand1980'],
  },
  adaptive.dormandPrince,
)
algorithm(
  {
    key: 'implicitEuler',
    name: 'Implicit Euler',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  implicit.implicitEuler,
)
algorithm(
  {
    key: 'implicitTrapezoid',
    name: 'Implicit trapezoid',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  implicit.implicitTrapezoid,
)
algorithm(
  {
    key: 'bdf',
    name: 'Backward differentiation formula',
    summary: 'The BDF methods of order 1 to 3 for stiff problems.',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  implicit.bdf,
)
algorithm(
  {
    key: 'adaptiveBdf',
    name: 'Adaptive BDF',
    summary:
      'Variable-order (1–5), variable-step backward differentiation formulae (NDF) for stiff problems, as scipy BDF.',
    problem: 'ode',
    state: { iterate: 'x', objective: 'error', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['shampine1997'],
  },
  variableBdf.adaptiveBdf,
)
algorithm(
  {
    key: 'symplectic',
    name: 'Symplectic integrator',
    summary: 'Symplectic Euler, leapfrog and higher-order splittings for separable Hamiltonians.',
    problem: 'ode',
    state: { iterate: 'x', objective: 'energy', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  symplectic.symplectic,
)

/** Every algorithm of the module, keyed by factory name. */
export const odeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', adaptive, explicit, implicit, symplectic, variableBdf) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'dynamics/ode')
const ODE = ['numerical-ode-solvers', 'ordinary-differential-equations']

fn(
  {
    key: 'solveIvp',
    name: 'Solve an initial value problem',
    summary: 'One entry point over the registered solvers, as scipy.integrate.solve_ivp.',
    role: 'solver',
    notes: ODE,
    cite: ['hairer1993'],
  },
  solve.solveIvp,
)
fn(
  {
    key: 'linearFlow',
    name: 'Flow of a linear system',
    tex: 'x(t) = e^{At} x_0',
    role: 'solver',
    notes: ['linear-systems-and-the-matrix-exponential', 'first-order-linear-odes'],
  },
  linear.linearFlow,
)
fn(
  {
    key: 'amplification',
    name: 'Amplification factor',
    tex: 'R(z)',
    role: 'property',
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1996'],
  },
  stability.amplification,
)
fn(
  {
    key: 'stabilityRegion',
    name: 'Absolute-stability region',
    role: 'property',
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1996'],
  },
  stability.stabilityRegion,
)
fn(
  { key: 'boundaryLocus', name: 'Boundary locus', role: 'property', notes: ['numerical-ode-solvers'] },
  stability.boundaryLocus,
)
fn(
  {
    key: 'hamiltonianSystem',
    name: 'Hamiltonian system',
    role: 'construction',
    notes: ['vector-fields-and-flows', 'hamiltonian-monte-carlo'],
  },
  symplectic.hamiltonianSystem,
)
fn({ key: 'withEvents', name: 'ODE solver with event detection', role: 'construction', notes: ODE }, events.withEvents)
fn(
  {
    key: 'odeAdjoint',
    name: 'Adjoint sensitivities of an ODE',
    summary: 'Gradients of a loss through an ODE solve by integrating the adjoint system backwards.',
    role: 'solver',
    notes: ['neural-ordinary-differential-equations', 'backpropagation'],
  },
  adjoint.odeAdjoint,
)
const NODE = ['neural-ordinary-differential-equations']
fn(
  {
    key: 'odeFlow',
    name: 'Differentiable ODE flow',
    summary:
      'States of an ODE at given times, differentiated by backpropagation through the solver or by the adjoint method.',
    role: 'solver',
    notes: [...NODE, 'numerical-ode-solvers', 'backpropagation'],
    cite: ['chen2018', 'kidger2022'],
  },
  neural.odeFlow,
)
fn(
  {
    key: 'jacobianTrace',
    name: 'Jacobian trace (exact or Hutchinson)',
    summary:
      'The trace of a row-wise map’s Jacobian per row, exactly by forward products or by Hutchinson’s estimator.',
    role: 'estimator',
    notes: ['normalising-flow', ...NODE],
    cite: ['hutchinson1989', 'grathwohl2019'],
  },
  neural.jacobianTrace,
)
fn(
  {
    key: 'traceProbe',
    name: 'Trace-estimator probe',
    summary: 'A Rademacher (±1) or Gaussian probe ε with E[εεᵀ] = I for Hutchinson’s trace estimator.',
    role: 'construction',
    random: true,
    notes: ['normalising-flow'],
    cite: ['hutchinson1989'],
  },
  neural.traceProbe,
)
fn(
  {
    key: 'augmentedDynamics',
    name: 'Augmented CNF dynamics',
    summary:
      'An ODE with the change in log density, the kinetic energy and the Jacobian Frobenius norm integrated alongside.',
    role: 'construction',
    notes: ['normalising-flow', ...NODE, 'vector-fields-and-flows'],
    cite: ['chen2018', 'grathwohl2019'],
  },
  neural.augmentedDynamics,
)
fn(
  {
    key: 'dormandPrinceRows',
    name: 'Dormand–Prince on many independent problems',
    summary:
      'Solve a batch of independent IVPs together, each row with its own adaptive steps: per-instance work at batched cost.',
    role: 'solver',
    notes: ['numerical-ode-solvers', ...NODE],
    cite: ['dormand1980'],
  },
  rows.dormandPrinceRows,
)

/** The functions of the module, keyed by name. */
export const odeFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', solve, linear, stability, symplectic, events, adjoint, neural, rows) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
