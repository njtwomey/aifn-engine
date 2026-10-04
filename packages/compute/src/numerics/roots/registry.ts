/**
 * The algorithms of `aifn-compute/numerics/roots`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as convenience from './convenience'
import * as minimize from './minimize'
import * as scalar from './scalar'
import * as systems from './systems'

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/roots')

algorithm(
  {
    key: 'bisection',
    stability: 'stable',
    name: 'Bisection',
    problem: 'root',
    state: { iterate: 'x', objective: 'fx', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding'],
  },
  scalar.bisection,
)
algorithm(
  {
    key: 'regulaFalsi',
    stability: 'stable',
    name: 'Regula falsi',
    problem: 'root',
    state: { iterate: 'x', objective: 'fx', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding'],
  },
  scalar.regulaFalsi,
)
algorithm(
  {
    key: 'brent',
    stability: 'stable',
    name: 'Brent',
    problem: 'root',
    state: { iterate: 'x', objective: 'fx', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding'],
  },
  scalar.brent,
)
algorithm(
  {
    key: 'secant',
    stability: 'stable',
    name: 'Secant',
    problem: 'root',
    state: { iterate: 'x', objective: 'fx', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding'],
  },
  scalar.secant,
)
algorithm(
  {
    key: 'newtonRoot',
    stability: 'stable',
    name: 'Newton (root)',
    problem: 'root',
    state: { iterate: 'x', objective: 'fx', grad: 'derivative', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding', 'newtons-method'],
  },
  scalar.newtonRoot,
)
algorithm(
  {
    key: 'newtonSystem',
    stability: 'stable',
    name: 'Newton (systems)',
    problem: 'system',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding', 'newtons-method'],
  },
  systems.newtonSystem,
)
algorithm(
  {
    key: 'broyden',
    stability: 'stable',
    name: 'Broyden',
    problem: 'system',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding', 'quasi-newton-methods'],
  },
  systems.broyden,
)
algorithm(
  {
    key: 'fixedPoint',
    stability: 'stable',
    name: 'Fixed-point iteration',
    problem: 'system',
    state: { iterate: 'x', objective: 'residualNorm', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding'],
  },
  systems.fixedPoint,
)
algorithm(
  {
    key: 'continuation',
    stability: 'stable',
    name: 'Continuation',
    summary: 'Natural-parameter continuation from an easy problem to the target, with a Newton corrector.',
    problem: 'system',
    state: { iterate: 'x', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['root-finding'],
  },
  systems.continuation,
)

/** Every algorithm of the module, keyed by factory name. */
export const rootsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', scalar, systems) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'numerics/roots')

fn(
  {
    key: 'findRoot',
    name: 'Find a root',
    summary: 'A bracketed root by Brent, or an unbracketed one by Newton or the secant method.',
    role: 'solver',
    notes: ['root-finding'],
  },
  convenience.findRoot,
)
fn(
  { key: 'solveSystem', name: 'Solve a nonlinear system', role: 'solver', notes: ['root-finding', 'newtons-method'] },
  convenience.solveSystem,
)
fn(
  {
    key: 'minimizeScalar',
    name: 'Minimise a function of one variable',
    summary: "Brent's method: golden-section search with parabolic steps.",
    role: 'solver',
    notes: ['line-search'],
  },
  minimize.minimizeScalar,
)

/** The functions of the module, keyed by name. */
export const rootsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', convenience, minimize) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
