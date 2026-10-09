/**
 * The functions and algorithms of `aifn-compute/dynamics/control`, registered with the notes they serve: the LQR,
 * pole-placement, MPC and LQG solvers as functions, and the receding-horizon and LQG loops as step algorithms.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as ackermann from './ackermann'
import * as lqg from './lqg'
import * as lqr from './lqr'
import * as mpc from './mpc'

const fn = definer<FunctionInfo>('function', 'dynamics/control')
const LQR = ['linear-quadratic-regulator', 'linear-quadratic-gaussian-control']

fn(
  {
    key: 'lqr',
    name: 'Continuous-time LQR',
    summary: 'The optimal state-feedback gain from the continuous algebraic Riccati equation.',
    role: 'solver',
    notes: LQR,
    cite: ['kalman1960'],
  },
  lqr.lqr,
)
fn(
  {
    key: 'dlqr',
    name: 'Discrete-time LQR',
    summary: 'The optimal state-feedback gain from the discrete algebraic Riccati equation.',
    role: 'solver',
    notes: LQR,
  },
  lqr.dlqr,
)
fn(
  {
    key: 'closedLoopPoles',
    name: 'Closed-loop poles',
    tex: '\\operatorname{eig}(A - BK)',
    role: 'property',
    notes: ['pole-placement', ...LQR],
  },
  lqr.closedLoopPoles,
)
fn(
  {
    key: 'ackermann',
    name: "Ackermann's formula",
    summary: 'The unique single-input gain placing the poles at chosen locations.',
    role: 'solver',
    notes: ['pole-placement'],
    cite: ['ackermann1972'],
  },
  ackermann.ackermann,
)

fn(
  {
    key: 'mpcController',
    name: 'Linear MPC controller',
    summary: 'The condensed finite-horizon QP of linear MPC with input and state bounds; plan(x₀) solves it.',
    role: 'construction',
    notes: ['model-predictive-control', 'model-predictive-control-stability-and-feasibility'],
    cite: ['rawlings2017', 'mayne2000'],
  },
  mpc.mpcController,
)
fn(
  {
    key: 'lqg',
    name: 'LQG design',
    summary: 'The LQR gain and the steady-state Kalman gain (the dual LQR), with the output-feedback compensator.',
    role: 'solver',
    notes: ['linear-quadratic-gaussian-control', 'kalman-filter', 'state-observer'],
    cite: ['kalman1960', 'anderson2007'],
  },
  lqg.lqg,
)

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/control')
algorithm(
  {
    key: 'recedingHorizon',
    name: 'Receding-horizon control',
    summary: 'Plan N steps by a QP from the measured state, apply the first input, repeat.',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'cost', flags: ['terminated', 'diverged'] },
    notes: ['model-predictive-control', 'model-predictive-control-stability-and-feasibility'],
    cite: ['rawlings2017'],
  },
  mpc.recedingHorizon,
)
algorithm(
  {
    key: 'lqgSimulation',
    name: 'LQG closed loop',
    summary: 'A noisy plant under u = −Kx̂ with the Kalman predictor x̂; noise drawn from the step stream.',
    problem: 'lti-system',
    state: { iterate: 'x', objective: 'cost', flags: ['diverged'] },
    random: true,
    notes: ['linear-quadratic-gaussian-control', 'kalman-filter'],
  },
  lqg.lqgSimulation,
)

/** Every algorithm of the module, keyed by factory name. */
export const controlAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', mpc, lqg) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const controlFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', lqr, ackermann, mpc, lqg) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
