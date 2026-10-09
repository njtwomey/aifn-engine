/**
 * The functions of `aifn-compute/dynamics/fields`, registered with the notes they serve: the calculus, flow, transport
 * and fixed-point functions, each with its role and the notes that use it.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as calculus from './calculus'
import * as fixed from './fixed'
import * as flow from './flow'

const fn = definer<FunctionInfo>('function', 'dynamics/fields')
const FLOWS = ['vector-fields-and-flows', 'ordinary-differential-equations']
const EQ = ['stability-and-equilibria', 'jacobian-linearisation']
const TRANSPORT = ['transport-and-continuity-equations', 'vector-fields-and-flows']

fn(
  {
    key: 'jacobianAt',
    name: 'Jacobian of a vector field',
    role: 'property',
    notes: ['jacobian', 'jacobian-linearisation'],
  },
  calculus.jacobianAt,
)
fn(
  { key: 'divergence', name: 'Divergence', tex: '\\nabla \\cdot f', role: 'property', notes: TRANSPORT },
  calculus.divergence,
)
fn(
  { key: 'curl', name: 'Curl', tex: '\\nabla \\times f', role: 'property', notes: ['vector-fields-and-flows'] },
  calculus.curl,
)
fn(
  { key: 'gradientAt', name: 'Gradient of a scalar field', role: 'property', notes: ['gradient'] },
  calculus.gradientAt,
)
fn(
  {
    key: 'gradientField',
    name: 'Gradient field',
    role: 'construction',
    notes: ['gradient', 'vector-fields-and-flows'],
  },
  calculus.gradientField,
)
fn(
  {
    key: 'hamiltonianField',
    name: 'Hamiltonian vector field',
    summary: 'The symplectic field (∂H/∂p, −∂H/∂q) of a Hamiltonian H(q, p).',
    role: 'construction',
    notes: FLOWS,
  },
  calculus.hamiltonianField,
)
fn({ key: 'autonomous', name: 'Autonomous right-hand side', role: 'construction', notes: FLOWS }, flow.autonomous)
fn({ key: 'flowMap', name: 'Flow map', tex: '\\varphi_t(x_0)', role: 'solver', notes: FLOWS }, flow.flowMap)
fn({ key: 'trajectory', name: 'Trajectory', role: 'solver', notes: FLOWS }, flow.trajectory)
fn({ key: 'streamline', name: 'Streamline', role: 'solver', notes: ['vector-fields-and-flows'] }, flow.streamline)
fn({ key: 'streamlines', name: 'Streamlines', role: 'solver', notes: ['vector-fields-and-flows'] }, flow.streamlines)
fn(
  {
    key: 'transportDensity',
    name: 'Transport a density',
    summary:
      'A density carried by a flow, by the continuity equation along characteristics (log-density and divergence).',
    role: 'solver',
    notes: TRANSPORT,
  },
  flow.transportDensity,
)
fn(
  { key: 'transportDensityFrames', name: 'Transported density frames', role: 'solver', notes: TRANSPORT },
  flow.transportDensityFrames,
)
fn(
  {
    key: 'pushForwardDensity',
    name: 'Push samples forward',
    role: 'simulation',
    notes: [...TRANSPORT, 'change-of-variables'],
  },
  flow.pushForwardDensity,
)
fn(
  { key: 'pushForwardDensityFrames', name: 'Pushed-forward sample frames', role: 'simulation', notes: TRANSPORT },
  flow.pushForwardDensityFrames,
)
fn(
  {
    key: 'classifyLinear',
    name: 'Classify a linear fixed point',
    summary: 'Node, saddle, focus, centre or degenerate, from the trace, determinant and eigenvalues of the Jacobian.',
    role: 'property',
    notes: [...EQ, 'linear-systems-and-the-matrix-exponential'],
    cite: ['strogatz2015'],
  },
  fixed.classifyLinear,
)
fn({ key: 'linearise', name: 'Linearise at a point', role: 'property', notes: EQ }, fixed.linearise)
fn({ key: 'fixedPoints', name: 'Fixed points in a box', role: 'solver', notes: EQ }, fixed.fixedPoints)
fn(
  {
    key: 'invariantManifolds',
    name: 'Stable and unstable manifolds',
    role: 'solver',
    notes: ['stability-and-equilibria'],
  },
  fixed.invariantManifolds,
)
fn(
  {
    key: 'lyapunovDerivative',
    name: 'Lyapunov derivative',
    tex: '\\dot V = \\nabla V \\cdot f',
    role: 'property',
    notes: ['lyapunov-stability', 'stability-and-equilibria'],
  },
  fixed.lyapunovDerivative,
)

/** The functions of the module, keyed by name. */
export const fieldsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', calculus, flow, fixed) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
