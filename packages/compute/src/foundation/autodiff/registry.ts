/**
 * The transforms of `aifn-compute/foundation/autodiff`, registered as functions with the notes they serve. (`stopGradient` is
 * a primitive and is listed in the primitive table.)
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as check from './check'
import * as custom from './custom'
import * as graph from './graph'
import * as transforms from './transforms'

const fn = definer<FunctionInfo>('function', 'foundation/autodiff')
const REVERSE = ['backpropagation', 'operator-overloading-and-tapes']
const FORWARD = ['forward-mode-autodiff', 'dual-numbers', 'jacobian-vector-products-and-higher-order-derivatives']

fn(
  {
    key: 'grad',
    name: 'Gradient (reverse mode)',
    tex: '\\nabla f',
    role: 'transform',
    notes: ['gradient', ...REVERSE],
    cite: ['baydin2018', 'griewank2008'],
  },
  transforms.grad,
)
fn(
  {
    key: 'valueAndGrad',
    name: 'Value and gradient',
    role: 'transform',
    notes: ['gradient', ...REVERSE],
    cite: ['baydin2018'],
  },
  transforms.valueAndGrad,
)
fn(
  {
    key: 'vjp',
    name: 'Vector–Jacobian product',
    tex: 'v^\\top J',
    role: 'transform',
    notes: ['jacobian-vector-products-and-higher-order-derivatives', ...REVERSE],
    cite: ['griewank2008'],
  },
  transforms.vjp,
)
fn(
  {
    key: 'jvp',
    name: 'Jacobian–vector product',
    tex: 'J v',
    role: 'transform',
    notes: FORWARD,
    cite: ['griewank2008'],
  },
  transforms.jvp,
)
fn(
  {
    key: 'linearize',
    name: 'Linearise',
    summary: 'f(x) and the linear map v ↦ J(x)v.',
    role: 'transform',
    notes: FORWARD,
  },
  transforms.linearize,
)
fn(
  {
    key: 'hvp',
    name: 'Hessian–vector product',
    tex: 'H v',
    role: 'transform',
    notes: ['hessian', 'jacobian-vector-products-and-higher-order-derivatives'],
    cite: ['pearlmutter1994'],
  },
  transforms.hvp,
)
fn(
  {
    key: 'jacobian',
    name: 'Jacobian',
    tex: 'J_f',
    role: 'transform',
    notes: ['jacobian', 'jacobian-vector-products-and-higher-order-derivatives'],
  },
  transforms.jacobian,
)
fn({ key: 'hessian', name: 'Hessian', tex: '\\nabla^2 f', role: 'transform', notes: ['hessian'] }, transforms.hessian)
fn(
  {
    key: 'vmap',
    name: 'Vectorising map',
    summary: 'Lift a function of one example to a batch, by batching rules.',
    role: 'transform',
    cite: ['bradbury2018'],
  },
  transforms.vmap,
)
fn(
  {
    key: 'checkpoint',
    name: 'Gradient checkpointing',
    summary: 'Recompute the forward pass of f in the backward pass instead of storing its intermediates.',
    role: 'transform',
    notes: ['gradient-checkpointing'],
    cite: ['chen2016', 'griewank2000'],
  },
  custom.checkpoint,
)
fn(
  { key: 'customVjp', name: 'Custom vector–Jacobian product', role: 'transform', notes: ['backpropagation'] },
  custom.customVjp,
)
fn(
  { key: 'customJvp', name: 'Custom Jacobian–vector product', role: 'transform', notes: ['forward-mode-autodiff'] },
  custom.customJvp,
)
fn(
  {
    key: 'defineCustomVjp',
    name: 'Define a function with its own VJP',
    role: 'construction',
    notes: ['backpropagation'],
  },
  custom.defineCustomVjp,
)
fn(
  {
    key: 'gradCheck',
    name: 'Gradient check',
    summary: 'Reverse- and forward-mode derivatives against central differences.',
    role: 'test',
    notes: ['gradient', 'floating-point-arithmetic'],
  },
  check.gradCheck,
)
fn(
  {
    key: 'traceGraph',
    name: 'Computation graph',
    summary: 'The graph of primitive operations a function applies to its input.',
    role: 'transform',
    returns: 'graph',
    notes: ['static-and-dynamic-computation-graphs', 'backpropagation'],
  },
  graph.traceGraph,
)

/** The functions of the module, keyed by name. */
export const autodiffFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', transforms, custom, check, graph) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
