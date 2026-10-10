/**
 * The registry of covariance kernels (design S §2.10): each factory with its hyperparameters as a `Space` (the fields
 * of its options object; `matern` and `polynomial` also take $\nu$ or the degree first) and whether it is stationary.
 * The kernel view (profile, Gram heatmap, prior draws) and the lab's kernel picker enumerate this table. The module's
 * other functions (`gram`, `kernelDiagonal`, `kernelProfile`) are registered as functions, in `kernelsFunctions`.
 */

import { definer, entries, type Entry, type FunctionInfo, type KernelInfo } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import * as kernels from './kernels'

const kernel = definer<KernelInfo>('kernel', 'learning/kernels')
const cite = ['rasmussen2006']
const lengthscale = real(0.05, 10, { default: 1, scale: 'log', label: '\\ell', doc: 'lengthscale' })
const variance = real(0.01, 10, { default: 1, scale: 'log', label: '\\sigma^2', doc: 'variance k(x, x)' })
const stationary = space({ lengthscale, variance })
const notes = ['covariance-functions', 'gaussian-process']

kernel(
  {
    key: 'rbf',
    name: 'Squared exponential (RBF)',
    hyper: stationary,
    stationary: true,
    cite,
    notes,
    glossary: 'rbf-kernel',
  },
  kernels.rbf,
)
kernel(
  {
    key: 'matern',
    name: 'Matérn',
    summary: 'The Matérn kernel with smoothness ν ∈ {1/2, 3/2, 5/2}, given as the first argument.',
    hyper: space({
      nu: oneOf([0.5, 1.5, 2.5], { default: 1.5, label: '\\nu', doc: 'smoothness' }),
      lengthscale,
      variance,
    }),
    stationary: true,
    cite,
    notes,
  },
  kernels.matern,
)
kernel({ key: 'matern12', name: 'Matérn 1/2', hyper: stationary, stationary: true, cite, notes }, kernels.matern12)
kernel({ key: 'matern32', name: 'Matérn 3/2', hyper: stationary, stationary: true, cite, notes }, kernels.matern32)
kernel({ key: 'matern52', name: 'Matérn 5/2', hyper: stationary, stationary: true, cite, notes }, kernels.matern52)
kernel(
  {
    key: 'rationalQuadratic',
    name: 'Rational quadratic',
    hyper: space({
      lengthscale,
      variance,
      alpha: real(0.05, 20, { default: 1, scale: 'log', label: '\\alpha', doc: 'scale mixture' }),
    }),
    stationary: true,
    cite,
    notes,
  },
  kernels.rationalQuadratic,
)
kernel(
  {
    key: 'periodic',
    name: 'Periodic',
    hyper: space({
      lengthscale,
      variance,
      period: real(0.1, 10, { default: 1, scale: 'log', label: 'p', doc: 'period' }),
    }),
    stationary: true,
    cite,
    notes,
  },
  kernels.periodic,
)
kernel(
  {
    key: 'linearKernel',
    name: 'Linear',
    hyper: space({ variance, bias: real(0, 10, { default: 0, label: '\\sigma_b^2', doc: 'bias variance' }) }),
    stationary: false,
    cite,
    notes: ['covariance-functions', 'gaussian-processes-and-bayesian-linear-regression'],
  },
  kernels.linearKernel,
)
kernel(
  {
    key: 'polynomial',
    name: 'Polynomial',
    summary: 'The polynomial kernel (σ_b² + σ² xᵀx′)^d, with the degree d given as the first argument.',
    hyper: space({
      degree: int(1, 6, { default: 2, label: 'd', doc: 'degree' }),
      variance,
      bias: real(0, 10, { default: 1, label: '\\sigma_b^2', doc: 'bias variance' }),
    }),
    stationary: false,
    cite,
    notes: ['covariance-functions', 'kernel-trick'],
  },
  kernels.polynomial,
)
kernel({ key: 'white', name: 'White noise', hyper: space({ variance }), stationary: true, cite, notes }, kernels.white)
kernel(
  { key: 'constant', name: 'Constant', hyper: space({ variance }), stationary: true, cite, notes },
  kernels.constant,
)
kernel(
  { key: 'sumKernel', name: 'Sum', hyper: space({}), stationary: false, composite: true, cite, notes },
  kernels.sumKernel,
)
kernel(
  { key: 'productKernel', name: 'Product', hyper: space({}), stationary: false, composite: true, cite, notes },
  kernels.productKernel,
)

/** Every kernel factory, keyed by export name. */
export const kernelRegistry: Readonly<Record<string, Entry<(...args: never[]) => kernels.Kernel, KernelInfo>>> =
  entries<KernelInfo>('kernel', kernels) as Readonly<
    Record<string, Entry<(...args: never[]) => kernels.Kernel, KernelInfo>>
  >

const fn = definer<FunctionInfo>('function', 'learning/kernels')

fn(
  {
    key: 'gram',
    name: 'Gram matrix',
    tex: 'K_{ij} = k(x_i, x_j)',
    role: 'construction',
    notes: ['kernel-trick', 'covariance-functions', 'positive-definite-matrices'],
  },
  kernels.gram,
)
fn(
  { key: 'kernelDiagonal', name: 'Kernel diagonal', role: 'construction', notes: ['covariance-functions'] },
  kernels.kernelDiagonal,
)
fn(
  {
    key: 'kernelProfile',
    name: 'Kernel profile',
    summary: 'k(x, x′) as a function of the distance, for plotting a stationary kernel.',
    role: 'property',
    notes: ['covariance-functions'],
  },
  kernels.kernelProfile,
)

/** The functions of the module that are not kernels, keyed by name. */
export const kernelsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', kernels) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
