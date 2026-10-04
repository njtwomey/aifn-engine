/** The functions of `aifn-compute/nn/quantise`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as quantise from './quantise'

const fn = definer<FunctionInfo>('function', 'nn/quantise')
const Q = ['model-quantisation']

fn(
  {
    key: 'quantisationParams',
    name: 'Quantisation parameters (min–max)',
    summary: 'Scale and zero point of an affine or symmetric b-bit quantiser, per tensor or per channel.',
    role: 'estimator',
    notes: [...Q, 'quantisation'],
    cite: ['jacob2018', 'nagel2021'],
  },
  quantise.quantisationParams,
)
fn(
  {
    key: 'quantise',
    name: 'Quantise',
    tex: 'q = \\operatorname{clamp}(\\lfloor x/s \\rceil + z,\\ q_{\\min},\\ q_{\\max})',
    role: 'transform',
    notes: [...Q, 'quantisation'],
    cite: ['jacob2018'],
  },
  quantise.quantise,
)
fn(
  {
    key: 'dequantise',
    name: 'Dequantise',
    tex: '\\hat x = s(q - z)',
    role: 'transform',
    notes: Q,
    cite: ['jacob2018'],
  },
  quantise.dequantise,
)
fn(
  {
    key: 'fakeQuantise',
    name: 'Fake quantisation (straight-through)',
    summary: 'Quantise and dequantise forward; pass the gradient through where x is in range (custom VJP).',
    role: 'transform',
    notes: Q,
    cite: ['bengio2013b', 'jacob2018'],
  },
  quantise.fakeQuantise,
)
fn(
  { key: 'quantisationError', name: 'Quantisation error and SQNR', role: 'property', notes: [...Q, 'quantisation'] },
  quantise.quantisationError,
)
fn(
  {
    key: 'quantisedMatmul',
    name: 'Integer matmul (simulated)',
    summary: 'Affine activations times symmetric per-column weights in integer arithmetic, then rescaled.',
    role: 'transform',
    notes: Q,
    cite: ['jacob2018', 'dettmers2022'],
  },
  quantise.quantisedMatmul,
)
fn(
  {
    key: 'gptqQuantise',
    name: 'GPTQ (second-order rounding)',
    summary: 'Quantise a layer column by column, feeding each rounding error forward through the inverse Hessian.',
    role: 'solver',
    notes: Q,
    cite: ['frantar2023'],
  },
  quantise.gptqQuantise,
)
fn(
  {
    key: 'awqQuantise',
    name: 'AWQ (activation-aware scaling)',
    summary:
      'Scale input channels by their mean activation to a searched power before rounding, then divide it back out.',
    tex: '\\hat W = Q(W\\operatorname{diag}(s))\\operatorname{diag}(s)^{-1},\\ s_j = \\overline{|X_{\\cdot j}|}^{\\alpha}',
    role: 'solver',
    notes: Q,
    cite: ['lin2024'],
  },
  quantise.awqQuantise,
)
fn(
  { key: 'roundHalfEven', name: 'Round half to even', role: 'transform', notes: ['quantisation'] },
  quantise.roundHalfEven,
)
fn(
  { key: 'integerRange', name: 'Integer range of b bits', role: 'property', notes: ['quantisation'] },
  quantise.integerRange,
)

/** The functions of the module, keyed by name. */
export const quantiseFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', quantise) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
