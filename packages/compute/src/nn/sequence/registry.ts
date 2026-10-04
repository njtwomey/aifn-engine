/** The functions and layer constructors of `aifn-compute/nn/sequence`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as gradients from './gradients'
import * as linearAttention from './linear-attention'
import * as selective from './selective'
import * as seq2seq from './seq2seq'
import * as ssm from './ssm'

const fn = definer<FunctionInfo>('function', 'nn/sequence')
const ssmNotes = ['structured-state-space-models', 'deep-state-space-models']

fn(
  {
    key: 'gradientsThroughTime',
    name: 'Backpropagation through time with gradient norms',
    role: 'transform',
    summary: 'The loss gradient with respect to every hidden state of an unrolled cell, and its norm per step.',
    notes: ['backpropagation-through-time', 'vanishing-and-exploding-gradients'],
    cite: ['bengio1994', 'pascanu2013'],
  },
  gradients.gradientsThroughTime,
)
fn(
  {
    key: 'discretiseDiagonal',
    name: 'Diagonal state-space discretisation',
    role: 'transform',
    notes: [...ssmNotes, 'mamba'],
    cite: ['gu2022s4d', 'gu2023mamba'],
  },
  ssm.discretiseDiagonal,
)
fn(
  {
    key: 'hippoLegS',
    name: 'HiPPO-LegS matrices',
    role: 'construction',
    summary: 'The A and B under which a state tracks the Legendre coefficients of the whole input history.',
    notes: ssmNotes,
    cite: ['gu2020hippo'],
  },
  ssm.hippoLegS,
)
fn(
  {
    key: 'ssmKernel',
    name: 'State-space convolution kernel',
    tex: '\\bar K = (C\\bar B, C\\bar A\\bar B, \\dots, C\\bar A^{L-1}\\bar B)',
    role: 'construction',
    notes: ssmNotes,
    cite: ['gu2022s4'],
  },
  ssm.ssmKernel,
)
fn(
  { key: 'causalConvolution', name: 'Causal convolution', role: 'transform', notes: ssmNotes, cite: ['gu2022s4'] },
  ssm.causalConvolution,
)
fn(
  {
    key: 'linearRecurrence',
    name: 'Linear recurrence by parallel scan',
    tex: 'h_t = a_t \\odot h_{t-1} + b_t',
    role: 'transform',
    notes: ['parallel-scan', 'mamba', 'linear-attention'],
    cite: ['blelloch1990'],
  },
  ssm.linearRecurrence,
)
fn(
  {
    key: 'matrixRecurrence',
    name: 'Matrix linear recurrence by parallel scan',
    tex: 'h_t = A_t h_{t-1} + b_t',
    role: 'transform',
    notes: ['parallel-scan', 'structured-state-space-models'],
    cite: ['blelloch1990'],
  },
  ssm.matrixRecurrence,
)
fn(
  {
    key: 'ssmRecurrent',
    name: 'State-space layer in recurrent mode',
    role: 'transform',
    notes: ssmNotes,
    cite: ['gu2022s4'],
  },
  ssm.ssmRecurrent,
)
fn(
  {
    key: 'selectiveScan',
    name: 'Selective scan',
    role: 'transform',
    summary: 'Mamba’s input-dependent diagonal state-space recurrence, run by an associative scan.',
    notes: ['mamba', 'hybrid-sequence-models'],
    cite: ['gu2023mamba'],
  },
  selective.selectiveScan,
)
fn(
  {
    key: 'SelectiveSsm',
    name: 'Selective state-space layer',
    role: 'construction',
    notes: ['mamba'],
    cite: ['gu2023mamba'],
  },
  selective.SelectiveSsm,
)
fn(
  {
    key: 'linearAttention',
    name: 'Linear attention',
    tex: '\\phi(Q)\\phi(K)^\\top V',
    role: 'transform',
    summary: 'Attention with a feature-map kernel φ(q)·φ(k), causal and optionally decayed, in its parallel form.',
    notes: ['linear-attention', 'efficient-attention', 'rwkv-and-retentive-networks'],
    cite: ['katharopoulos2020'],
  },
  linearAttention.linearAttention,
)
fn(
  {
    key: 'linearAttentionRecurrent',
    name: 'Linear attention as a recurrent network',
    role: 'transform',
    summary: 'The same outputs from a matrix-valued state S_t = γS_{t−1} + φ(k_t)v_tᵀ, by a scan.',
    notes: ['linear-attention', 'rwkv-and-retentive-networks'],
    cite: ['katharopoulos2020'],
  },
  linearAttention.linearAttentionRecurrent,
)
fn(
  {
    key: 'eluFeatureMap',
    name: 'ELU + 1 feature map',
    role: 'transform',
    notes: ['linear-attention'],
    cite: ['katharopoulos2020'],
  },
  linearAttention.eluFeatureMap,
)
fn(
  {
    key: 'BahdanauAttention',
    name: 'Bahdanau (additive) attention',
    tex: 'v^\\top\\tanh(W_q s + W_k h_j)',
    role: 'construction',
    notes: ['bahdanau-attention', 'sequence-to-sequence'],
    cite: ['bahdanau2015'],
  },
  seq2seq.BahdanauAttention,
)
fn(
  {
    key: 'LuongAttention',
    name: 'Luong (multiplicative) attention',
    role: 'construction',
    notes: ['sequence-to-sequence', 'bahdanau-attention'],
    cite: ['luong2015'],
  },
  seq2seq.LuongAttention,
)

/** Every function and layer constructor of the module, keyed by name. */
export const sequenceFunctions = entries<FunctionInfo>(
  'function',
  gradients,
  linearAttention,
  selective,
  seq2seq,
  ssm,
) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
