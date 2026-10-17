/** The functions and layer constructors of `aifn-compute/nn/sequence`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as deltaRule from './delta-rule'
import * as gradients from './gradients'
import * as hippo from './hippo'
import * as hyena from './hyena'
import * as linearAttention from './linear-attention'
import * as lru from './lru'
import * as mlstm from './mlstm'
import * as rgLru from './rg-lru'
import * as scanOrders from './scan-orders'
import * as selective from './selective'
import * as seq2seq from './seq2seq'
import * as ssd from './ssd'
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
    key: 'linearAttentionWeights',
    name: 'Linear attention weights',
    tex: 'W = \\mathrm{rownorm}(\\phi(Q)\\phi(K)^\\top)',
    role: 'transform',
    summary: 'The T × T weight matrix of linear attention, causal and optionally decayed, rows normalised.',
    notes: ['linear-attention'],
    cite: ['katharopoulos2020'],
  },
  linearAttention.linearAttentionWeights,
)
fn(
  {
    key: 'positiveRandomFeatures',
    name: 'Positive random features (FAVOR+)',
    tex: '\\phi(x) = m^{-1/2}\\exp(\\Omega x - \\lVert x \\rVert^2/2)',
    role: 'construction',
    summary: 'The Performer feature map whose inner products estimate the softmax kernel exp(qᵀk) without bias.',
    notes: ['linear-attention'],
    cite: ['choromanski2021'],
    random: true,
  },
  linearAttention.positiveRandomFeatures,
)
fn(
  {
    key: 'deltaRule',
    name: 'Delta rule (DeltaNet, Gated DeltaNet)',
    tex: 'S_t = \\alpha_t S_{t-1} + \\beta_t (v_t - \\alpha_t S_{t-1} k_t) k_t^\\top',
    role: 'transform',
    summary: 'A fast-weight memory that takes one gradient step per token on the key–value regression loss.',
    notes: ['delta-rule-and-test-time-training', 'long-range-and-recall-benchmarks'],
    cite: ['schlag2021', 'yang2024deltanet', 'yang2025gateddeltanet'],
  },
  deltaRule.deltaRule,
)
fn(
  {
    key: 'mlstmRecurrent',
    name: 'mLSTM (recurrent form)',
    tex: 'C_t = f_t C_{t-1} + i_t v_t k_t^\\top',
    role: 'transform',
    summary: 'The xLSTM matrix memory with exponential input gates and the max stabiliser, by an associative scan.',
    notes: ['xlstm'],
    cite: ['beck2024xlstm'],
  },
  mlstm.mlstmRecurrent,
)
fn(
  {
    key: 'mlstmParallel',
    name: 'mLSTM (parallel form)',
    role: 'transform',
    summary: 'The mLSTM as a stabilised, gated attention matrix, equal to the recurrent form.',
    notes: ['xlstm'],
    cite: ['beck2024xlstm'],
  },
  mlstm.mlstmParallel,
)
fn(
  {
    key: 'rgLru',
    name: 'Real-gated linear recurrent unit',
    tex: 'h_t = a_t \\odot h_{t-1} + \\sqrt{1 - a_t^2} \\odot (i_t \\odot x_t)',
    role: 'transform',
    summary: 'Griffin’s RG-LRU: an input-gated decay a_t = a^{c r_t}, with the input scaled to keep the state’s size.',
    notes: ['gated-linear-recurrence'],
    cite: ['de2024griffin'],
  },
  rgLru.rgLru,
)
fn(
  {
    key: 'rgLruLambda',
    name: 'RG-LRU decay parameter',
    tex: '\\Lambda = \\operatorname{logit}((a^c)^{1/c})',
    role: 'construction',
    notes: ['gated-linear-recurrence'],
    cite: ['de2024griffin'],
  },
  rgLru.rgLruLambda,
)
fn(
  {
    key: 'lruRingInit',
    name: 'LRU ring initialisation',
    role: 'construction',
    summary: 'Eigenvalues drawn uniformly by area on a ring r_min ≤ |λ| ≤ r_max, with the γ normalisation.',
    notes: ['linear-recurrent-unit-and-s5'],
    cite: ['orvieto2023'],
    random: true,
  },
  lru.lruRingInit,
)
fn(
  {
    key: 'lruEigenvalues',
    name: 'LRU eigenvalues',
    tex: '\\lambda_j = \\exp(-e^{\\nu_j} + i e^{\\theta_j})',
    role: 'property',
    notes: ['linear-recurrent-unit-and-s5'],
    cite: ['orvieto2023'],
  },
  lru.lruEigenvalues,
)
fn(
  { key: 'lruLambda', name: 'LRU eigenvalues (complex)', role: 'property', notes: ['linear-recurrent-unit-and-s5'] },
  lru.lruLambda,
)
fn(
  {
    key: 'lruScan',
    name: 'Linear recurrent unit (scan)',
    tex: 'h_k = \\lambda \\odot h_{k-1} + \\gamma \\odot B u_k',
    role: 'transform',
    summary: 'The LRU’s complex diagonal recurrence by an associative scan, read out as Re(C h_k) + D u_k.',
    notes: ['linear-recurrent-unit-and-s5'],
    cite: ['orvieto2023'],
  },
  lru.lruScan,
)
fn(
  {
    key: 'lruKernel',
    name: 'LRU convolution kernel',
    tex: 'K_k = \\operatorname{Re}\\sum_j C_j \\gamma_j \\lambda_j^k B_j',
    role: 'construction',
    notes: ['linear-recurrent-unit-and-s5'],
    cite: ['orvieto2023'],
  },
  lru.lruKernel,
)
fn(
  {
    key: 'lruConvolve',
    name: 'Linear recurrent unit (convolution)',
    role: 'transform',
    notes: ['linear-recurrent-unit-and-s5'],
    cite: ['orvieto2023'],
  },
  lru.lruConvolve,
)
fn(
  {
    key: 'hippoLegT',
    name: 'HiPPO-LegT matrices',
    role: 'construction',
    summary: 'The A and B under which a state tracks the Legendre coefficients of a sliding window of the input.',
    notes: ['hippo'],
    cite: ['gu2020hippo'],
  },
  hippo.hippoLegT,
)
fn(
  {
    key: 'hippoOnline',
    name: 'HiPPO memory, online',
    role: 'simulation',
    summary: 'Run the LegS or LegT system over a sampled input by the bilinear rule, every state returned.',
    notes: ['hippo'],
    cite: ['gu2020hippo'],
  },
  hippo.hippoOnline,
)
fn({ key: 'hippoWindow', name: 'HiPPO window', role: 'property', notes: ['hippo'] }, hippo.hippoWindow)
fn(
  {
    key: 'hippoBasis',
    name: 'HiPPO orthonormal Legendre basis',
    tex: 'g_n(x) = \\sqrt{2n+1}\\,P_n(z)',
    role: 'construction',
    notes: ['hippo'],
    cite: ['gu2020hippo'],
  },
  hippo.hippoBasis,
)
fn(
  {
    key: 'hippoReconstruct',
    name: 'HiPPO reconstruction',
    tex: 'u(x) \\approx \\sum_n h_n g_n(x)',
    role: 'transform',
    notes: ['hippo'],
    cite: ['gu2020hippo'],
  },
  hippo.hippoReconstruct,
)
fn(
  {
    key: 'hippoProject',
    name: 'HiPPO projection',
    role: 'transform',
    summary: 'The exact Legendre coefficients of the input over the window, which the online state approximates.',
    notes: ['hippo'],
    cite: ['gu2020hippo'],
  },
  hippo.hippoProject,
)
fn(
  {
    key: 'hyenaPositionalEncoding',
    name: 'Hyena positional encoding',
    role: 'construction',
    notes: ['long-convolution-models'],
    cite: ['poli2023hyena'],
  },
  hyena.hyenaPositionalEncoding,
)
fn(
  {
    key: 'implicitFilter',
    name: 'Implicit long filter',
    tex: 'h_t = w(t)\\,\\mathrm{FFN}(\\mathrm{PE}(t))',
    role: 'construction',
    summary: 'A filter as long as the sequence generated by a small sine network of the position, under a window.',
    notes: ['long-convolution-models'],
    cite: ['poli2023hyena'],
    random: true,
  },
  hyena.implicitFilter,
)
fn(
  {
    key: 'hyenaOperator',
    name: 'Hyena operator',
    tex: 'z^{n+1} = x^n \\odot (h^n * z^n)',
    role: 'transform',
    notes: ['long-convolution-models'],
    cite: ['poli2023hyena', 'fu2023h3'],
  },
  hyena.hyenaOperator,
)
fn(
  {
    key: 'semiseparableMask',
    name: '1-semiseparable mask',
    tex: 'L_{ji} = \\prod_{k=i+1}^{j} a_k',
    role: 'construction',
    notes: ['mamba-2-and-state-space-duality'],
    cite: ['dao2024'],
  },
  ssd.semiseparableMask,
)
fn(
  {
    key: 'ssdQuadratic',
    name: 'State space duality (quadratic form)',
    tex: 'Y = (L \\circ C B^\\top) X',
    role: 'transform',
    notes: ['mamba-2-and-state-space-duality'],
    cite: ['dao2024'],
  },
  ssd.ssdQuadratic,
)
fn(
  {
    key: 'ssdRecurrent',
    name: 'State space duality (recurrence)',
    role: 'transform',
    notes: ['mamba-2-and-state-space-duality'],
    cite: ['dao2024'],
  },
  ssd.ssdRecurrent,
)
fn(
  {
    key: 'ssdChunked',
    name: 'Chunked SSD algorithm',
    role: 'transform',
    summary: 'Mamba-2’s block decomposition: quadratic within chunks, a recurrence over chunk states between them.',
    notes: ['mamba-2-and-state-space-duality'],
    cite: ['dao2024'],
  },
  ssd.ssdChunked,
)
fn(
  {
    key: 'scanOrder',
    name: 'Scan order of a grid',
    role: 'construction',
    summary: 'The order in which a sequence layer reads the cells of a grid: raster, column, snake or Hilbert.',
    notes: ['state-space-models-beyond-text'],
    cite: ['zhu2024vim', 'liu2024vmamba'],
  },
  scanOrders.scanOrder,
)
fn(
  {
    key: 'routePositions',
    name: 'Positions along scan routes',
    role: 'property',
    notes: ['state-space-models-beyond-text'],
  },
  scanOrders.routePositions,
)
fn(
  {
    key: 'neighbourPairs',
    name: 'Grid neighbour pairs',
    role: 'construction',
    notes: ['state-space-models-beyond-text'],
  },
  scanOrders.neighbourPairs,
)
fn(
  {
    key: 'sequenceDistance',
    name: 'Sequence distance under a scan',
    role: 'property',
    notes: ['state-space-models-beyond-text'],
    cite: ['liu2024vmamba'],
  },
  scanOrders.sequenceDistance,
)
fn(
  {
    key: 'visibleTo',
    name: 'Visibility under a causal scan',
    role: 'property',
    notes: ['state-space-models-beyond-text'],
  },
  scanOrders.visibleTo,
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
  deltaRule,
  gradients,
  hippo,
  hyena,
  linearAttention,
  lru,
  mlstm,
  rgLru,
  scanOrders,
  selective,
  seq2seq,
  ssd,
  ssm,
) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>
