/** The registry of `aifn-methods/neural/quantisation`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as cost from './cost'
import * as qat from './qat'
import * as study from './study'

const fn = definer<FunctionInfo>('function', 'neural/quantisation')

fn(
  {
    key: 'servingMemory',
    name: 'Serving memory (weights and KV cache)',
    role: 'property',
    notes: ['model-quantisation', 'key-value-cache'],
  },
  cost.servingMemory,
)
fn(
  {
    key: 'decodeThroughput',
    name: 'Decoding throughput (roofline)',
    summary: 'Tokens per second bounded by memory bandwidth (weights read once per step) or by 2N FLOPs per token.',
    role: 'property',
    notes: ['model-quantisation', 'batching-and-serving-throughput', 'key-value-cache'],
  },
  cost.decodeThroughput,
)

fn(
  {
    key: 'adapterParameters',
    name: 'LoRA adapter parameters',
    tex: '\\Psi_a = L\\,r \\sum_{\\text{targets}} (d_{\\text{in}} + d_{\\text{out}})',
    summary: 'The trainable parameters of rank-r LoRA adapters on chosen linear layers of every block.',
    role: 'property',
    notes: ['low-rank-adaptation', 'bootstrapping-a-language-model-for-a-new-task', 'quantised-low-rank-adaptation'],
  },
  cost.adapterParameters,
)
fn(
  {
    key: 'linearParameters',
    name: 'Linear-layer parameters of a transformer',
    summary: 'The weights of the seven linear layers of every block: what QLoRA stores in 4 bits.',
    role: 'property',
    notes: ['quantised-low-rank-adaptation'],
  },
  cost.linearParameters,
)
fn(
  {
    key: 'fineTuningMemory',
    name: 'Fine-tuning memory (full, LoRA, QLoRA)',
    summary:
      'Weights, adapter and optimiser state, gradients, activations and logits of a fine-tuning run: 16Ψ, 2Ψ + 16Ψ_a or about 0.52Ψ + 16Ψ_a of model state.',
    role: 'property',
    notes: ['quantised-low-rank-adaptation', 'bootstrapping-a-language-model-for-a-new-task', 'low-rank-adaptation'],
    cite: ['dettmers2023', 'rajbhandari2020', 'korthikanti2022'],
  },
  cost.fineTuningMemory,
)
fn(
  {
    key: 'fineTuningCompute',
    name: 'Fine-tuning compute and price',
    summary:
      'FLOPs per token (6Ψ for full fine-tuning; about 6Ψ for checkpointed LoRA), run time at a model FLOP utilisation, and price.',
    role: 'property',
    notes: ['bootstrapping-a-language-model-for-a-new-task', 'quantised-low-rank-adaptation'],
    cite: ['kaplan2020'],
  },
  cost.fineTuningCompute,
)

fn(
  {
    key: 'quantisationStudy',
    name: 'Streamed quantisation study of a small MLP',
    summary:
      'Train a ReLU MLP by Adam, then quantise its weights per tensor, per channel, by GPTQ and by AWQ at each bit width.',
    role: 'simulation',
    random: true,
    notes: ['model-quantisation', 'multilayer-perceptron'],
    cite: ['nagel2021', 'frantar2023', 'lin2024'],
  },
  study.quantisationStudy,
)
fn(
  {
    key: 'quantisationAwareTraining',
    name: 'Quantisation-aware training against post-training quantisation',
    summary:
      'Train a ReLU MLP, then at each bit width round it once (PTQ) or fine-tune it with fake-quantised weights and the straight-through estimator (QAT).',
    role: 'simulation',
    random: true,
    notes: ['model-quantisation', 'multilayer-perceptron'],
    cite: ['jacob2018', 'bengio2013b'],
  },
  qat.quantisationAwareTraining,
)
fn(
  { key: 'studyModel', name: 'The quantisation study MLP', role: 'construction', notes: ['multilayer-perceptron'] },
  study.studyModel,
)

/** The functions of the module, keyed by name. */
export const quantisationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', cost, qat, study) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
