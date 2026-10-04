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
