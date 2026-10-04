/** The language models of `aifn-methods/neural/language-models`, registered as models. */

import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { defineModel } from 'aifn-compute/learning/estimators'
import { charGpt } from './gpt'
import { kneserNey } from './ngram'

defineModel(
  {
    key: 'kneserNey',
    module: 'neural/language-models',
    name: 'Kneser–Ney n-gram language model',
    summary: 'Interpolated (optionally modified) Kneser–Ney smoothing of n-gram counts, with continuation counts.',
    task: 'density',
    capabilities: ['score'],
    hyper: space({ order: int(1, 8, { default: 3 }), modified: bool() }),
    notes: ['n-gram-language-model', 'cross-entropy-and-perplexity'],
    cite: ['kneser1995'],
  },
  kneserNey,
)
defineModel(
  {
    key: 'charGpt',
    module: 'neural/language-models',
    name: 'Tiny character-level GPT',
    summary: 'A decoder-only transformer trained by next-character prediction, small enough to train in a browser.',
    task: 'density',
    capabilities: ['score'],
    hyper: space({
      width: int(8, 128, { default: 32 }),
      layers: int(1, 6, { default: 2 }),
      heads: int(1, 8, { default: 4 }),
      context: int(4, 128, { default: 32 }),
      position: oneOf(['learned', 'sinusoidal', 'rope', 'alibi', 'none']),
      steps: int(1, 5000, { default: 300 }),
      stepSize: real(1e-4, 0.1, { default: 0.01, scale: 'log' }),
    }),
    notes: ['autoregressive-transformer-language-model', 'decoder-only-transformers', 'neural-language-model'],
    cite: ['radford2019', 'vaswani2017', 'press2017'],
  },
  charGpt,
)

export { charGpt, kneserNey }
