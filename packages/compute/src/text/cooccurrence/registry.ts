/** The functions of `aifn-compute/text/cooccurrence`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as c from './cooccurrence'
import * as coherence from './coherence'
import * as svd from './svd'

const fn = definer<FunctionInfo>('function', 'text/cooccurrence')
const NOTE = 'co-occurrence-matrices-and-pointwise-mutual-information'

fn(
  {
    key: 'cooccurrence',
    name: 'Co-occurrence matrix',
    role: 'estimator',
    returns: 'cooccurrence',
    summary: 'Word × context counts within a symmetric or asymmetric window, uniform or weighted by distance (HAL).',
    notes: [NOTE, 'distributional-semantics'],
    cite: ['turney2010'],
  },
  c.cooccurrence,
)
fn(
  {
    key: 'pmi',
    name: 'Pointwise mutual information',
    tex: '\\operatorname{PMI}(w, c)',
    role: 'transform',
    summary: 'log p(w, c) / (p(w) P_α(c)) for every cell of a count matrix.',
    notes: [NOTE],
    glossary: 'pmi',
    cite: ['church1990', 'levy2015'],
  },
  c.pmi,
)
fn(
  {
    key: 'ppmi',
    name: 'Positive (shifted) PMI',
    tex: '\\max(\\operatorname{PMI} - \\log k, 0)',
    role: 'transform',
    summary: 'PMI clipped at zero, optionally shifted by log k (SPPMI).',
    notes: [NOTE, 'word-embeddings'],
    cite: ['levy2015', 'levy2014'],
  },
  c.ppmi,
)
fn(
  {
    key: 'wordVectors',
    name: 'Word vectors by truncated SVD',
    role: 'transform',
    summary: 'The rows of U_d Σ_d^p of a word × context matrix.',
    notes: [NOTE, 'word-embeddings', 'latent-semantic-analysis'],
    cite: ['levy2015'],
  },
  c.wordVectors,
)

fn(
  {
    key: 'truncatedSvd',
    name: 'Truncated SVD',
    tex: 'A \\approx U_k \\Sigma_k V_k^\\top',
    role: 'transform',
    returns: 'truncated-svd',
    summary: 'The k largest singular triplets of a word × context or term × document matrix, with each energy share.',
    notes: ['latent-semantic-analysis', 'singular-value-decomposition', NOTE],
    cite: ['deerwester1990', 'levy2015'],
  },
  svd.truncatedSvd,
)

fn(
  {
    key: 'topicCoherence',
    name: 'Topic coherence (NPMI, UMass)',
    role: 'estimator',
    summary: 'How often each topic’s top words co-occur in documents: mean pairwise NPMI, or the UMass log ratio.',
    notes: ['topic-model-evaluation'],
    cite: ['bouma2009npmi', 'lau2014coherence', 'mimno2011coherence', 'roder2015coherence'],
  },
  coherence.topicCoherence,
)

/** The functions of the module, keyed by name. */
export const cooccurrenceFunctions = entries<FunctionInfo>('function', c, svd, coherence) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
