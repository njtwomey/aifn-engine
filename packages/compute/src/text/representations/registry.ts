/** The functions of `aifn-compute/text/representations`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as lsa from './lsa'
import * as matrices from './matrices'
import * as ri from './random-indexing'

const fn = definer<FunctionInfo>('function', 'text/representations')
const DS = 'distributional-semantics'
const LSA = 'latent-semantic-analysis'
const PMI = 'co-occurrence-matrices-and-pointwise-mutual-information'

fn(
  {
    key: 'weightMatrix',
    name: 'Weighting of a term matrix',
    role: 'transform',
    summary: 'Raw, binary, log(1 + count), TF-IDF per column, or PPMI.',
    notes: [DS, 'term-frequency-inverse-document-frequency', PMI],
    cite: ['turney2010'],
  },
  matrices.weightMatrix,
)
fn(
  {
    key: 'termDocumentMatrix',
    name: 'Term–document matrix',
    role: 'transform',
    returns: 'term-matrix',
    summary: 'Words × documents, counted and weighted: the transpose of the bag of words, and the input of LSA.',
    notes: [DS, LSA, 'bag-of-words'],
    cite: ['turney2010', 'deerwester1990'],
  },
  matrices.termDocumentMatrix,
)
fn(
  {
    key: 'termTermMatrix',
    name: 'Term–term matrix',
    role: 'transform',
    returns: 'term-matrix',
    summary: 'Words × context words in a symmetric or asymmetric, optionally distance-weighted window, then weighted.',
    notes: [DS, PMI, 'word-embeddings'],
    cite: ['turney2010', 'levy2015'],
  },
  matrices.termTermMatrix,
)
fn(
  {
    key: 'lsa',
    name: 'Latent semantic analysis',
    tex: 'U_k \\Sigma_k,\\ V_k \\Sigma_k',
    role: 'transform',
    returns: 'lsa',
    summary: 'Term and document coordinates from the rank-k truncated SVD, with each component’s energy share.',
    notes: [LSA, DS, 'singular-value-decomposition'],
    cite: ['deerwester1990'],
  },
  lsa.lsa,
)
fn(
  {
    key: 'cosineSimilarities',
    name: 'Cosine similarity of every pair of rows',
    tex: '\\frac{x_i^\\top x_j}{\\lVert x_i \\rVert \\lVert x_j \\rVert}',
    role: 'property',
    summary: 'The rows’ pairwise cosines; zero rows are similar to nothing.',
    notes: ['pairwise-distances-and-cosine-similarity', DS],
    cite: ['turney2010'],
  },
  lsa.cosineSimilarities,
)
fn(
  {
    key: 'cosineMap',
    name: 'Cosine map of rows',
    role: 'transform',
    summary: 'Unit-length rows, centred and projected on their top principal axes: a plane picture of cosine geometry.',
    notes: ['word-embeddings', DS],
  },
  lsa.cosineMap,
)
fn(
  {
    key: 'nearestByCosine',
    name: 'Nearest rows by cosine',
    role: 'transform',
    summary: 'The rows most similar by cosine to a row or a query vector, ranked.',
    notes: ['word-embeddings', DS],
    cite: ['turney2010'],
  },
  lsa.nearestByCosine,
)
fn(
  {
    key: 'analogy',
    name: 'Analogy by vector offset',
    tex: '\\arg\\max_w \\cos(w, \\hat a - \\hat b + \\hat c)',
    role: 'transform',
    summary: 'The words nearest by cosine to a − b + c (unit vectors), the inputs excluded (3CosAdd).',
    notes: ['word-embeddings'],
    cite: ['mikolov2013', 'levy2014'],
  },
  lsa.analogy,
)
fn(
  {
    key: 'indexVector',
    name: 'Random index vector',
    role: 'construction',
    summary: 'A seeded sparse ternary vector per token: a few ±1 entries in d dimensions.',
    notes: [DS],
    cite: ['achlioptas2003'],
  },
  ri.indexVector,
)
fn(
  {
    key: 'randomIndexing',
    name: 'Random indexing',
    role: 'transform',
    returns: 'random-indexing',
    summary:
      'Word vectors as sums of the random index vectors of their contexts: a random projection of co-occurrence.',
    notes: [DS, 'word-embeddings'],
    cite: ['achlioptas2003'],
  },
  ri.randomIndexing,
)

/** The functions of the module, keyed by name. */
export const representationsFunctions = entries<FunctionInfo>('function', matrices, lsa, ri) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
