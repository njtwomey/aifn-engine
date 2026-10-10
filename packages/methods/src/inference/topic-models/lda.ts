/**
 * Latent Dirichlet allocation (Blei, Ng and Jordan, 2003, JMLR 3): the model in the model language (a structured graph
 * with the topic, document and word plates as groups), its match, collapsed Gibbs sampling (Griffiths and Steyvers,
 * 2004, PNAS 101), and the engine to pass to `aifn-compute/inference/engines`' `infer`.
 *
 * Each topic is a distribution over the $V$ words, $\phivec_k \sim \Dir(\beta)$; each document mixes the $K$ topics,
 * $\thetavec_d \sim \Dir(\alpha)$; and each token picks a topic $z_{dn} \sim \Cat(\thetavec_d)$ and then a word
 * $w_{dn} \sim \Cat(\phivec_{z_{dn}})$. Both Dirichlet priors are symmetric. The sampler integrates out
 * $\thetavec$ and $\phivec$ and keeps only the topic assignments and the counts they imply: $n_{dk}$ (tokens of
 * document $d$ in topic $k$), $n_{kw}$ (tokens of word $w$ in topic $k$) and $n_k = \sum_w n_{kw}$.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { categorical, child, integers } from 'aifn-compute/foundation/random'
import { logGamma } from 'aifn-compute/numerics/special'
import { fromData, isTensor, toFlat, type Matrix, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  argRefs,
  dist,
  model as describe,
  plateChain,
  type Arg,
  type Bindings,
  type Model,
  type Nested,
} from 'aifn-compute/inference/model'
import { builtInEngines, withEngines, type EngineRegistration, type EngineTable } from 'aifn-compute/inference/engines'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── LDA ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * LDA in the model language: topics $\phivec_k \sim \Dir(\beta)$, mixtures $\thetavec_d \sim \Dir(\alpha)$,
 * $z_{dn} \sim \Cat(\thetavec_d)$ and $w_{dn} \sim \Cat(\phivec_{z_{dn}})$, with sizes `K` and `V`, constants `α` and
 * `β`, and plates of topics, documents and words. The same model object is returned on every call.
 *
 * @returns The model, which `matchLda` recognises and `ldaEngine` runs.
 *
 * @example The nodes of the model, and what `matchLda` finds in it
 * const m = ldaModel()
 * print('nodes:', m.attributes.map((n) => `${n.name} (${n.role}, in ${n.group})`))
 * print('match:', matchLda(m))
 */
export function ldaModel(): Model {
  return ldaDescription
}

/** The model `ldaModel` returns, built once. */
const ldaDescription: Model = describe('Latent Dirichlet allocation', (m) => {
  const K = m.size('K')
  const V = m.size('V')
  const topics = m.plate('topics', K, { label: 'K' })
  const docs = m.plate('documents', 'D', { label: 'D' })
  const words = docs.plate('words', 'N', { label: 'N_d' })
  const alpha = m.constant('α', undefined, { label: '\\alpha' })
  const beta = m.constant('β', undefined, { label: '\\beta' })
  const phi = topics.variable('φ', dist.Dirichlet(beta, V), { label: '\\boldsymbol{\\phi}_k' })
  const theta = docs.variable('θ', dist.Dirichlet(alpha, K), { label: '\\boldsymbol{\\theta}_d' })
  const z = words.variable('z', dist.Categorical(theta), { label: 'z_{dn}' })
  words.observed('w', dist.Categorical(phi.at(z)), { label: 'w_{dn}' })
})

/**
 * The names of an LDA-shaped model's nodes and its hyperparameters, or null when the model is not LDA-shaped. A model
 * is LDA-shaped when it has an observed categorical $w$ whose probabilities are a Dirichlet node $\phivec$ selected by
 * a latent categorical $z$ in the same plate, and $z$'s probabilities are a Dirichlet node $\thetavec$ in a plate that
 * encloses $z$'s.
 *
 * @param m The model to examine.
 * @returns The node names `phi`, `theta`, `z` and `w`; `alpha` and `beta`, the first arguments of the Dirichlets on
 *   $\thetavec$ and $\phivec$; and `K` and `V`, their second arguments (0 when absent). Each of the last four is an
 *   argument as the model holds it: a number, or a reference to a constant or a size. Null when the model does not
 *   match.
 *
 * @example The LDA model matches; its arguments are references to constants and sizes
 * print(matchLda(ldaModel()))
 */
export function matchLda(
  m: Model,
): { phi: string; theta: string; z: string; w: string; alpha: Arg; beta: Arg; K: Arg; V: Arg } | null {
  const nodes = m.attributes
  const w = nodes.find(
    (n) =>
      n.role === 'observed' &&
      n.data?.dist?.family === 'Categorical' &&
      typeof n.data.dist.args[0] === 'object' &&
      'select' in (n.data.dist.args[0] as object),
  )
  if (!w) return null
  const ref = w.data!.dist!.args[0] as { node: string; select: string }
  const phi = nodes.find((n) => n.name === ref.node)
  const z = nodes.find((n) => n.name === ref.select)
  const phiDist = phi?.data?.dist
  const zDist = z?.data?.dist
  if (!phi || !z || phiDist?.family !== 'Dirichlet' || zDist?.family !== 'Categorical' || z.role !== 'latent')
    return null
  const thetaName = argRefs(zDist.args)[0]
  const theta = nodes.find((n) => n.name === thetaName)
  const thetaDist = theta?.data?.dist
  if (!theta || thetaDist?.family !== 'Dirichlet' || z.group !== w.group) return null
  // θ is per document, z and w per word inside the document plate.
  if (!plateChain(m, z.group).includes(theta.group ?? '')) return null
  return {
    phi: phi.name,
    theta: theta.name,
    z: z.name,
    w: w.name,
    alpha: thetaDist.args[0],
    beta: phiDist.args[0],
    K: thetaDist.args[1] ?? 0,
    V: phiDist.args[1] ?? 0,
  }
}

/** The problem of {@link ldaCollapsedGibbs}. */
export interface LdaOptions {
  /** Documents as arrays of word ids in $\{0, \dots, V - 1\}$. */
  documents: readonly (readonly number[])[]
  /** The number of topics $K$. */
  topics: number
  /** The vocabulary size $V$. */
  vocabulary: number
  /** $\alpha$, the symmetric Dirichlet concentration on each document's proportions $\thetavec_d$. */
  alpha: number
  /** $\beta$, the symmetric Dirichlet concentration on each topic's word distribution $\phivec_k$. */
  beta: number
  /**
   * The topics each document may use (labelled LDA, Ramage et al., 2009: a document's labels are its topics); default
   * every topic for every document.
   */
  allowed?: readonly (readonly number[])[]
}

/** The state of collapsed Gibbs sampling for LDA: topic assignments and the count tables they imply. */
export interface LdaState extends Status {
  /** Sweeps done. */
  t: number
  /** The topic $z_{dn}$ of every token, one int32 vector per document. */
  assignments: Tensor[]
  /** The counts $n_{dk}$ of each document's tokens in each topic, $D \times K$. */
  docTopic: Matrix
  /** The counts $n_{kw}$ of each word's tokens in each topic, $K \times V$. */
  topicWord: Matrix
  /** The tokens in each topic, $n_k = \sum_w n_{kw}$ ($K$ values). */
  topicTotals: Tensor
  /** $\log p(\wvec \mid \zvec)$ with $\phivec$ integrated out (`ldaLogLikelihood`). */
  logLikelihood: number
}

/**
 * $\log p(\wvec \mid \zvec)$ with the topics $\phivec_k \sim \Dir(\beta)$ integrated out, from the topic-word counts:
 * $\sum_k [\log\Gamma(V\beta) - V\log\Gamma(\beta) - \log\Gamma(n_k + V\beta) + \sum_w \log\Gamma(n_{kw} + \beta)]$.
 * Shared with the HDP sampler.
 *
 * @param nkw The counts $n_{kw}$, $K \times V$ row-major.
 * @param nk The topic totals $n_k$, $K$ values.
 * @param K The number of topics.
 * @param V The vocabulary size.
 * @param beta The symmetric Dirichlet concentration $\beta$ on the topics.
 * @returns The log marginal likelihood of the words given the assignments.
 */
export function ldaLogLikelihood(
  nkw: ArrayLike<number>,
  nk: ArrayLike<number>,
  K: number,
  V: number,
  beta: number,
): number {
  let total = 0
  for (let k = 0; k < K; k++) {
    total += logGamma(V * beta) - V * logGamma(beta) - logGamma(nk[k] + V * beta)
    for (let w = 0; w < V; w++) total += logGamma(nkw[k * V + w] + beta)
  }
  return total
}

/**
 * Collapsed Gibbs sampling for LDA as a traceable algorithm (Griffiths and Steyvers, 2004): $\thetavec$ and $\phivec$
 * are integrated out, and each step (a sweep) resamples every token's topic from
 * $p(z = k \mid \text{rest}) \propto (n_{dk} + \alpha)(n_{kw} + \beta)/(n_k + V\beta)$, with the token's own counts
 * removed. The initial topics are drawn uniformly from the `init` stream; token $(d, n)$ in a sweep draws from
 * `child(ctx.stream, d, n)`. No start. With `allowed`, each document's tokens are drawn only from its own topics:
 * labelled LDA, where the labels of a document are the topics it may use. The state is flagged `diverged` when its
 * log-likelihood is NaN.
 *
 * @param options The corpus, the number of topics, the priors and, for labelled LDA, each document's topics. Throws
 *   `DomainError` when a document's allowed set is empty or holds a topic outside $\{0, \dots, K - 1\}$.
 * @returns The algorithm, run with no start; read point estimates of a state with `ldaEstimates`.
 *
 * @example Two vocabularies, two topics: a few sweeps separate them
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const options = { documents, topics: 2, vocabulary: 6, alpha: 0.5, beta: 0.1 }
 * const start = run(ldaCollapsedGibbs(options), undefined, 0, { stream: stream(0) })
 * const final = run(ldaCollapsedGibbs(options), undefined, 20, { stream: stream(0) })
 * print('counts n_dk at the start', start.docTopic)
 * print('counts n_dk after 20 sweeps', final.docTopic)
 * print('log p(w | z):', start.logLikelihood, '->', final.logLikelihood)
 *
 * @example Labelled LDA: each document restricted to its label's topic
 * const documents = [[0, 1, 2, 0], [1, 2, 0, 2], [3, 4, 5, 3], [4, 5, 3, 5]]
 * const options = { documents, topics: 2, vocabulary: 6, alpha: 0.5, beta: 0.1, allowed: [[0], [0], [1], [1]] }
 * const final = run(ldaCollapsedGibbs(options), undefined, 3, { stream: stream(0) })
 * print('counts n_dk', final.docTopic)
 * print('top words', topWords(ldaEstimates(final, options).topicWord, 3))
 */
export function ldaCollapsedGibbs(options: LdaOptions): Algorithm<void, LdaState> {
  const { topics: K, vocabulary: V, alpha, beta, documents } = options
  const D = documents.length
  const every = Array.from({ length: K }, (_, k) => k)
  const allowed = documents.map((_, d) => options.allowed?.[d] ?? every)
  for (const a of allowed)
    if (a.length === 0 || a.some((k) => !(Number.isInteger(k) && k >= 0 && k < K)))
      throw new DomainError(
        'ldaCollapsedGibbs',
        'ldaCollapsedGibbs: each document needs a non-empty set of topics in 0 … K − 1',
      )
  return {
    name: 'lda-collapsed-gibbs',
    init: (_start, s) => {
      const ndk = new Float64Array(D * K)
      const nkw = new Float64Array(K * V)
      const nk = new Float64Array(K)
      const assignments = documents.map((doc, d) => {
        const z = Int32Array.from(doc, (w, n) => {
          const k = allowed[d][integers(child(s, d, n), allowed[d].length)]
          ndk[d * K + k]++
          nkw[k * V + w]++
          nk[k]++
          return k
        })
        return fromData(z, [z.length])
      })
      return {
        t: 0,
        assignments,
        docTopic: fromData(ndk, [D, K]),
        topicWord: fromData(nkw, [K, V]),
        topicTotals: fromData(nk, [K]),
        logLikelihood: ldaLogLikelihood(nkw, nk, K, V, beta),
      }
    },
    step: (s, ctx) => {
      const ndk = Float64Array.from(s.docTopic.data)
      const nkw = Float64Array.from(s.topicWord.data)
      const nk = Float64Array.from(s.topicTotals.data)
      const p = new Float64Array(K)
      const assignments = documents.map((doc, d) => {
        const z = Int32Array.from(s.assignments[d].data)
        doc.forEach((w, n) => {
          const old = z[n]
          ndk[d * K + old]--
          nkw[old * V + w]--
          nk[old]--
          const topics = allowed[d]
          const q = p.subarray(0, topics.length)
          topics.forEach((k, j) => (q[j] = ((ndk[d * K + k] + alpha) * (nkw[k * V + w] + beta)) / (nk[k] + V * beta)))
          const k = topics[categorical(child(ctx.stream, d, n), q)]
          z[n] = k
          ndk[d * K + k]++
          nkw[k * V + w]++
          nk[k]++
        })
        return fromData(z, [z.length])
      })
      const logLikelihood = ldaLogLikelihood(nkw, nk, K, V, beta)
      return {
        t: s.t + 1,
        assignments,
        docTopic: fromData(ndk, s.docTopic.shape),
        topicWord: fromData(nkw, s.topicWord.shape),
        topicTotals: fromData(nk, s.topicTotals.shape),
        logLikelihood,
        diverged: Number.isNaN(logLikelihood),
      }
    },
  }
}

/**
 * Point estimates from a state of `ldaCollapsedGibbs`: the posterior means given the assignments,
 * $\phi_{kw} = (n_{kw} + \beta)/(n_k + V\beta)$ and $\theta_{dk} = (n_{dk} + \alpha)/(N_d + K\alpha)$, with $N_d$ the
 * length of document $d$. Each row sums to 1 (for labelled LDA too, where topics a document may not use keep their
 * prior share).
 *
 * @param s The sampler's state.
 * @param options The options the sampler was built with; `topics`, `vocabulary`, `alpha`, `beta` and the document
 *   lengths are read.
 * @returns `topicWord`, $\phivec$ as a $K \times V$ matrix, and `docTopic`, $\thetavec$ as a $D \times K$ matrix.
 *
 * @example Topics and proportions after a few sweeps
 * const documents = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 2, 1],
 *   [3, 4, 5, 3, 4, 5], [4, 5, 3, 3, 5, 4], [5, 3, 4, 4, 3, 5]]
 * const options = { documents, topics: 2, vocabulary: 6, alpha: 0.5, beta: 0.1 }
 * const { topicWord, docTopic } = ldaEstimates(run(ldaCollapsedGibbs(options), undefined, 20), options)
 * print('top words of each topic', topWords(topicWord, 3))
 * print('topic proportions', docTopic)
 */
export function ldaEstimates(s: LdaState, options: LdaOptions): { topicWord: Matrix; docTopic: Matrix } {
  const { topics: K, vocabulary: V, alpha, beta } = options
  const phi = s.topicWord.data.map((n, i) => (n + beta) / (s.topicTotals.data[Math.floor(i / V)] + V * beta))
  const theta = s.docTopic.data.map((n, i) => {
    const d = Math.floor(i / K)
    return (n + alpha) / (options.documents[d].length + K * alpha)
  })
  return { topicWord: fromData(phi, [K, V]), docTopic: fromData(theta, s.docTopic.shape) }
}

/**
 * The number an argument of the model stands for under the bindings: a literal, a size, or a constant's bound value.
 * Throws `DomainError` for anything else (a constant without a numeric binding, or a node).
 *
 * @param a The argument, as the model holds it.
 * @param b The bindings, whose `sizes` and `constants` are read.
 * @returns The number.
 */
const scalar = (a: Arg, b: Bindings): number => {
  if (typeof a === 'number') return a
  if (typeof a === 'object' && a !== null && 'kind' in a) {
    if (a.kind === 'size') return b.sizes?.[a.name] as number
    const c = b.constants?.[a.node]
    if (typeof c === 'number') return c
  }
  throw new DomainError('lda', 'lda: α, β, K and V must be numbers')
}

/**
 * The problem of an LDA-shaped model under its bindings: the words are the observed node's data, and $\alpha$,
 * $\beta$, $K$ and $V$ are read from the bindings' constants and sizes (or the model's literals). Throws `DomainError`
 * when the model is not LDA-shaped or one of the four is not a number.
 *
 * @param m The model; see `matchLda`.
 * @param b The bindings: `data` holds the observed node's words, one array (or tensor) of word ids per document; with
 *   no data the corpus is empty.
 * @returns The options for `ldaCollapsedGibbs` (no `allowed`).
 *
 * @example The LDA model bound to a two-document corpus
 * const bindings = { sizes: { K: 2, V: 4 }, constants: { 'α': 0.5, 'β': 0.1 }, data: { w: [[0, 1, 1], [2, 3]] } }
 * print(ldaOptions(ldaModel(), bindings))
 */
export function ldaOptions(m: Model, b: Bindings): LdaOptions {
  const shape = matchLda(m)
  if (!shape) throw new DomainError('lda', `lda: ${m.name} is not LDA-shaped`)
  const docs = b.data?.[shape.w] as Nested
  const documents = (Array.isArray(docs) ? docs : []).map((d) =>
    isTensor(d) ? Array.from(toFlat(d as Tensor)) : Array.from(d as number[]),
  )
  return {
    documents,
    topics: scalar(shape.K, b),
    vocabulary: scalar(shape.V, b),
    alpha: scalar(shape.alpha, b),
    beta: scalar(shape.beta, b),
  }
}

/**
 * The LDA engine: matches LDA-shaped models (`matchLda`) and runs collapsed Gibbs (`ldaCollapsedGibbs`) on the observed
 * words, with the problem `ldaOptions` reads from the bindings. Pass it to `infer` through an engine table, e.g.
 * `infer(m, b, { engines: ldaEngines })`.
 */
export const ldaEngine: EngineRegistration = {
  name: 'lda-collapsed-gibbs',
  matches: (c) => matchLda(c.model) !== null,
  create: (c) => ldaCollapsedGibbs(ldaOptions(c.model, c.bindings)),
}

/** The built-in engines with LDA's first, so `infer` runs collapsed Gibbs on LDA-shaped models. */
export const ldaEngines: EngineTable = withEngines(builtInEngines, ldaEngine)
