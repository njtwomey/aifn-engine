/**
 * Latent Dirichlet allocation (Blei, Ng and Jordan, 2003, JMLR 3), part of `aifn-methods/inference/topic-models`: the
 * model in the model language (a structured graph with the topic, document and word plates as groups), its match,
 * collapsed Gibbs sampling (Griffiths and Steyvers, 2004, PNAS 101), and the engine to pass to
 * `aifn-compute/inference/engines`' `infer`.
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

/** The plan's LDA description: topics φ_k ~ Dir(β), mixtures θ_d ~ Dir(α), z_dn ~ Cat(θ_d), w_dn ~ Cat(φ_{z_dn}). */
export function ldaModel(): Model {
  return ldaDescription
}

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

/** The names of an LDA-shaped model's nodes, or null when the model is not LDA-shaped. */
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
  /** Documents as arrays of word ids in 0 … V − 1. */
  documents: readonly (readonly number[])[]
  topics: number
  vocabulary: number
  /** Symmetric Dirichlet concentrations on θ (α) and φ (β). */
  alpha: number
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
  /** z_dn, per document (int32 vectors). */
  assignments: Tensor[]
  /** n_dk (D × K), n_kw (K × V) and n_k (K). */
  docTopic: Matrix
  topicWord: Matrix
  topicTotals: Tensor
  /** log p(w | z) with φ integrated out. */
  logLikelihood: number
}

/**
 * log p(w | z) with the topics φ_k ~ Dir(β) integrated out, from the topic–word counts n_kw (K × V, row-major) and
 * their totals n_k: Σ_k [log Γ(Vβ) − V log Γ(β) − log Γ(n_k + Vβ) + Σ_w log Γ(n_kw + β)]. Shared with the HDP sampler.
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
 * Collapsed Gibbs sampling for LDA as a traceable algorithm (Griffiths and Steyvers, 2004): θ and φ are integrated
 * out, and each step (a sweep) resamples every token's topic from p(z = k | rest) ∝ (n_dk + α)(n_kw + β)/(n_k + Vβ),
 * with the token's own counts removed. The initial topics come from the `init` stream; token (d, n) in a sweep draws
 * from `child(ctx.stream, d, n)`. No start. With `allowed`, each document's tokens are drawn only from its own topics:
 * labelled LDA, where the labels of a document are the topics it may use.
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

/** Point estimates φ_kw = (n_kw + β)/(n_k + Vβ) (K × V) and θ_dk = (n_dk + α)/(N_d + Kα) (D × K) from a state. */
export function ldaEstimates(s: LdaState, options: LdaOptions): { topicWord: Matrix; docTopic: Matrix } {
  const { topics: K, vocabulary: V, alpha, beta } = options
  const phi = s.topicWord.data.map((n, i) => (n + beta) / (s.topicTotals.data[Math.floor(i / V)] + V * beta))
  const theta = s.docTopic.data.map((n, i) => {
    const d = Math.floor(i / K)
    return (n + alpha) / (options.documents[d].length + K * alpha)
  })
  return { topicWord: fromData(phi, [K, V]), docTopic: fromData(theta, s.docTopic.shape) }
}

const scalar = (a: Arg, b: Bindings): number => {
  if (typeof a === 'number') return a
  if (typeof a === 'object' && a !== null && 'kind' in a) {
    if (a.kind === 'size') return b.sizes?.[a.name] as number
    const c = b.constants?.[a.node]
    if (typeof c === 'number') return c
  }
  throw new DomainError('lda', 'lda: α, β, K and V must be numbers')
}

/** The problem of an LDA-shaped model under its bindings (the words are the observed node's data). */
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
 * The LDA engine: matches LDA-shaped models and runs collapsed Gibbs on the observed words. Pass it to `infer` through
 * an engine table, e.g. `infer(m, b, { engines: ldaEngines })`.
 */
export const ldaEngine: EngineRegistration = {
  name: 'lda-collapsed-gibbs',
  matches: (c) => matchLda(c.model) !== null,
  create: (c) => ldaCollapsedGibbs(ldaOptions(c.model, c.bindings)),
}

/** The built-in engines with LDA's first, so `infer` runs collapsed Gibbs on LDA-shaped models. */
export const ldaEngines: EngineTable = withEngines(builtInEngines, ldaEngine)
