/**
 * Probabilistic latent semantic analysis (Hofmann, 1999) fitted by expectation–maximisation: each document d is a
 * mixture P(w | d) = Σ_z P(z | d) P(w | z) of K topics. The E-step gives each (document, word) pair its topic
 * responsibilities P(z | d, w) ∝ P(z | d) P(w | z); the M-step re-estimates both distributions from the expected
 * counts n(d, w) P(z | d, w). The log-likelihood Σ n(d, w) log P(w | d) never decreases.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { child, units } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { documentTermCounts, type Documents } from './corpus'

/** The state of `plsaSteps`. */
export interface PlsaState extends Status {
  t: Size
  /** P(w | z) [K, V] and P(z | d) [D, K]. */
  topicWord: Tensor
  docTopic: Tensor
  logLikelihood: number
}

/** Options of `plsaSteps`. */
export type PlsaOptions = {
  documents: Documents
  topics: Size
  vocabulary: Size
  /** Initial P(w | z) [K, V] and P(z | d) [D, K], row-major (default random from the init stream). */
  init?: { topicWord: ArrayLike<number>; docTopic: ArrayLike<number> }
}

function logLikelihoodOf(n: ArrayLike<number>, theta: Float64Array, phi: Float64Array, D: Size, K: Size, V: Size) {
  let ll = 0
  for (let d = 0; d < D; d++)
    for (let w = 0; w < V; w++) {
      const c = n[d * V + w]
      if (c === 0) continue
      let p = 0
      for (let k = 0; k < K; k++) p += theta[d * K + k] * phi[k * V + w]
      ll += c * Math.log(p)
    }
  return ll
}

/** pLSA by EM as a step-through algorithm; `init` draws random distributions from its stream. */
export function plsaSteps(options: PlsaOptions): Algorithm<void, PlsaState> {
  const { topics: K, vocabulary: V, documents } = options
  const D = documents.length
  const n = toFlat(documentTermCounts(documents, V))
  const normalise = (a: Float64Array, rows: Size, cols: Size) => {
    for (let r = 0; r < rows; r++) {
      let s = 0
      for (let c = 0; c < cols; c++) s += a[r * cols + c]
      for (let c = 0; c < cols; c++) a[r * cols + c] /= s
    }
    return a
  }
  return {
    name: 'plsa-em',
    init: (_start, s) => {
      const phi = options.init
        ? Float64Array.from(options.init.topicWord)
        : normalise(
            Float64Array.from(units(child(s, 'topics'), K * V), (u) => 0.5 + u),
            K,
            V,
          )
      const theta = options.init
        ? Float64Array.from(options.init.docTopic)
        : normalise(
            Float64Array.from(units(child(s, 'documents'), D * K), (u) => 0.5 + u),
            D,
            K,
          )
      return {
        t: 0,
        topicWord: fromData(phi, [K, V]),
        docTopic: fromData(theta, [D, K]),
        logLikelihood: logLikelihoodOf(n, theta, phi, D, K, V),
      }
    },
    step: (state) => {
      const theta = Float64Array.from(state.docTopic.data)
      const phi = Float64Array.from(state.topicWord.data)
      const newPhi = new Float64Array(K * V)
      const newTheta = new Float64Array(D * K)
      const r = new Float64Array(K)
      for (let d = 0; d < D; d++)
        for (let w = 0; w < V; w++) {
          const c = n[d * V + w]
          if (c === 0) continue
          let z = 0
          for (let k = 0; k < K; k++) z += r[k] = theta[d * K + k] * phi[k * V + w]
          for (let k = 0; k < K; k++) {
            const e = (c * r[k]) / z
            newPhi[k * V + w] += e
            newTheta[d * K + k] += e
          }
        }
      normalise(newPhi, K, V)
      normalise(newTheta, D, K)
      const ll = logLikelihoodOf(n, newTheta, newPhi, D, K, V)
      return {
        t: state.t + 1,
        topicWord: fromData(newPhi, [K, V]),
        docTopic: fromData(newTheta, [D, K]),
        logLikelihood: ll,
        converged: Math.abs(ll - state.logLikelihood) < 1e-10 * Math.abs(ll),
      }
    },
  }
}
