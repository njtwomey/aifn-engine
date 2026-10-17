/**
 * Linear attention (Katharopoulos et al., 2020): replace the softmax kernel $\exp(\qvec \cdot \kvec)$ by a feature
 * map $\phi$ with $\mathrm{sim}(\qvec, \kvec) = \phi(\qvec) \cdot \phi(\kvec)$. Then the causal output
 * $\ovec_t = \phi(\qvec_t)^\top \Smat_t / \phi(\qvec_t)^\top \zvec_t$, with
 * $\Smat_t = \sum_{s \le t} \phi(\kvec_s) \vvec_s^\top$ and $\zvec_t = \sum_{s \le t} \phi(\kvec_s)$, is a recurrent
 * network whose state is the $d_k \times d_v$ matrix $\Smat_t$: constant memory per token, as an RNN, yet trainable
 * in parallel as attention. A decay $\gamma$,
 * $\Smat_t = \gamma \Smat_{t-1} + \phi(\kvec_t) \vvec_t^\top$, gives retention (Sun et al., 2023), whose parallel
 * form weights score $(t, s)$ by $\gamma^{t-s}$.
 *
 * The Performer's positive random features (Choromanski et al., 2021, FAVOR+),
 * $\phi(\xvec) = m^{-1/2} \exp(\Omegamat\xvec - \lVert \xvec \rVert^2/2)$, give a feature map whose inner products
 * estimate the softmax kernel without bias, $\expect[\phi(\qvec)^\top\phi(\kvec)] = \exp(\qvec^\top\kvec)$, and
 * `linearAttentionWeights` returns the $T \times T$ weight matrix itself, to draw beside softmax attention's.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  div,
  exp,
  expandDims,
  fromData,
  matmul,
  mul,
  shapeOfValue,
  square,
  squeeze,
  sub,
  sum,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { elu } from 'aifn-compute/nn/functional'
import { linearRecurrence } from './ssm'

/**
 * The feature map of Katharopoulos et al. (2020), $\phi(x) = \mathrm{elu}(x) + 1$, elementwise: positive everywhere,
 * $e^x$ below zero and $x + 1$ above. Differentiable.
 *
 * @param x The queries or keys, any shape.
 * @returns $\phi(x)$, with the shape of `x`.
 *
 * @example Positive everywhere
 * print('phi =', eluFeatureMap(tensor([-2, 0, 1])))
 */
export function eluFeatureMap(x: Value): Value {
  return add(elu(x), 1)
}

/** Options of `linearAttention` and `linearAttentionRecurrent`. */
export type LinearAttentionOptions = {
  /** The feature map $\phi$, applied to queries and keys (default `eluFeatureMap`). */
  featureMap?: (x: Value) => Value
  /** Causal: each query sees the keys up to its own position (default true). */
  causal?: boolean
  /** A decay $\gamma \in (0, 1]$ per step on older keys (retention; default 1, none). */
  decay?: number
  /** Divide by $\phi(\qvec_t)^\top \zvec_t$, the sum of the weights (default true; retention leaves it out). */
  normalise?: boolean
}

/**
 * The $T \times T$ weights on the scores: $\gamma^{t-s}$ for $s \le t$ and 0 above when causal, or
 * $\gamma^{\lvert t-s \rvert}$ everywhere when not.
 *
 * @param T The sequence length $T$.
 * @param decay The decay $\gamma$ (1 for none).
 * @param causal Whether the entries above the diagonal (keys after the query) are zeroed.
 * @returns The mask, $T \times T$, row $t$ for the query and column $s$ for the key.
 */
function decayMask(T: Size, decay: number, causal: boolean): Tensor {
  const out = new Float64Array(T * T)
  for (let t = 0; t < T; t++)
    for (let s = 0; s < T; s++) out[t * T + s] = !causal || s <= t ? decay ** Math.abs(t - s) : 0
  return fromData(out, [T, T])
}

/**
 * Linear attention in its parallel (attention) form: the weights $\phi(\Qmat)\phi(\Kmat)^\top$, masked causally and
 * decayed by $\gamma^{t-s}$, normalised by their row sums, times $\Vmat$. $O(T^2)$ like softmax attention, but equal
 * (when causal) to `linearAttentionRecurrent`, the $O(T)$ form. Not causal, the weights are
 * $\gamma^{\lvert t-s \rvert}$ in both directions. Differentiable.
 *
 * @param q The queries $\Qmat$, $[\dots, T, d_k]$.
 * @param k The keys $\Kmat$, $[\dots, T, d_k]$.
 * @param v The values $\Vmat$, $[\dots, T, d_v]$.
 * @param options The feature map, causality, decay and normalisation.
 * @returns The outputs, $[\dots, T, d_v]$.
 *
 * @example The parallel and recurrent forms agree
 * const q = tensor([[1, 0], [0, 1], [1, 1]])
 * const k = tensor([[1, 0], [0, 1], [1, 1]])
 * const v = tensor([[1], [2], [3]])
 * print('parallel:', linearAttention(q, k, v))
 * print('recurrent:', linearAttentionRecurrent(q, k, v).output)
 *
 * @example Retention: a decay of one half, unnormalised
 * const q = tensor([[1, 0], [0, 1], [1, 1]])
 * const v = tensor([[1], [2], [3]])
 * print('o =', linearAttention(q, q, v, { decay: 0.5, normalise: false }))
 */
export function linearAttention(q: Value, k: Value, v: Value, options: LinearAttentionOptions = {}): Value {
  const { featureMap = eluFeatureMap, causal = true, decay = 1, normalise = true } = options
  const scores = maskedScores(q, k, featureMap, causal, decay)
  const out = matmul(scores, v)
  return normalise ? div(out, sum(scores, -1, true)) : out
}

/**
 * The unnormalised scores $\phi(\Qmat)\phi(\Kmat)^\top$, masked causally and decayed by `decayMask`, shared by
 * `linearAttention` and `linearAttentionWeights`.
 *
 * @param q The queries, $[\dots, T, d_k]$.
 * @param k The keys, $[\dots, T, d_k]$.
 * @param featureMap The feature map $\phi$.
 * @param causal Whether keys after the query get weight 0.
 * @param decay The decay $\gamma$ (1 for none).
 * @returns The scores, $[\dots, T, T]$.
 */
function maskedScores(q: Value, k: Value, featureMap: (x: Value) => Value, causal: boolean, decay: number): Value {
  const fq = featureMap(q)
  const fk = featureMap(k)
  const r = shapeOfValue(fk).length
  const axes = Array.from({ length: r }, (_, i) => i)
  ;[axes[r - 2], axes[r - 1]] = [axes[r - 1], axes[r - 2]]
  const scores: Value = matmul(fq, transpose(fk, axes))
  const T = shapeOfValue(q).at(-2)!
  return causal || decay !== 1 ? mul(scores, decayMask(T, decay, causal)) : scores
}

/**
 * The weights of linear attention,
 * $W_{ij} = \phi(\qvec_i)^\top\phi(\kvec_j) / \sum_l \phi(\qvec_i)^\top\phi(\kvec_l)$ with the sums over the
 * keys query $i$ may see (and $\gamma^{i-j}$ on each score with a decay), so a figure can draw linear attention's
 * matrix beside softmax's; `linearAttention` is these weights times $\Vmat$. Differentiable.
 *
 * @param q The queries, $[\dots, T, d_k]$.
 * @param k The keys, $[\dots, T, d_k]$.
 * @param featureMap The feature map $\phi$ (non-negative outputs, such as `eluFeatureMap` or a
 *   `positiveRandomFeatures` map).
 * @param causal Hide keys after the query.
 * @param decay The decay $\gamma$ per step on older keys (1, the default, for none).
 * @returns The $T \times T$ weights (with the batch axes of `q`), rows summing to 1.
 *
 * @example Causal weights of three tokens under the ELU feature map
 * const x = tensor([[1, 0], [0, 1], [1, 1]])
 * print('W =', linearAttentionWeights(x, x, eluFeatureMap))
 */
export function linearAttentionWeights(
  q: Value,
  k: Value,
  featureMap: (x: Value) => Value,
  causal = true,
  decay = 1,
): Value {
  const scores = maskedScores(q, k, featureMap, causal, decay)
  return div(scores, sum(scores, -1, true))
}

/**
 * The Performer's positive random feature map for the softmax kernel $\exp(\qvec^\top\kvec)$ (Choromanski et
 * al., 2021, §2.3 and Lemma 1): $\phi(\xvec) = m^{-1/2}\exp(\Omegamat\xvec - \lVert\xvec\rVert^2/2)$ for an
 * $m \times d$ matrix $\Omegamat$ of independent standard normal rows, so that
 * $\expect[\phi(\qvec)^\top\phi(\kvec)] = \exp(\qvec^\top\kvec)$. The map fits `featureMap` of
 * `linearAttention` and `linearAttentionRecurrent`. Apply the $1/\sqrt{d}$ temperature by scaling queries and keys by
 * $d^{-1/4}$. (The paper's orthogonal features, Gram–Schmidt over blocks of $d$ rows, lower the variance further.)
 *
 * @param s The stream $\Omegamat$ is drawn from ($m \times d$ normals, in one block).
 * @param d The input width $d$.
 * @param m The number of features $m$.
 * @returns `featureMap`, mapping $[\dots, d]$ to $[\dots, m]$ (differentiable in its input), and the matrix `omega`.
 *
 * @example Many features estimate the softmax kernel
 * const { featureMap } = positiveRandomFeatures(stream(3), 4, 20000)
 * const q = tensor([[0.3, -0.2, 0.5, 0.1]])
 * const k = tensor([[0.4, 0.1, -0.3, 0.2]])
 * print('estimate =', sum(mul(featureMap(q), featureMap(k))))
 * print('exp(q.k) =', Math.exp(0.3 * 0.4 - 0.2 * 0.1 - 0.5 * 0.3 + 0.1 * 0.2))
 */
export function positiveRandomFeatures(
  s: Stream,
  d: Size,
  m: Size,
): { featureMap: (x: Value) => Value; omega: Tensor } {
  const omega = normals(s, [m, d])
  const featureMap = (x: Value): Value => {
    const projected = matmul(x, transpose(omega))
    const half = mul(0.5, sum(square(x), -1, true))
    return div(exp(sub(projected, half)), Math.sqrt(m))
  }
  return { featureMap, omega }
}

/**
 * The outputs of `linearAttentionRecurrent`: `output` $\ovec_t$ ($T \times d_v$), the `states` $\Smat_t$
 * ($T \times d_k \times d_v$) and the `normalisers` $\zvec_t$ ($T \times d_k$).
 */
export type LinearAttentionStates = { output: Value; states: Value; normalisers: Value }

/**
 * Causal linear attention as a recurrent network: $\Smat_t = \gamma \Smat_{t-1} + \phi(\kvec_t) \vvec_t^\top$ and
 * $\zvec_t = \gamma \zvec_{t-1} + \phi(\kvec_t)$ by `linearRecurrence` (an associative scan), then
 * $\ovec_t = \phi(\qvec_t)^\top \Smat_t / \phi(\qvec_t)^\top \zvec_t$. Every state is returned, so a figure can
 * show the memory the network carries. One sequence only (no batch axes). Differentiable.
 *
 * @param q The queries, $T \times d_k$.
 * @param k The keys, $T \times d_k$.
 * @param v The values, $T \times d_v$.
 * @param options The feature map, decay and normalisation (always causal).
 * @returns The outputs with every state $\Smat_t$ and normaliser $\zvec_t$.
 *
 * @example The memory of three tokens
 * const q = tensor([[1, 0], [0, 1], [1, 1]])
 * const v = tensor([[1], [2], [3]])
 * const { output, states, normalisers } = linearAttentionRecurrent(q, q, v)
 * print('o =', output)
 * print('S_3 =', slice(states, 2))
 * print('z =', normalisers)
 */
export function linearAttentionRecurrent(
  q: Value,
  k: Value,
  v: Value,
  options: Omit<LinearAttentionOptions, 'causal'> = {},
): LinearAttentionStates {
  const { featureMap = eluFeatureMap, decay = 1, normalise = true } = options
  const fq = featureMap(q)
  const fk = featureMap(k)
  const updates = mul(expandDims(fk, 2), expandDims(v, 1))
  const states = linearRecurrence(decay, updates)
  const normalisers = linearRecurrence(decay, fk)
  const num = squeeze(matmul(expandDims(fq, 1), states), 1)
  const output = normalise ? div(num, sum(mul(fq, normalisers), -1, true)) : num
  return { output, states, normalisers }
}
