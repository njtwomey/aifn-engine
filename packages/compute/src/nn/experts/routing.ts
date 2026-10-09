/**
 * Routing for a mixture of experts: turn a router's logits `[T, N]` ($T$ tokens, $N$ experts) into combine weights
 * `[T, N]` that are differentiable in the logits, and a constant dispatch mask saying which expert sees which token.
 * Below, $\zvec_t$ is token $t$'s row of logits and $\tau$ the temperature.
 *
 * - `softmax`: dense gating, every expert sees every token with weight $\mathrm{softmax}(\zvec_t/\tau)$ (Jacobs,
 *   Jordan, Nowlan and Hinton, 1991, "Adaptive mixtures of local experts", Neural Computation 3(1)).
 * - `top-k`: each token keeps its $k$ largest logits and renormalises the softmax over them (Shazeer et al., 2017,
 *   "Outrageously large neural networks", ICLR, eq. 3–5; Mixtral's convention).
 * - `noisy-top-k`: the same on logits plus Gaussian noise of a per-token, per-expert scale (Shazeer et al., 2017,
 *   eq. 4), drawn only when a stream is given (training).
 * - `switch`: top-1 with the unrenormalised probability as the weight, so the router gets a gradient (Fedus, Zoph and
 *   Shazeer, 2022, "Switch Transformers", JMLR 23, §2.1).
 * - `expert-choice`: each expert takes its $C$ highest-probability tokens (Zhou et al., 2022, "Mixture-of-experts with
 *   expert choice routing", NeurIPS): balanced by construction, but a token may get no expert or several.
 *
 * Capacity (token-choice gates): each expert takes at most $C = \lceil c T k / N \rceil$ assignments ($c$ the
 * `capacityFactor`), filled in priority order: every token's first choice in token order, then every second choice,
 * and so on (Lepikhin et al., 2021, "GShard", ICLR, Algorithm 1). An assignment past capacity is dropped: its weight
 * becomes 0, and a token with none left passes through the layer as zero (a residual connection around the layer
 * carries it on).
 *
 * The selection is discrete and read from the primal values of the logits, so it is constant to every derivative
 * transform; gradients reach the router through the weights of the experts it chose.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  fromData,
  log,
  mul,
  shapeOfValue,
  toFlat,
  unwrap,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The gating rules of `route`, described in the file comment. */
export type GateKind = 'softmax' | 'top-k' | 'noisy-top-k' | 'switch' | 'expert-choice'

/** The gating rules, in order, for controls and registries. */
export const GATE_KINDS: readonly GateKind[] = ['softmax', 'top-k', 'noisy-top-k', 'switch', 'expert-choice']

/** Options of `route`. */
export type RoutingOptions = {
  /** The gating rule (default `top-k`). */
  gate?: GateKind
  /**
   * Experts per token for `top-k` and `noisy-top-k` (default 2, at most $N$); `switch` uses 1. For `expert-choice` it
   * is the default capacity factor, the average number of experts per token.
   */
  k?: Size
  /**
   * Temperature $\tau > 0$: the weights are a softmax of $\zvec/\tau$ (default 1). Small $\tau$ makes soft gating
   * nearly hard.
   */
  temperature?: number
  /**
   * Capacity factor $c > 0$ (default $\infty$: no dropping; `expert-choice` defaults to $k$). Ignored by `softmax`,
   * whose experts see every token.
   */
  capacityFactor?: number
  /**
   * Renormalise the softmax over each token's chosen experts (default true for `top-k` and `noisy-top-k`, false for
   * `switch` and `expert-choice`, which weight by the full softmax's probability).
   */
  normalise?: boolean
  /**
   * `noisy-top-k`: the noise scale, a number or `[T, N]` (e.g. $\mathrm{softplus}(\xvec\Wmat_{\mathrm{noise}})$);
   * default 1.
   */
  noiseScale?: Value
  /** `noisy-top-k`: the stream the noise is drawn from; without one (evaluation) no noise is added. */
  stream?: Stream
}

/** The result of `route`. */
export type Routing = {
  /** The gating rule used. */
  readonly gate: GateKind
  /** The number of tokens $T$. */
  readonly tokens: Size
  /** The number of experts $N$. */
  readonly experts: Size
  /**
   * Experts chosen per token (token-choice gates; $N$ for `softmax`; for `expert-choice` the option `k`, the default
   * capacity factor).
   */
  readonly k: number
  /** The router's logits `[T, N]`, before temperature and noise. */
  readonly logits: Value
  /** The logits the selection used: $\zvec/\tau$ plus noise for `noisy-top-k`, $\zvec/\tau$ otherwise. */
  readonly scores: Value
  /** The softmax of `scores` over every expert, `[T, N]`: the router's probabilities (used by the auxiliary losses). */
  readonly probs: Value
  /** The combine weights `[T, N]`: zero where an expert was not chosen or its assignment was dropped. */
  readonly combine: Value
  /** 1 where the gate chose expert $i$ for token $t$, before capacity, `[T, N]` (constant). */
  readonly selected: Tensor
  /** 1 where expert $i$ processes token $t$, after capacity, `[T, N]` (constant). */
  readonly dispatch: Tensor
  /** Assignments each expert may take ($\infty$ without a capacity). */
  readonly capacity: number
}

/** The logit given to unchosen experts when renormalising: finite, so no infinity enters a derivative. */
const NEG = -1e30

/**
 * The $k$ largest entries of a row, by index, largest first (ties by lower index).
 *
 * @param row The values to rank.
 * @param k How many indices to return (all of them when $k$ exceeds the length).
 * @returns The indices of the $k$ largest values.
 */
function topIndices(row: ArrayLike<number>, k: Size): number[] {
  const idx = Array.from({ length: row.length }, (_, i) => i)
  idx.sort((a, b) => row[b] - row[a] || a - b)
  return idx.slice(0, k)
}

/**
 * Rows of a `[T, N]` value's primal values, as copies.
 *
 * @param v The `[T, N]` value; a traced one is read through its primal.
 * @param T The number of rows.
 * @param N The number of columns.
 * @returns $T$ arrays of $N$ values.
 */
function rowsOf(v: Value, T: Size, N: Size): Float64Array[] {
  const flat = toFlat(unwrap(v) as Tensor)
  return Array.from({ length: T }, (_, t) => Float64Array.from(flat.slice(t * N, (t + 1) * N)))
}

/**
 * An expert's capacity $\lceil c T k / N \rceil$, the assignments it may take ($\infty$ when $c$ is). Throws
 * `DomainError` unless $c > 0$.
 *
 * @param tokens The number of tokens $T$.
 * @param experts The number of experts $N$.
 * @param k The assignments per token $k$, so $Tk$ in all.
 * @param capacityFactor The capacity factor $c$: 1 gives each expert exactly its even share.
 * @returns The capacity, or $\infty$ without a factor.
 *
 * @example Eight tokens, four experts, two choices each: an even share is four
 * print('no limit:', expertCapacity(8, 4, 2))
 * print('factor 1:', expertCapacity(8, 4, 2, 1))
 * print('factor 1.25:', expertCapacity(8, 4, 2, 1.25))
 */
export function expertCapacity(tokens: Size, experts: Size, k: number, capacityFactor = Infinity): number {
  if (!(capacityFactor > 0))
    throw new DomainError('expertCapacity', `expertCapacity: the capacity factor must be positive`)
  return Number.isFinite(capacityFactor) ? Math.ceil((capacityFactor * tokens * k) / experts) : Infinity
}

/**
 * Route $T$ tokens to $N$ experts from the router's logits `[T, N]`: the combine weights (differentiable in the logits
 * and the noise scale), the router's probabilities, and the constant selection and dispatch masks (see the file
 * comment for each gate and for capacity). Throws `ShapeError` unless the logits are a matrix, and `DomainError` for a
 * temperature that is not positive or a $k$ below 1.
 *
 * @param logits The router's logits `[T, N]`, one row per token; a traced value makes the weights differentiable.
 * @param options The gate, $k$, temperature, capacity factor, renormalisation and, for `noisy-top-k`, the noise.
 * @returns The routing: weights, probabilities, masks and capacity.
 *
 * @example Top-2 of three experts: each row of weights sums to one over its two experts
 * const logits = tensor([[2, 1, 0], [0, 3, 1], [1, 1, 4]])
 * const routing = route(logits, { gate: 'top-k', k: 2 })
 * print('selected:', routing.selected)
 * print('combine:', routing.combine)
 * print('row sums:', sum(routing.combine, 1))
 *
 * @example Dense softmax against Switch's top-1, which keeps the unrenormalised probability
 * const logits = tensor([[2, 1, 0], [0, 3, 1], [1, 1, 4]])
 * print('softmax:', route(logits, { gate: 'softmax' }).combine)
 * print('switch:', route(logits, { gate: 'switch' }).combine)
 *
 * @example Capacity: four tokens all choose expert 0, which takes two, in token order
 * const routing = route(tensor([[3, 0], [2, 0], [1, 0], [4, 0]]), { gate: 'switch', capacityFactor: 1 })
 * print('capacity:', routing.capacity)
 * print('selected:', routing.selected)
 * print('dispatch:', routing.dispatch)
 *
 * @example Expert choice: each expert takes its two likeliest tokens, so the last token gets all three
 * const logits = tensor([[2, 1, 0], [0, 3, 1], [1, 1, 4], [2, 2, 2]])
 * const routing = route(logits, { gate: 'expert-choice', k: 1 })
 * print('capacity per expert:', routing.capacity)
 * print('dispatch:', routing.dispatch)
 *
 * @example Noisy top-k adds noise only when given a stream
 * const logits = tensor([[2, 1, 0], [0, 3, 1]])
 * print('evaluation:', route(logits, { gate: 'noisy-top-k', k: 1 }).scores)
 * print('training:', route(logits, { gate: 'noisy-top-k', k: 1, stream: stream(0) }).scores)
 */
export function route(logits: Value, options: RoutingOptions = {}): Routing {
  const { gate = 'top-k', temperature = 1 } = options
  const shape = shapeOfValue(logits)
  if (shape.length !== 2)
    throw new ShapeError('route', `route: logits must be [tokens, experts], got [${shape.join(', ')}]`)
  const [T, N] = shape
  if (!(temperature > 0)) throw new DomainError('route', 'route: the temperature must be positive')
  const k = gate === 'switch' ? 1 : gate === 'softmax' ? N : Math.min(options.k ?? 2, N)
  if (!(k >= 1)) throw new DomainError('route', 'route: k must be at least 1')

  let scores: Value = temperature === 1 ? logits : mul(logits, 1 / temperature)
  if (gate === 'noisy-top-k' && options.stream) {
    const eps = normal(options.stream, 0, 1, { shape: [T, N] }) as Tensor
    scores = add(scores, mul(eps, options.noiseScale ?? 1))
  }
  const probs = softmax(scores)
  const ones = fromData(new Float64Array(T * N).fill(1), [T, N])

  if (gate === 'softmax')
    return {
      gate,
      tokens: T,
      experts: N,
      k: N,
      logits,
      scores,
      probs,
      combine: probs,
      selected: ones,
      dispatch: ones,
      capacity: Infinity,
    }

  const rows = rowsOf(scores, T, N)
  const selected = new Float64Array(T * N)
  const dispatch = new Float64Array(T * N)
  let capacity: number

  if (gate === 'expert-choice') {
    // Each expert ranks the tokens by its probability and takes the top C.
    const factor = options.capacityFactor ?? k
    capacity = Math.min(T, expertCapacity(T, N, 1, factor))
    const p = rowsOf(probs, T, N)
    for (let i = 0; i < N; i++) {
      const column = p.map((r) => r[i])
      for (const t of topIndices(column, capacity)) selected[t * N + i] = dispatch[t * N + i] = 1
    }
  } else {
    capacity = expertCapacity(T, N, k, options.capacityFactor)
    const choices = rows.map((r) => topIndices(r, k))
    for (let t = 0; t < T; t++) for (const i of choices[t]) selected[t * N + i] = 1
    // Fill each expert's buffer by priority: all first choices in token order, then all second choices, …
    const load = new Array<number>(N).fill(0)
    for (let r = 0; r < k; r++)
      for (let t = 0; t < T; t++) {
        const i = choices[t][r]
        if (load[i] < capacity) {
          load[i]++
          dispatch[t * N + i] = 1
        }
      }
  }

  const selectedT = fromData(selected, [T, N])
  const dispatchT = fromData(dispatch, [T, N])
  const normalise = options.normalise ?? (gate === 'top-k' || gate === 'noisy-top-k')
  // Renormalising: a softmax over the chosen experts only (the others pushed to a large negative logit, so their
  // weight is exactly 0 and no infinity enters a derivative).
  const weights = normalise ? softmax(where(selectedT, scores, NEG)) : probs
  const combine = mul(weights, dispatchT)
  return {
    gate,
    tokens: T,
    experts: N,
    k,
    logits,
    scores,
    probs,
    combine,
    selected: selectedT,
    dispatch: dispatchT,
    capacity,
  }
}

/**
 * A dense routing from given probabilities `[T, N]` (each row summing to 1), for gates that are not a single softmax of
 * logits, such as the product of softmaxes down a hierarchical mixture of experts (Jordan and Jacobs, 1994). Every
 * expert sees every token; `logits` and `scores` are log-probabilities. Throws `ShapeError` unless the probabilities
 * are a matrix.
 *
 * @param probs The gate probabilities `[T, N]`, used as the combine weights.
 * @param logProbs Their logarithms, when the caller has them more accurately (default: the log of `probs`).
 * @returns A `softmax` routing with these weights and no capacity.
 *
 * @example Two tokens' given gate probabilities
 * const routing = denseRouting(tensor([[0.5, 0.5], [0.9, 0.1]]))
 * print('combine:', routing.combine)
 * print('logits:', routing.logits)
 */
export function denseRouting(probs: Value, logProbs?: Value): Routing {
  const shape = shapeOfValue(probs)
  if (shape.length !== 2) throw new ShapeError('denseRouting', `denseRouting: probabilities must be [tokens, experts]`)
  const [T, N] = shape
  const ones = fromData(new Float64Array(T * N).fill(1), [T, N])
  const lp = logProbs ?? log(probs)
  return {
    gate: 'softmax',
    tokens: T,
    experts: N,
    k: N,
    logits: lp,
    scores: lp,
    probs,
    combine: probs,
    selected: ones,
    dispatch: ones,
    capacity: Infinity,
  }
}
