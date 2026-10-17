/**
 * State space duality (SSD, Mamba-2). A selective state space with scalar transitions,
 * $\Hmat_t = a_t \Hmat_{t-1} + \bvec_t \xvec_t^\top$ (state $N \times P$) and $\yvec_t = \Hmat_t^\top \cvec_t$, has the
 * closed form $\Ymat = (\Lmat \circ \Cmat\Bmat^\top)\Xmat$, where $\Lmat$ is the 1-semiseparable mask
 * $L_{ji} = a_j a_{j-1} \cdots a_{i+1}$ for $j \ge i$ and 0 above the diagonal. This file builds the mask (by a masked
 * segment sum of $\log a$), and computes the output three ways: by the quadratic (attention-like) form, by the linear
 * recurrence, and by the chunked algorithm of Mamba-2 (within each chunk of length $Q$ the quadratic form, then each
 * chunk's final state, a recurrence over the chunk states, and the carried state's contribution to each chunk's
 * outputs).
 *
 * Source: Dao and Gu (2024), "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
 * State Space Duality", §3 (Definition 3.1, eq. (6)), §5 (SSD as masked attention), §6 (the block decomposition in
 * four steps) and Theorem 6.1 (costs). One head, time first ($\Xmat$ is $T \times P$, $\Bmat$ and $\Cmat$ are
 * $T \times N$); `vmap` batches heads. Built from differentiable primitives. Decays may be 0 (Mamba's
 * $\Delta \to \infty$): the mask sums $\log a$ over each segment, so a zero gives an exact 0 rather than the NaN of
 * $\exp(-\infty + \infty)$; a negative decay throws `DomainError`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import {
  add,
  arange,
  associativeScan,
  broadcastTo,
  concat,
  cumsum,
  einsum,
  exp,
  expandDims,
  greater,
  greaterEqual,
  isTensor,
  isTraced,
  log,
  matmul,
  mul,
  shapeOfValue,
  slice,
  tensor,
  toFlat,
  transpose,
  where,
  zeros,
  type MatrixLike,
  type Value,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import { linearRecurrence } from './ssm'

/**
 * A value from a tensor or plain (nested) array argument, so callers may pass rows of numbers.
 *
 * @param x A value, an array of numbers or an array of rows.
 * @returns `x` itself when it is a value, else the tensor of the array.
 */
function asValue(x: Value | VectorLike | MatrixLike): Value {
  return Array.isArray(x) ? tensor(x as number[]) : (x as Value)
}

/**
 * The decays as a value, checked to be non-negative where they are concrete (a traced value is not checked).
 *
 * @param a The decays $a_1, \dots, a_T$.
 * @param where The caller's name, for error messages.
 * @returns The decays as a value.
 */
function decays(a: Value | VectorLike, where: string): Value {
  const v = asValue(a)
  if (!isTraced(v)) {
    const bad = (isTensor(v) ? toFlat(v) : [v as number]).find((x) => !(x >= 0))
    if (bad !== undefined)
      throw new DomainError(where, `${where}: the decays a_t must be non-negative numbers (got ${bad})`)
  }
  return v
}

/**
 * The 1-semiseparable mask of the decays $a_1, \dots, a_T$: $L_{ji} = \prod_{k=i+1}^{j} a_k$ for $j \ge i$ (1 on
 * the diagonal) and 0 above it. Computed as $\exp$ of the segment sum $\sum_{k=i+1}^{j} \log a_k$ (Mamba-2's
 * `segsum`: a cumulative sum of $\log a_j$ masked to $i < j$), so a zero decay cuts the mask to exact zeros.
 * Differentiable in $a$ where $a > 0$; throws `DomainError` for a negative decay.
 *
 * @param a The decays $a_t$, $T$ values in $[0, 1]$ (an array or a vector).
 * @returns The mask $\Lmat$, $T \times T$.
 *
 * @example Six decays; the zero cuts every path through it
 * print('L =', semiseparableMask([0.9, 0.5, 1, 0.8, 0.2, 0.9]))
 * print('with a zero:', semiseparableMask([0.9, 0, 0.5]))
 */
export function semiseparableMask(a: Value | VectorLike): Value {
  const av = decays(a, 'semiseparableMask')
  const T = shapeOfValue(av)[0]
  const steps = arange(T)
  const j = expandDims(steps, -1)
  const i = expandDims(steps, 0)
  // x[j, i] = log a_j below the diagonal (i < j), else 0; summed down each column it is Σ_{k=i+1}^{j} log a_k.
  const x = where(greater(j, i), broadcastTo(expandDims(log(av), -1), [T, T]), 0)
  return exp(where(greaterEqual(j, i), cumsum(x, 0), -Infinity))
}

/**
 * The SSD output by its quadratic (masked attention) form, $\Ymat = (\Lmat \circ \Cmat\Bmat^\top)\Xmat$.
 * $O(T^2(N + P))$. Differentiable.
 *
 * @param x The inputs $\Xmat$, $T \times P$ (a tensor or rows of numbers).
 * @param a The decays $a_t$, $T$ values in $[0, 1]$.
 * @param B The input maps $\bvec_t$, $T \times N$.
 * @param C The output maps $\cvec_t$, $T \times N$.
 * @returns The outputs $\Ymat$, $T \times P$.
 *
 * @example One state and one channel: y_t sums the inputs, decayed
 * const a = [0.9, 0.5, 1, 0.8, 0.2, 0.9]
 * const ones = a.map(() => [1])
 * print('y =', ssdQuadratic([[1], [0], [2], [0], [0], [1]], a, ones, ones))
 */
export function ssdQuadratic(
  x: Value | MatrixLike,
  a: Value | VectorLike,
  B: Value | MatrixLike,
  C: Value | MatrixLike,
): Value {
  const L = semiseparableMask(a)
  const scores = matmul(asValue(C), transpose(asValue(B)))
  return matmul(mul(L, scores), asValue(x))
}

/**
 * The SSD output by its recurrence, $\Hmat_t = a_t \Hmat_{t-1} + \bvec_t \xvec_t^\top$ ($N \times P$) and
 * $\yvec_t = \Hmat_t^\top \cvec_t$, from $\Hmat_0$ (zeros by default), every state at once by `linearRecurrence`.
 * $O(TNP)$. Differentiable.
 *
 * @param x The inputs $\Xmat$, $T \times P$.
 * @param a The decays $a_t$, $T$ values in $[0, 1]$.
 * @param B The input maps $\bvec_t$, $T \times N$.
 * @param C The output maps $\cvec_t$, $T \times N$.
 * @param initial The state $\Hmat_0$, $N \times P$ (default zeros).
 * @returns The outputs `y` ($T \times P$) and the `final` state $\Hmat_T$ ($N \times P$).
 *
 * @example The recurrence gives the quadratic form's outputs
 * const a = [0.9, 0.5, 1, 0.8, 0.2, 0.9]
 * const ones = a.map(() => [1])
 * const x = [[1], [0], [2], [0], [0], [1]]
 * print('recurrent:', ssdRecurrent(x, a, ones, ones).y)
 * print('quadratic:', ssdQuadratic(x, a, ones, ones))
 */
export function ssdRecurrent(
  x: Value | MatrixLike,
  a: Value | VectorLike,
  B: Value | MatrixLike,
  C: Value | MatrixLike,
  initial?: Value | MatrixLike,
): { y: Value; final: Value } {
  const xv = asValue(x)
  const av = decays(a, 'ssdRecurrent')
  const T = shapeOfValue(xv)[0]
  const writes = einsum('tn,tp->tnp', asValue(B), xv)
  const h = linearRecurrence(
    expandDims(expandDims(av, -1), -1),
    writes,
    initial === undefined ? {} : { initial: asValue(initial) },
  )
  return { y: einsum('tnp,tn->tp', h, asValue(C)), final: slice(h, T - 1) }
}

/** Multiply counts of `ssdChunked`, for illustration (dense blocks, triangular halves counted once). */
export type SsdCost = {
  /** Multiplies in the diagonal blocks (the quadratic form inside each chunk): matrix products. */
  intraChunk: number
  /** Multiplies forming each chunk's final state and reading carried states out: matrix products. */
  stateProducts: number
  /** Sequential steps of the recurrence between chunks, $\lceil T/Q \rceil$. */
  sequentialSteps: number
}

/**
 * The chunked SSD algorithm with chunk length $Q$ (Dao and Gu, 2024, §6): (1) the quadratic form inside each chunk
 * from a zero state; (2) each chunk's final state from a zero state; (3) a recurrence over the $\lceil T/Q \rceil$
 * chunk states with each chunk's total decay; (4) the carried state's contribution
 * $(\prod_{k \le t} a_k)\, \Hmat^\top \cvec_t$ to each output. Equal to `ssdQuadratic` for every $Q$.
 * Differentiable.
 *
 * @param x The inputs $\Xmat$, $T \times P$.
 * @param a The decays $a_t$, $T$ values in $[0, 1]$.
 * @param B The input maps $\bvec_t$, $T \times N$.
 * @param C The output maps $\cvec_t$, $T \times N$.
 * @param Q The chunk length (the last chunk may be shorter).
 * @returns The outputs `y` ($T \times P$), the `final` state ($N \times P$) and the multiply counts `cost`.
 *
 * @example Chunks of three give the quadratic form's outputs, with two sequential steps
 * const a = [0.9, 0.5, 1, 0.8, 0.2, 0.9]
 * const ones = a.map(() => [1])
 * const { y, cost } = ssdChunked([[1], [0], [2], [0], [0], [1]], a, ones, ones, 3)
 * print('y =', y)
 * print('cost =', cost)
 */
export function ssdChunked(
  x: Value | MatrixLike,
  a: Value | VectorLike,
  B: Value | MatrixLike,
  C: Value | MatrixLike,
  Q: number,
): { y: Value; final: Value; cost: SsdCost } {
  if (!(Number.isInteger(Q) && Q >= 1))
    throw new DomainError('ssdChunked', `ssdChunked: the chunk length must be a positive integer (got ${Q})`)
  const xv = asValue(x)
  const av = decays(a, 'ssdChunked')
  const Bv = asValue(B)
  const Cv = asValue(C)
  const [T, P] = shapeOfValue(xv)
  const N = shapeOfValue(Bv)[1]
  const ys: Value[] = []
  let carried: Value = zeros([N, P])
  let intraChunk = 0
  let stateProducts = 0
  let sequentialSteps = 0
  for (let start = 0; start < T; start += Q) {
    const end = Math.min(T, start + Q)
    const q = end - start
    const [xs, as, Bs, Cs] = [xv, av, Bv, Cv].map((v) => slice(v, [start, end]))
    // (1) Diagonal block: the quadratic form within the chunk.
    const L = semiseparableMask(as)
    const inner = matmul(mul(L, matmul(Cs, transpose(Bs))), xs)
    intraChunk += ((q * (q + 1)) / 2) * (N + P)
    // (4) Carried state read out: y_t += (a_start ⋯ a_t) C_t carried, the decay from the chunk start.
    const decay = associativeScan(mul, as)
    ys.push(add(inner, mul(expandDims(decay, -1), matmul(Cs, carried))))
    stateProducts += q * N * P
    // (2) The chunk's state from a zero start: Σ_k (a_{k+1} ⋯ a_q) b_k x_kᵀ, the last row of the mask.
    const local = matmul(transpose(mul(expandDims(slice(L, q - 1), -1), Bs)), xs)
    stateProducts += q * N * P
    // (3) The recurrence over chunk states: carried ← (Π a) carried + local.
    carried = add(mul(slice(decay, q - 1), carried), local)
    sequentialSteps += 1
  }
  return { y: concat(ys, 0), final: carried, cost: { intraChunk, stateProducts, sequentialSteps } }
}
