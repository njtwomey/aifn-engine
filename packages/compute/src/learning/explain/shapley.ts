/**
 * Shapley values (Shapley, 1953) of a set function $v$ over $d$ players, and KernelSHAP (Lundberg and Lee, 2017), which
 * explains a model's output $f(\xvec)$ by the Shapley values of the interventional value function
 * $v(S) = \frac{1}{b}\sum_{\bvec} f(\xvec_S, \bvec_{\bar S})$ over the background rows $\bvec$ (features in $S$ from
 * $\xvec$, the rest from $\bvec$).
 *
 * The Shapley value of player $i$ averages its marginal contribution $v(S \cup \{i\}) - v(S)$ over the coalitions
 * $S \not\ni i$ with weight $\lvert S \rvert!\,(d - \lvert S \rvert - 1)!/d!$. KernelSHAP finds the same values as
 * the solution of a weighted least-squares problem: fit $v(S) - v(\emptyset) \approx \sum_{i \in S} \phi_i$ subject to
 * $\sum_i \phi_i = v(\{1, \dots, d\}) - v(\emptyset)$, weighting $S$ by the Shapley kernel
 * $\pi(S) = (d - 1)/\big(\binom{d}{\lvert S \rvert} \lvert S \rvert (d - \lvert S \rvert)\big)$. Over all
 * $2^d - 2$ proper coalitions the solution is exact; with sampled coalitions (sizes drawn in proportion to the kernel's
 * mass per size, each with its complement) it converges to it.
 *
 * For a linear model $f(\xvec) = \wvec^\top\xvec + c$ every method here gives
 * $\phi_i = w_i (x_i - \bar b_i)$, with $\bar b_i$ the background mean of feature $i$.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A model to explain: maps a batch of rows ($m \times d$ tensor) to one output per row ($m$ values, as a tensor or an
 * array).
 */
export type ScalarModel = (x: Tensor) => Tensor | ArrayLike<number>

/** Shapley values and the decomposition they give. */
export type Attribution = {
  /** One value $\phi_i$ per feature ($d$ values). */
  values: Float64Array
  /** $v(\emptyset)$: the expected output (over the background) with no feature known. */
  base: number
  /**
   * $v(\{1, \dots, d\}) = f(\xvec)$. $\text{base} + \sum_i \phi_i = \text{output}$ (exactly for exact methods, and
   * by construction for KernelSHAP).
   */
  output: number
}

const MAX_EXACT = 20

/**
 * The binomial coefficient $\binom{n}{k}$, as a float, by the product $\prod_{i=1}^{k} (n - k + i)/i$.
 *
 * @param n The size of the set.
 * @param k The size of the subset chosen ($0 \le k \le n$; 0 gives 1).
 * @returns $\binom{n}{k}$.
 */
function binomial(n: Size, k: Size): number {
  let r = 1
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i
  return r
}

/**
 * The Shapley kernel weight $\pi(s) = (d - 1)/\big(\binom{d}{s}\, s\, (d - s)\big)$ of a coalition of size $s$,
 * $0 < s < d$. It is infinite for the empty and the full coalition, which KernelSHAP imposes as constraints instead.
 *
 * @param d The number of players (features).
 * @param s The coalition's size.
 * @returns $\pi(s)$, or `Infinity` when $s \le 0$ or $s \ge d$.
 *
 * @example Small and large coalitions weigh most
 * for (let s = 0; s <= 4; s++) print(`s = ${s}:`, shapleyKernelWeight(4, s))
 */
export function shapleyKernelWeight(d: Size, s: Size): number {
  if (s <= 0 || s >= d) return Infinity
  return (d - 1) / (binomial(d, s) * s * (d - s))
}

/**
 * Exact Shapley values of a set function over $d \le 20$ players by enumerating all $2^d$ coalitions (each evaluated
 * once). Throws `DomainError` when $d$ is not an integer in $1, \dots, 20$.
 *
 * @param value The set function $v$: receives a coalition as a membership mask of $d$ booleans (`mask[i]` true when
 *   player $i$ is in it) and returns its worth.
 * @param d The number of players.
 * @returns The Shapley values, with `base` $= v(\emptyset)$ and `output` $= v$ of all players.
 *
 * @example A two-player AND splits its worth evenly
 * const and = (mask) => (mask[0] && mask[1] ? 1 : 0)
 * print(exactShapley(and, 2))
 *
 * @example A linear model: weight times (x minus the background mean)
 * const model = (X) => matmul(X, tensor([2, -1, 0.5]))
 * const v = interventionalValue(model, [1, 2, 3], [[0, 0, 0], [2, 0, 2]])
 * const { values, base, output } = exactShapley(v, 3)
 * print('values =', values)
 * print('expected =', [2 * (1 - 1), -1 * (2 - 0), 0.5 * (3 - 1)])
 * print('base + sum =', base + values.reduce((a, b) => a + b), ' output =', output)
 */
export function exactShapley(value: (mask: readonly boolean[]) => number, d: Size): Attribution {
  if (!(Number.isInteger(d) && d >= 1 && d <= MAX_EXACT))
    throw new DomainError('exactShapley', `exactShapley: d must lie in 1 … ${MAX_EXACT}, got ${d}`)
  const total = 1 << d
  const v = new Float64Array(total)
  for (let c = 0; c < total; c++) v[c] = value(Array.from({ length: d }, (_, i) => ((c >> i) & 1) === 1))
  // Weights by coalition size: s!(d − s − 1)!/d!.
  const weight = Float64Array.from({ length: d }, (_, s) => 1 / (d * binomial(d - 1, s)))
  const phi = new Float64Array(d)
  for (let c = 0; c < total; c++) {
    let size = 0
    for (let i = 0; i < d; i++) size += (c >> i) & 1
    for (let i = 0; i < d; i++) if (!((c >> i) & 1)) phi[i] += weight[size] * (v[c | (1 << i)] - v[c])
  }
  return { values: phi, base: v[0], output: v[total - 1] }
}

/**
 * The interventional value function of `model` at $\xvec$ over `background` rows: $v(S)$ is the mean output when the
 * features in the coalition $S$ come from $\xvec$ and the others from each background row. Each call evaluates the
 * model once, on a batch of $b$ rows. Throws `ShapeError` when $\xvec$ and the background differ in their number of
 * features.
 *
 * @param model The model, called on the $b \times d$ batch of mixed rows.
 * @param x The instance $\xvec$ to explain ($d$ values).
 * @param background The background rows ($b \times d$) that stand in for the features left out of a coalition.
 * @returns $v$: a function of a membership mask of $d$ booleans (`mask[i]` true when feature $i$ is taken from
 *   $\xvec$), returning the mean model output.
 *
 * @example The empty and the full coalition
 * const model = (X) => matmul(X, tensor([1, 1]))
 * const v = interventionalValue(model, [3, 4], [[0, 0], [2, 2]])
 * print('v(none) =', v([false, false]))
 * print('v({0}) =', v([true, false]))
 * print('v(all) =', v([true, true]))
 */
export function interventionalValue(
  model: ScalarModel,
  x: VectorLike,
  background: MatrixLike,
): (mask: readonly boolean[]) => number {
  const xv = dense.toF64(x, 'interventionalValue')
  const bg = dense.toMatrixF64(background, 'interventionalValue')
  const d = xv.length
  if (bg.n !== d)
    throw new ShapeError('interventionalValue', `interventionalValue: x has ${d} features, the background ${bg.n}`)
  return (mask) => {
    const rows = Float64Array.from(bg.data)
    for (let r = 0; r < bg.m; r++) for (let i = 0; i < d; i++) if (mask[i]) rows[r * d + i] = xv[i]
    const out = model(fromData(rows, [bg.m, d]))
    const o = 'shape' in out ? dense.data(out as Tensor) : out
    let s = 0
    for (let r = 0; r < bg.m; r++) s += o[r]
    return s / bg.m
  }
}

/** Options of `kernelShap`. */
export type KernelShapOptions = {
  /**
   * Coalitions to evaluate (default $2d + 2048$). When it reaches $2^d - 2$ (and $d \le 20$) every proper coalition is
   * enumerated with its exact kernel weight and the result is the exact Shapley value; otherwise coalitions are
   * sampled from `stream`, in complementary pairs (an odd budget is rounded down).
   */
  samples?: Size
  /** The random stream coalitions are sampled from; required when they are sampled rather than enumerated. */
  stream?: Stream
}

/**
 * KernelSHAP attribution of `model` at $\xvec$ against `background` rows (see the file comment): the weighted
 * least-squares fit over enumerated or sampled coalitions, with $\sum_i \phi_i = f(\xvec) - v(\emptyset)$ imposed
 * exactly. With one feature it returns $\phi_1 = f(\xvec) - v(\emptyset)$ without fitting. Throws `DomainError` when
 * coalitions must be sampled and no `stream` is given.
 *
 * @param model The model, called on batches of $b$ mixed rows ($b \times d$).
 * @param x The instance $\xvec$ to explain ($d$ values).
 * @param background The background rows ($b \times d$) that stand in for features left out of a coalition.
 * @param options The coalition budget and the stream to sample them from.
 * @returns The attribution, with `coalitions`, the number of coalitions fitted, and `exact`, true when every proper
 *   coalition was enumerated (the values are then the exact Shapley values).
 *
 * @example A linear model, enumerated exactly
 * const model = (X) => matmul(X, tensor([2, -1, 0.5]))
 * const r = kernelShap(model, [1, 2, 3], [[0, 0, 0], [2, 0, 2]])
 * print('values =', r.values, ' exact =', r.exact)
 * print('w (x - mean b) =', [2 * (1 - 1), -1 * (2 - 0), 0.5 * (3 - 1)])
 *
 * @example Sampled coalitions approximate the exact values
 * // Features 0 to 2 act only together, features 3 to 7 add up: exact values 1/3, 1/3, 1/3, then 1 each.
 * const model = (X) => toArray(X).map((r) => r[0] * r[1] * r[2] + r.slice(3).reduce((a, b) => a + b))
 * const x = [1, 1, 1, 1, 1, 1, 1, 1]
 * const background = [[0, 0, 0, 0, 0, 0, 0, 0]]
 * const r = kernelShap(model, x, background, { samples: 30, stream: stream(0) })
 * print('exact =', r.exact, ' coalitions =', r.coalitions)
 * print('sampled =', r.values)
 * print('enumerated =', kernelShap(model, x, background).values)
 */
export function kernelShap(
  model: ScalarModel,
  x: VectorLike,
  background: MatrixLike,
  options: KernelShapOptions = {},
): Attribution & { coalitions: Size; exact: boolean } {
  const v = interventionalValue(model, x, background)
  const d = dense.toF64(x, 'kernelShap').length
  const base = v(new Array<boolean>(d).fill(false))
  const output = v(new Array<boolean>(d).fill(true))
  if (d === 1) return { values: Float64Array.of(output - base), base, output, coalitions: 0, exact: true }
  const budget = options.samples ?? 2 * d + 2048
  const exact = d <= MAX_EXACT && budget >= 2 ** d - 2
  const masks: boolean[][] = []
  const weights: number[] = []
  if (exact) {
    for (let c = 1; c < (1 << d) - 1; c++) {
      const mask = Array.from({ length: d }, (_, i) => ((c >> i) & 1) === 1)
      masks.push(mask)
      weights.push(shapleyKernelWeight(d, mask.filter(Boolean).length))
    }
  } else {
    if (!options.stream) throw new DomainError('kernelShap', 'kernelShap: sampling coalitions needs a stream')
    const s = options.stream
    // Kernel mass per size s: C(d, s) π(s) ∝ 1/(s(d − s)).
    const mass = Float64Array.from({ length: d - 1 }, (_, k) => 1 / ((k + 1) * (d - k - 1)))
    const total = mass.reduce((a, b) => a + b, 0)
    for (let k = 0; k + 1 < budget; k += 2) {
      let u = (uniform(s) as number) * total
      let size = 1
      while (size < d - 1 && u > mass[size - 1]) u -= mass[size++ - 1]
      const order = Array.from({ length: d }, (_, i) => i)
      for (let i = 0; i < size; i++) {
        const j = i + integers(s, d - i)
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      const mask = new Array<boolean>(d).fill(false)
      for (let i = 0; i < size; i++) mask[order[i]] = true
      masks.push(
        mask,
        mask.map((b) => !b),
      )
      weights.push(1, 1)
    }
  }
  // Weighted least squares with Σφ = output − base, eliminating φ_{d−1} (minimum-norm when under-determined).
  const delta = output - base
  const q = d - 1
  const A = new Float64Array(q * q)
  const rhs = new Float64Array(q)
  masks.forEach((mask, k) => {
    const w = weights[k]
    const zl = mask[q] ? 1 : 0
    const y = v(mask) - base - zl * delta
    const row = Float64Array.from({ length: q }, (_, i) => (mask[i] ? 1 : 0) - zl)
    for (let i = 0; i < q; i++) {
      rhs[i] += w * row[i] * y
      for (let j = 0; j < q; j++) A[i * q + j] += w * row[i] * row[j]
    }
  })
  // The pseudo-inverse solution: with few sampled coalitions the normal equations can be singular.
  const head = dense.data(lstsq(fromData(A, [q, q]), fromData(rhs, [q])).x)
  const values = new Float64Array(d)
  values.set(head)
  values[q] = delta - head.reduce((a, b) => a + b, 0)
  return { values, base, output, coalitions: masks.length, exact }
}
