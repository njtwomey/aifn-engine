/**
 * Routing by agreement between two layers of capsules (Sabour, Frosst and Hinton, 2017, "Dynamic Routing Between
 * Capsules", §2 and Procedure 1).
 *
 * A capsule's output is a vector whose length, kept below 1 by the squash nonlinearity
 * $\vvec = \frac{\lVert \svec \rVert^2}{1 + \lVert \svec \rVert^2} \frac{\svec}{\lVert \svec \rVert}$, is the
 * probability that its entity is present. Each lower capsule $i$ predicts the output of every upper capsule $j$,
 * $\hat{\uvec}_{j|i} = \Wmat_{ij}\uvec_i$. Routing starts from logits $b_{ij}$ (zero, or given priors), and each
 * iteration takes the couplings $c_{i\cdot} = \operatorname{softmax}_j(b_{i\cdot})$, forms
 * $\svec_j = \sum_i c_{ij}\hat{\uvec}_{j|i}$, squashes it to $\vvec_j$, and adds the agreement
 * $a_{ij} = \hat{\uvec}_{j|i} \cdot \vvec_j$ to $b_{ij}$. The paper's output is the last iteration's $\vvec_j$.
 *
 * Plain arrays of numbers, forward only, no randomness: a $1152 \times 10 \times 16$ routing (CapsNet's DigitCaps
 * layer) costs about 0.4 M multiply-adds per iteration. A capsule's length is `norm` of
 * `aifn-compute/foundation/tensor`.
 */

/**
 * Squash a capsule's total input $\svec$ to a vector of the same direction and length
 * $\lVert \svec \rVert^2 / (1 + \lVert \svec \rVert^2) < 1$ (zero stays zero).
 *
 * @param s The total input $\svec$.
 * @returns The squashed vector $\vvec$, a new array.
 *
 * @example Long inputs approach length 1, short ones length 0
 * for (const s of [[3, 4], [0.1, 0]]) print(s, '->', squash(s))
 */
export function squash(s: readonly number[]): number[] {
  const sq = s.reduce((sum, x) => sum + x * x, 0)
  if (sq === 0) return s.map(() => 0)
  const scale = sq / (1 + sq) / Math.sqrt(sq)
  return s.map((x) => scale * x)
}

/**
 * The predictions $\hat{\uvec}_{j|i} = \Wmat_{ij}\uvec_i$ of every upper capsule $j$ by every lower capsule $i$.
 *
 * @param weights `weights[i][j]` is the $D_{\text{out}} \times D_{\text{in}}$ matrix $\Wmat_{ij}$, as rows of length
 *   $D_{\text{in}}$.
 * @param poses `poses[i]` is the lower capsule's output $\uvec_i$, of length $D_{\text{in}}$.
 * @returns `u[i][j]`, the prediction $\hat{\uvec}_{j|i}$, of length $D_{\text{out}}$.
 *
 * @example A rotation by 90 degrees
 * print('prediction =', capsulePredictions([[[[0, -1], [1, 0]]]], [[1, 2]]))
 */
export function capsulePredictions(
  weights: readonly (readonly (readonly (readonly number[])[])[])[],
  poses: readonly (readonly number[])[],
): number[][][] {
  return weights.map((row, i) => row.map((w) => w.map((r) => r.reduce((sum, wk, k) => sum + wk * poses[i][k], 0))))
}

/** One iteration of routing by agreement. */
export type RoutingIteration = {
  /** Couplings $c_{ij}$ used in this iteration (each row sums to 1 over $j$). */
  couplings: number[][]
  /** Total inputs $\svec_j$ of the upper capsules. */
  inputs: number[][]
  /** Upper capsule outputs $\vvec_j$, the squashed inputs. */
  outputs: number[][]
  /** Agreements $a_{ij} = \hat{\uvec}_{j|i} \cdot \vvec_j$. */
  agreements: number[][]
  /** Logits $b_{ij}$ after adding the agreements. */
  logits: number[][]
}

/**
 * Routing by agreement for `iterations` rounds, every round recorded (for a player over iterations); the paper's
 * output is the last entry's `outputs`. The softmax subtracts each row's maximum.
 *
 * @param predictions `predictions[i][j]` is $\hat{\uvec}_{j|i}$, as `capsulePredictions` returns.
 * @param iterations The number of rounds, at least 1 for a non-empty trace.
 * @param priors The initial logits `priors[i][j]`; left out, 0, so every lower capsule starts by sending equally to
 *   every upper capsule.
 * @returns One `RoutingIteration` per round, in order.
 *
 * @example Two lower and two upper capsules: the agreeing pair's coupling rises
 * const u = [[[1.5, 0.5], [-0.5, 1.5]], [[1.5, 0.9], [1, -1]]]
 * const trace = dynamicRoutingTrace(u, 3)
 * for (const t of trace) print('c11 =', t.couplings[0][0], ', |v1| =', norm(tensor(t.outputs[0])))
 */
export function dynamicRoutingTrace(
  predictions: readonly (readonly (readonly number[])[])[],
  iterations: number,
  priors?: readonly (readonly number[])[],
): RoutingIteration[] {
  const lower = predictions.length
  const upper = lower > 0 ? predictions[0].length : 0
  const dim = upper > 0 ? predictions[0][0].length : 0
  let logits = Array.from({ length: lower }, (_, i) => Array.from({ length: upper }, (_, j) => priors?.[i]?.[j] ?? 0))
  const trace: RoutingIteration[] = []
  for (let r = 0; r < iterations; r++) {
    const couplings = logits.map((b) => {
      const top = Math.max(...b)
      const e = b.map((x) => Math.exp(x - top))
      const z = e.reduce((a, x) => a + x, 0)
      return e.map((x) => x / z)
    })
    const inputs = Array.from({ length: upper }, (_, j) => {
      const s = new Array<number>(dim).fill(0)
      for (let i = 0; i < lower; i++) for (let k = 0; k < dim; k++) s[k] += couplings[i][j] * predictions[i][j][k]
      return s
    })
    const outputs = inputs.map((s) => squash(s))
    const agreements = predictions.map((row) => row.map((u, j) => u.reduce((sum, x, k) => sum + x * outputs[j][k], 0)))
    logits = logits.map((b, i) => b.map((x, j) => x + agreements[i][j]))
    trace.push({ couplings, inputs, outputs, agreements, logits })
  }
  return trace
}
