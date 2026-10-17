/**
 * How well a one-state, input-dependent linear recurrence tracks parity when its state is stored with finite
 * precision.
 *
 * The recurrence is $h_0 = 1$ and $h_t = a(x_t) h_{t-1}$ for bits $x_t$, with $a(0) = 1$ (a zero leaves the state
 * alone) and $a(1) = \lambda$. After a string with $c$ ones, $h = \lambda^c$ in exact arithmetic. Here the state is
 * rounded at every step to a fixed-point grid of spacing $2^{-b}$, truncating toward zero as a fixed-point register
 * does, so $h$ depends only on $c$ and is $\lambda^c$ (truncated) until its magnitude falls below the grid, after which
 * it is 0. Truncation, unlike round-to-nearest, has no nonzero fixed points for $\lvert\lambda\rvert < 1$, so every
 * decaying state eventually reaches 0. Parity is read from the final state by either a single threshold (the best
 * linear readout of one number) or any function of the stored value (a lookup table: the best possible readout). Over
 * all $2^T$ strings of length $T$ the number of ones is $\mathrm{Binomial}(T, 1/2)$, so each accuracy is an exact sum
 * over $c$.
 *
 * Sources: Grazzi et al. (2024), "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues", Theorem 1
 * (finite-precision linear RNNs with only nonnegative eigenvalues cannot solve parity) and the construction with
 * $a(1) = -1$; Sarrof, Veitsman and Hahn (2024), Theorem 2 (nonnegative gates cannot recognise PARITY). The
 * fixed-point rounding is a stand-in for finite precision, chosen so the loss of information is visible at small
 * lengths; it truncates on purpose, which the format rounding of `aifn-compute/nn/quantise` does not.
 */

/**
 * Truncate toward zero to the grid of spacing $2^{-b}$ ($-0$ is returned as 0).
 *
 * @param x The value.
 * @param bits The binary digits $b$ kept after the point.
 * @returns $\operatorname{trunc}(x\,2^b)/2^b$.
 *
 * @example Truncation, not rounding
 * print('0.9 at 0 bits:', roundToGrid(0.9, 0))
 * print('-1.5 at 1 bit:', roundToGrid(-1.5, 1))
 * print('0.7 at 2 bits:', roundToGrid(0.7, 2))
 */
export function roundToGrid(x: number, bits: number): number {
  const n = Math.trunc(x * 2 ** bits)
  return n === 0 ? 0 : n / 2 ** bits
}

/**
 * The rounded states $h_1, \dots, h_T$ of the recurrence over `bits`, from $h_0 = 1$.
 *
 * @param bits The bits $x_t$, each 0 or 1.
 * @param lambda The transition $\lambda$ of a one bit.
 * @param precision The binary digits $b$ kept after the point.
 * @returns The $T$ states.
 *
 * @example The string 1011 with lambda = -1 ends at -1: three ones, odd
 * print('h =', parityStates([1, 0, 1, 1], -1, 8))
 */
export function parityStates(bits: readonly number[], lambda: number, precision: number): number[] {
  const out: number[] = []
  let h = 1
  for (const x of bits) {
    if (x === 1) h = roundToGrid(lambda * h, precision)
    out.push(h)
  }
  return out
}

/**
 * The rounded state after $c$ ones, for $c = 0, \dots, C$ ($h = 1$ before any one).
 *
 * @param lambda The transition $\lambda$ of a one bit.
 * @param precision The binary digits $b$ kept after the point.
 * @param maxCount The largest count $C$.
 * @returns The $C + 1$ states.
 *
 * @example Halving at 8 bits reaches 0 after nine ones
 * print('h(c) =', stateValues(0.5, 8, 10))
 */
export function stateValues(lambda: number, precision: number, maxCount: number): number[] {
  const values: number[] = [1]
  for (let c = 1; c <= maxCount; c++) values.push(roundToGrid(lambda * values[c - 1], precision))
  return values
}

/**
 * The number of ones after which the rounded state is 0, or `Infinity` if it never is within `maxCount` ones.
 *
 * @param lambda The transition $\lambda$ of a one bit.
 * @param precision The binary digits $b$ kept after the point.
 * @param maxCount The most ones to try.
 * @returns The count $c$ at which the state first truncates to 0.
 *
 * @example A slow decay lasts longer, a flip never decays
 * print('lambda 0.5:', onesUntilZero(0.5, 8))
 * print('lambda 0.9:', onesUntilZero(0.9, 8))
 * print('lambda -1:', onesUntilZero(-1, 8, 1000))
 */
export function onesUntilZero(lambda: number, precision: number, maxCount = 1 << 16): number {
  let h = 1
  for (let c = 1; c <= maxCount; c++) {
    h = roundToGrid(lambda * h, precision)
    if (h === 0) return c
  }
  return Infinity
}

/** How parity is read from the final state: one `threshold` on it, or `any` function of it. */
export type ParityReadout = 'threshold' | 'any'

/** Options of `parityAccuracyByLength`. */
export type ParityAccuracyOptions = {
  /** The transition $a(1) = \lambda$ for a one bit, in $[-1, 1]$. */
  lambda: number
  /** The binary digits $b$ after the point kept in the state. */
  precision: number
  /** The longest string length $T$. */
  maxLength: number
  /** The readout: one threshold on the state, or any function of it. */
  readout: ParityReadout
}

/**
 * The best parity accuracy over all strings of each length $T = 1, \dots, T_{\max}$ (uniform bits), and the state
 * after $c$ ones for $c = 0, \dots, T_{\max}$. Exact sums over the binomial counts, $O(T_{\max}^2)$ (about 3 ms at
 * $T_{\max} = 256$).
 *
 * @param options The transition, the precision, the longest length and the readout.
 * @returns The `accuracy` at each length and the stored `values` after each count of ones.
 *
 * @example A flip tracks parity exactly; a decay falls to chance once its state truncates to 0
 * const flip = parityAccuracyByLength({ lambda: -1, precision: 8, maxLength: 40, readout: 'threshold' })
 * const decay = parityAccuracyByLength({ lambda: 0.5, precision: 8, maxLength: 40, readout: 'any' })
 * print('lambda -1 at T = 40:', flip.accuracy[39])
 * print('lambda 0.5 at T = 8, 40:', decay.accuracy[7], decay.accuracy[39])
 */
export function parityAccuracyByLength(options: ParityAccuracyOptions): { accuracy: number[]; values: number[] } {
  const { lambda, precision, maxLength, readout } = options
  const values = stateValues(lambda, precision, maxLength)
  const accuracy: number[] = []
  for (let T = 1; T <= maxLength; T++) {
    // Binomial(T, 1/2) probabilities of c ones, by the ratio recurrence.
    const p: number[] = [2 ** -T]
    for (let c = 0; c < T; c++) p.push((p[c] * (T - c)) / (c + 1))
    // Probability mass of even and odd counts at each distinct stored value.
    const groups = new Map<number, [number, number]>()
    for (let c = 0; c <= T; c++) {
      const v = values[c] === 0 ? 0 : values[c] // −0 and +0 are one stored value
      const g = groups.get(v) ?? [0, 0]
      g[c % 2] += p[c]
      groups.set(v, g)
    }
    if (readout === 'any') {
      let acc = 0
      for (const [even, odd] of groups.values()) acc += Math.max(even, odd)
      accuracy.push(acc)
    } else {
      const sorted = [...groups.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1])
      const totalEven = sorted.reduce((a, g) => a + g[0], 0)
      const totalOdd = sorted.reduce((a, g) => a + g[1], 0)
      let best = Math.max(totalEven, totalOdd)
      let belowEven = 0
      let belowOdd = 0
      for (let k = 0; k < sorted.length - 1; k++) {
        belowEven += sorted[k][0]
        belowOdd += sorted[k][1]
        // Even below the threshold and odd above it, or the other way round.
        best = Math.max(best, belowEven + (totalOdd - belowOdd), belowOdd + (totalEven - belowEven))
      }
      accuracy.push(best)
    }
  }
  return { accuracy, values }
}
