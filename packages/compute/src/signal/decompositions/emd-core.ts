/**
 * Empirical mode decomposition, the array-level compute (private to the module; the public API is in emd.ts). Ported from
 * the site's `time-frequency/_shared/emd.ts`, which follows the reference algorithm of Rilling, Flandrin and Gonçalvès
 * (2003, "On empirical mode decomposition and its algorithms", IEEE-EURASIP NSIP) as implemented by PyEMD: extrema
 * mirrored twice at each end, not-a-knot cubic-spline envelopes (natural for three knots), and PyEMD's default
 * stopping rule. Checked against PyEMD when it was written.
 */
export type Extrema = {
  /** Indices of local maxima and minima. */
  maxima: number[]
  minima: number[]
  /** Number of zero crossings. */
  zeroCrossings: number
}

/** Local extrema (a point above or below both neighbours; flat plateaus count once, at their middle). */
export function findExtrema(x: ArrayLike<number>): Extrema {
  const n = x.length
  const maxima: number[] = []
  const minima: number[] = []
  let i = 1
  while (i < n - 1) {
    const before = x[i] - x[i - 1]
    if (before === 0) {
      i++
      continue
    }
    // Walk across a plateau, if any.
    let j = i
    while (j < n - 1 && x[j + 1] === x[j]) j++
    if (j >= n - 1) break
    const after = x[j + 1] - x[j]
    const mid = j === i ? i : Math.round((i + j) / 2)
    if (before > 0 && after < 0) maxima.push(mid)
    else if (before < 0 && after > 0) minima.push(mid)
    i = j + 1
  }
  return { maxima, minima, zeroCrossings: countZeroCrossings(x) }
}

/** Sign changes between neighbours, plus runs of exact zeros counted once each. */
export function countZeroCrossings(x: ArrayLike<number>): number {
  let count = 0
  for (let i = 0; i + 1 < x.length; i++) if (x[i] * x[i + 1] < 0) count++
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== 0) continue
    count++
    while (i + 1 < x.length && x[i + 1] === 0) i++
  }
  return count
}

type Knots = { t: number[]; y: number[] }

/**
 * Mirror up to `nbsym` extrema about each end so that the envelopes are interpolated, not extrapolated, over the
 * whole signal. The mirror is the first (last) extremum when the signal's end lies inside the envelopes, and the end
 * sample itself otherwise, which then also becomes a knot. A port of Rilling's `boundary_conditions`.
 */
export function mirrorKnots(x: ArrayLike<number>, maxima: number[], minima: number[], nbsym = 2) {
  const n = x.length
  const last = n - 1
  const take = (a: number[], from: number, to: number) => a.slice(Math.max(0, from), Math.max(0, to)).reverse()
  let lmax: number[], lmin: number[], lsym: number
  if (maxima[0] < minima[0]) {
    if (x[0] > x[minima[0]]) {
      lmax = take(maxima, 1, Math.min(maxima.length, nbsym + 1))
      lmin = take(minima, 0, Math.min(minima.length, nbsym))
      lsym = maxima[0]
    } else {
      lmax = take(maxima, 0, Math.min(maxima.length, nbsym))
      lmin = [...take(minima, 0, Math.min(minima.length, nbsym - 1)), 0]
      lsym = 0
    }
  } else if (x[0] < x[maxima[0]]) {
    lmax = take(maxima, 0, Math.min(maxima.length, nbsym))
    lmin = take(minima, 1, Math.min(minima.length, nbsym + 1))
    lsym = minima[0]
  } else {
    lmax = [...take(maxima, 0, Math.min(maxima.length, nbsym - 1)), 0]
    lmin = take(minima, 0, Math.min(minima.length, nbsym))
    lsym = 0
  }
  const nMax = maxima.length
  const nMin = minima.length
  let rmax: number[], rmin: number[], rsym: number
  if (maxima[nMax - 1] < minima[nMin - 1]) {
    if (x[last] < x[maxima[nMax - 1]]) {
      rmax = take(maxima, nMax - nbsym, nMax)
      rmin = take(minima, nMin - nbsym - 1, nMin - 1)
      rsym = minima[nMin - 1]
    } else {
      rmax = [...maxima.slice(Math.max(nMax - nbsym + 1, 0)), last].reverse()
      rmin = take(minima, nMin - nbsym, nMin)
      rsym = last
    }
  } else if (x[last] > x[minima[nMin - 1]]) {
    rmax = take(maxima, nMax - nbsym - 1, nMax - 1)
    rmin = take(minima, nMin - nbsym, nMin)
    rsym = maxima[nMax - 1]
  } else {
    rmax = take(maxima, nMax - nbsym, nMax)
    rmin = [...minima.slice(Math.max(nMin - nbsym + 1, 0)), last].reverse()
    rsym = last
  }
  if (!lmin.length) lmin = minima
  if (!rmin.length) rmin = minima
  if (!lmax.length) lmax = maxima
  if (!rmax.length) rmax = maxima
  const reflect = (about: number, idx: number[]) => idx.map((i) => 2 * about - i)
  // If a mirrored knot lands inside the signal, mirror about the end sample instead.
  if (reflect(lsym, lmin)[0] > 0 || reflect(lsym, lmax)[0] > 0) {
    if (lsym === maxima[0]) lmax = take(maxima, 0, Math.min(nMax, nbsym))
    else lmin = take(minima, 0, Math.min(nMin, nbsym))
    lsym = 0
  }
  if (reflect(rsym, rmin).at(-1)! < last || reflect(rsym, rmax).at(-1)! < last) {
    if (rsym === maxima[nMax - 1]) rmax = take(maxima, nMax - nbsym, nMax)
    else rmin = take(minima, nMin - nbsym, nMin)
    rsym = last
  }
  const knots = (l: number[], mid: number[], r: number[]): Knots => {
    const t = [...reflect(lsym, l), ...mid, ...reflect(rsym, r)]
    const y = [...l.map((i) => x[i]), ...mid.map((i) => x[i]), ...r.map((i) => x[i])]
    // Drop repeated knots (a mirrored end sample can coincide with its neighbour).
    const keep = t.map((ti, i) => i === t.length - 1 || t[i + 1] !== ti)
    return { t: t.filter((_, i) => keep[i]), y: y.filter((_, i) => keep[i]) }
  }
  return { upper: knots(lmax, maxima, rmax), lower: knots(lmin, minima, rmin) }
}

/**
 * Cubic spline through (t, y), evaluated at the integers 0..n−1. Four or more knots use not-a-knot end conditions
 * (SciPy's default), three use natural ones, two a straight line. Outside the knots the end pieces are extrapolated.
 * Second derivatives M solve a tridiagonal system after the not-a-knot rows are folded into their neighbours.
 */
export function cubicSpline(t: number[], y: number[], n: number): Float64Array {
  const k = t.length
  const out = new Float64Array(n)
  if (k < 2) return out.fill(y[0] ?? 0)
  const h = Array.from({ length: k - 1 }, (_, i) => t[i + 1] - t[i])
  const slope = h.map((hi, i) => (y[i + 1] - y[i]) / hi)
  const M = new Array<number>(k).fill(0)
  if (k >= 3) {
    // Unknowns M_1..M_{k−2}: rows h_{i−1} M_{i−1} + 2(h_{i−1}+h_i) M_i + h_i M_{i+1} = 6 (slope_i − slope_{i−1}).
    const m = k - 2
    const a = new Array<number>(m).fill(0)
    const b = new Array<number>(m).fill(0)
    const c = new Array<number>(m).fill(0)
    const d = new Array<number>(m).fill(0)
    for (let r = 0; r < m; r++) {
      const i = r + 1
      a[r] = h[i - 1]
      b[r] = 2 * (h[i - 1] + h[i])
      c[r] = h[i]
      d[r] = 6 * (slope[i] - slope[i - 1])
    }
    const notAKnot = k >= 4
    if (notAKnot) {
      // M_0 = ((h0+h1) M_1 − h0 M_2)/h1 and M_{k−1} = ((h_{k−3}+h_{k−2}) M_{k−2} − h_{k−2} M_{k−3})/h_{k−3}.
      b[0] += (h[0] * (h[0] + h[1])) / h[1]
      c[0] -= (h[0] * h[0]) / h[1]
      const p = h[k - 3]
      const q = h[k - 2]
      b[m - 1] += (q * (p + q)) / p
      a[m - 1] -= (q * q) / p
    }
    // Thomas algorithm.
    for (let r = 1; r < m; r++) {
      const w = a[r] / b[r - 1]
      b[r] -= w * c[r - 1]
      d[r] -= w * d[r - 1]
    }
    const sol = new Array<number>(m)
    sol[m - 1] = d[m - 1] / b[m - 1]
    for (let r = m - 2; r >= 0; r--) sol[r] = (d[r] - c[r] * sol[r + 1]) / b[r]
    for (let r = 0; r < m; r++) M[r + 1] = sol[r]
    if (notAKnot) {
      M[0] = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1]
      M[k - 1] = ((h[k - 3] + h[k - 2]) * M[k - 2] - h[k - 2] * M[k - 3]) / h[k - 3]
    }
  }
  let seg = 0
  for (let s = 0; s < n; s++) {
    while (seg < k - 2 && s > t[seg + 1]) seg++
    const hi = h[seg]
    const u = t[seg + 1] - s
    const v = s - t[seg]
    out[s] =
      (M[seg] * u * u * u + M[seg + 1] * v * v * v) / (6 * hi) +
      (y[seg] / hi - (M[seg] * hi) / 6) * u +
      (y[seg + 1] / hi - (M[seg + 1] * hi) / 6) * v
  }
  return out
}

/** One sifting step: the next candidate, the envelopes and their mean, and the (mirrored) knots. */
export type SiftStep = {
  /** The signal being sifted, before this step. */
  h: Float64Array
  maxima: number[]
  minima: number[]
  upper: Float64Array
  lower: Float64Array
  mean: Float64Array
  /** h − mean: the candidate IMF after this step. */
  next: Float64Array
  /** Knots of the upper and lower envelopes, including mirrored ones. */
  knots: { upper: Knots; lower: Knots }
}

/** One sifting step, or null when h has too few extrema to build envelopes. */
export function siftStep(h: Float64Array, mirror = true): SiftStep | null {
  const { maxima, minima } = findExtrema(h)
  if (maxima.length + minima.length < 3 || !maxima.length || !minima.length) return null
  const n = h.length
  const knots = mirror
    ? mirrorKnots(h, maxima, minima)
    : {
        upper: { t: maxima, y: maxima.map((i) => h[i]) },
        lower: { t: minima, y: minima.map((i) => h[i]) },
      }
  const upper = cubicSpline(knots.upper.t, knots.upper.y, n)
  const lower = cubicSpline(knots.lower.t, knots.lower.y, n)
  const mean = new Float64Array(n)
  const next = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    mean[i] = 0.5 * (upper[i] + lower[i])
    next[i] = h[i] - mean[i]
  }
  return { h, maxima, minima, upper, lower, mean, next, knots }
}

/**
 * When to stop sifting one IMF.
 * - `fixed`: a fixed number of sifts (Wu & Huang use 10 in ensemble EMD).
 * - `sd`: Huang's SD = Σ (h_{k−1} − h_k)² / Σ h_{k−1}² below a threshold, 0.2 to 0.3 in Huang et al. (1998). (The
 *   original sums the ratio pointwise; the ratio of sums used here avoids division by near-zero samples.)
 * - `snumber`: the numbers of extrema and zero crossings differ by at most one for S consecutive sifts.
 * - `rilling`: with σ(t) = |mean| / amplitude, σ < θ1 on all but a fraction α of samples and σ < θ2 everywhere.
 * - `pyemd`: PyEMD's default rule (scaled variance, SD or energy ratio, plus the extrema/zero-crossing condition).
 */
export type StopRule =
  | { kind: 'fixed'; sifts: number }
  | { kind: 'sd'; threshold: number }
  | { kind: 'snumber'; s: number }
  | { kind: 'rilling'; theta1: number; theta2: number; alpha: number }
  | { kind: 'pyemd' }

/** Huang's SD between consecutive sifts, as a ratio of sums. */
export function sdCriterion(prev: ArrayLike<number>, next: ArrayLike<number>): number {
  let num = 0
  let den = 0
  for (let i = 0; i < prev.length; i++) {
    num += (prev[i] - next[i]) ** 2
    den += prev[i] ** 2
  }
  return den > 0 ? num / den : 0
}

/** Rilling's evaluation: fraction of samples with σ(t) = |mean|/amplitude above θ1, and max σ. */
export function rillingStats(step: SiftStep, theta1: number) {
  let above = 0
  let max = 0
  for (let i = 0; i < step.mean.length; i++) {
    const amp = Math.abs(step.upper[i] - step.lower[i]) / 2
    const s = Math.abs(step.mean[i]) / Math.max(amp, 1e-12)
    if (s > theta1) above++
    if (s > max) max = s
  }
  return { fractionAbove: above / step.mean.length, maxSigma: max }
}

/** PyEMD's default stopping test (scaled variance, SD or energy ratio) for the step from `prev` to `next`. */
export function pyemdCheck(next: Float64Array, prev: Float64Array, step: SiftStep): boolean {
  // Mirrored maxima must be positive and minima negative.
  if (step.knots.upper.y.some((v) => v < 0) || step.knots.lower.y.some((v) => v > 0)) return false
  let energy = 0
  let diff2 = 0
  let prevEnergy = 0
  let std = 0
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < next.length; i++) {
    const d = next[i] - prev[i]
    energy += next[i] * next[i]
    diff2 += d * d
    prevEnergy += prev[i] * prev[i]
    std += (d / next[i]) ** 2
    lo = Math.min(lo, prev[i])
    hi = Math.max(hi, prev[i])
  }
  if (energy < 1e-10) return false
  if (diff2 / (hi - lo) < 0.001) return true
  if (std < 0.2) return true
  return diff2 / prevEnergy < 0.2
}

/** The result of `sift`. */
export type SiftResult = {
  imf: Float64Array
  /** Every step taken, in order (empty when x had too few extrema). */
  steps: SiftStep[]
  /** False when x had too few extrema to sift: it is then a residue, not an IMF. */
  oscillating: boolean
}

/**
 * The stopping test after one sifting step from `prev` to `h` = `step.next`: whether the rule holds, and the updated
 * count of consecutive balanced steps (the S-number rule's counter). `n` is the number of sifts made so far.
 */
export function stopTest(
  rule: StopRule,
  prev: Float64Array,
  step: SiftStep,
  n: number,
  balancedRun: number,
  mirror: boolean,
): { converged: boolean; balancedRun: number } {
  const h = step.next
  if (rule.kind === 'fixed') return { converged: n >= rule.sifts, balancedRun }
  const e = findExtrema(h)
  const balanced = Math.abs(e.maxima.length + e.minima.length - e.zeroCrossings) < 2
  if (rule.kind === 'pyemd') return { converged: balanced && pyemdCheck(h, prev, step), balancedRun }
  if (rule.kind === 'sd') return { converged: sdCriterion(prev, h) < rule.threshold, balancedRun }
  if (rule.kind === 'snumber') {
    const run = balanced ? balancedRun + 1 : 0
    return { converged: run >= rule.s, balancedRun: run }
  }
  const next = siftStep(h, mirror)
  if (!next) return { converged: true, balancedRun }
  const { fractionAbove, maxSigma } = rillingStats(next, rule.theta1)
  return { converged: balanced && fractionAbove <= rule.alpha && maxSigma < rule.theta2, balancedRun }
}

/** Sift one IMF out of x. */
export function sift(x: ArrayLike<number>, rule: StopRule = { kind: 'pyemd' }, mirror = true, maxSifts = 1000) {
  let h: Float64Array = Float64Array.from(x)
  const steps: SiftStep[] = []
  let balancedRun = 0
  let oscillating = true
  for (let n = 1; n < maxSifts; n++) {
    const step = siftStep(h, mirror)
    if (!step) {
      oscillating = false
      break
    }
    steps.push(step)
    const prev = h
    h = step.next
    const test = stopTest(rule, prev, step, n, balancedRun, mirror)
    balancedRun = test.balancedRun
    if (test.converged) break
  }
  return { imf: h, steps, oscillating } satisfies SiftResult
}

/** The arrays of an empirical mode decomposition. */
export type EmdArrays = { imfs: Float64Array[]; residue: Float64Array }

/** Options of the array-level `emd`. */
export type EmdCoreOptions = { maxImfs?: number; rule?: StopRule; mirror?: boolean }

/** Empirical mode decomposition: x = Σ imfs + residue, fastest IMF first. */
export function emd(
  x: ArrayLike<number>,
  { maxImfs = -1, rule = { kind: 'pyemd' }, mirror = true }: EmdCoreOptions = {},
) {
  const n = x.length
  const imfs: Float64Array[] = []
  const residue = Float64Array.from(x)
  while (maxImfs < 0 || imfs.length < maxImfs) {
    const { imf, oscillating } = sift(residue, rule, mirror)
    if (!oscillating) break
    imfs.push(imf)
    let lo = Infinity
    let hi = -Infinity
    let l1 = 0
    for (let i = 0; i < n; i++) {
      residue[i] -= imf[i]
      lo = Math.min(lo, residue[i])
      hi = Math.max(hi, residue[i])
      l1 += Math.abs(residue[i])
    }
    if (hi - lo < 0.001 || l1 < 0.005) break
  }
  return { imfs, residue } satisfies EmdArrays
}
