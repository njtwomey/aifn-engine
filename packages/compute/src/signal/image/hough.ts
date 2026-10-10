/**
 * The Hough transform: every edge pixel votes for every shape through it, and shapes are the peaks of the vote
 * accumulator. Lines in the normal form $\rho = x \cos\theta + y \sin\theta$ (Duda and Hart, 1972, "Use of the
 * Hough transformation to detect lines and curves in pictures", CACM 15(1); after Hough, 1962) and circles of given
 * radii.
 *
 * $x$ is the column and $y$ the row (downwards), so $\theta = 0$ is a vertical line at column $\rho$ and
 * $\theta = -\pi/2$ a horizontal line at row $-\rho$. Every nonzero pixel of the input is an edge pixel.
 */

import { fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { readImage, type ImageInput } from 'aifn-compute/foundation/convolution'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Rounding with halves away from zero (C's `round`, which scikit-image's accumulator uses).
 *
 * @param v The value to round.
 * @returns The nearest integer, $\pm 2.5$ going to $\pm 3$.
 */
const roundHalfAway = (v: number) => Math.sign(v) * Math.round(Math.abs(v))

/** The line accumulator: `votes` indexed by distance $\rho$ (row) and angle $\theta$ (column). */
export interface HoughLines {
  /** Votes, (number of distances) $\times$ (number of angles). */
  votes: Tensor
  /** The angles $\theta$ (radians) of the accumulator's columns. */
  angles: Vector
  /** The distances $\rho$ (pixels) of the accumulator's rows. */
  distances: Vector
}

/** A detected line $\rho = x \cos\theta + y \sin\theta$ ($x$ = column, $y$ = row) with its votes. */
export interface HoughLine {
  /** The angle $\theta$ of the line's normal, in radians. */
  angle: number
  /** The signed distance $\rho$ of the line from the origin (the top-left pixel), in pixels. */
  distance: number
  /** The votes of its accumulator cell: the number of edge pixels on the line. */
  votes: number
}

/**
 * The line Hough accumulator of a binary (or nonzero-marked) edge image, as `skimage.transform.hough_line`: angles
 * default to 180 values from $-\pi/2$ (inclusive) to $\pi/2$ (exclusive), distances run from $-D$ to $D$ in unit
 * steps with $D = \lceil \sqrt{h^2 + w^2} \rceil$, and each nonzero pixel ($x$ = column, $y$ = row) adds one vote at
 * $\operatorname{round}(x \cos\theta + y \sin\theta)$ for each $\theta$ (halves rounded away from zero).
 *
 * @param edges The edge image, $h \times w$: every nonzero pixel votes.
 * @param options The angles to test.
 * @param options.angles The angles $\theta$ in radians (default 180 steps of one degree from $-\pi/2$).
 * @returns The $(2D + 1) \times n_\theta$ accumulator with its angles and distances.
 *
 * @example A vertical and a horizontal line, at angles 15 degrees apart
 * const img = Array.from({ length: 20 }, (_, r) => Array.from({ length: 20 }, (_, c) => +(c === 5 || r === 12)))
 * // Angles every 15 degrees, from -90 up to 75.
 * const angles = Array.from({ length: 12 }, (_, i) => ((i * 15 - 90) * Math.PI) / 180)
 * const acc = houghLines(img, { angles })
 * print('accumulator:', acc.votes.shape)
 * print('most votes:', max(acc.votes))
 */
export function houghLines(edges: ImageInput, { angles }: { angles?: ArrayLike<number> } = {}): HoughLines {
  const { v, h, w } = readImage(edges, 'houghLines')
  const theta = angles
    ? Float64Array.from(angles)
    : Float64Array.from({ length: 180 }, (_, i) => -Math.PI / 2 + (Math.PI * i) / 180)
  const D = Math.ceil(Math.sqrt(h * h + w * w))
  const nr = 2 * D + 1
  const nt = theta.length
  const votes = new Float64Array(nr * nt)
  const cos = theta.map(Math.cos)
  const sin = theta.map(Math.sin)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!v[y * w + x]) continue
      for (let j = 0; j < nt; j++) votes[(roundHalfAway(cos[j] * x + sin[j] * y) + D) * nt + j]++
    }
  return {
    votes: fromData(votes, [nr, nt]),
    angles: fromData(theta, [nt]),
    distances: fromData(
      Float64Array.from({ length: nr }, (_, i) => i - D),
      [nr],
    ),
  }
}

/** Options for `houghLinePeaks`. */
export interface HoughPeakOptions {
  /** Suppress neighbours within this many distance bins of a peak. Default 9. */
  minDistance?: number
  /** Suppress neighbours within this many angle bins of a peak. Default 10. */
  minAngle?: number
  /** Keep peaks with at least this many votes. Default half the largest vote. */
  threshold?: number
  /** Keep at most this many. Default all. */
  count?: number
}

/**
 * Lines from the accumulator's peaks, strongest first: the largest remaining cell is taken and the cells within
 * `minDistance` distance bins and `minAngle` angle bins of it are suppressed (angles wrap with a flip of $\rho$), as
 * `skimage.transform.hough_line_peaks`. Of equal cells, the one with the smallest distance, then angle, is taken first.
 * Stops below the threshold, at `count` lines, or when no votes remain.
 *
 * @param acc The accumulator of `houghLines`; not modified.
 * @param options The suppression window, the threshold and the most lines.
 * @returns The lines, strongest first.
 *
 * @example A vertical line at column 5 and a horizontal one at row 12
 * const img = Array.from({ length: 20 }, (_, r) => Array.from({ length: 20 }, (_, c) => +(c === 5 || r === 12)))
 * // Angles every 15 degrees, from -90 up to 75.
 * const angles = Array.from({ length: 12 }, (_, i) => ((i * 15 - 90) * Math.PI) / 180)
 * // With 15-degree bins, suppress one bin either side rather than the default ten.
 * for (const line of houghLinePeaks(houghLines(img, { angles }), { minAngle: 1 }))
 *   print('angle (degrees):', (line.angle * 180) / Math.PI, 'distance:', line.distance, 'votes:', line.votes)
 */
export function houghLinePeaks(acc: HoughLines, options: HoughPeakOptions = {}): HoughLine[] {
  const [nr, nt] = acc.votes.shape
  const votes = Float64Array.from(acc.votes.data)
  const md = options.minDistance ?? 9
  const ma = options.minAngle ?? 10
  const max = votes.reduce((a, b) => Math.max(a, b), 0)
  const threshold = options.threshold ?? 0.5 * max
  const out: HoughLine[] = []
  const theta = acc.angles.data
  const rho = acc.distances.data
  while (options.count === undefined || out.length < options.count) {
    let best = -1
    for (let i = 0; i < votes.length; i++) if (best < 0 || votes[i] > votes[best]) best = i
    if (best < 0 || !(votes[best] >= threshold) || votes[best] <= 0) break
    const r = Math.floor(best / nt)
    const t = best % nt
    out.push({ angle: theta[t], distance: rho[r], votes: votes[best] })
    for (let dt = -ma; dt <= ma; dt++) {
      let tt = t + dt
      let rr0 = r
      // Wrapping θ past ±π/2 flips the sign of ρ: (ρ, θ) and (−ρ, θ ± π) are the same line.
      if (tt < 0 || tt >= nt) {
        tt = (tt + nt) % nt
        rr0 = nr - 1 - r
      }
      for (let dr = -md; dr <= md; dr++) {
        const rr = rr0 + dr
        if (rr >= 0 && rr < nr) votes[rr * nt + tt] = 0
      }
    }
  }
  return out
}

/** A detected circle: centre (row, column), radius and its normalised votes. */
export interface HoughCircle {
  /** The centre's row. */
  row: number
  /** The centre's column. */
  col: number
  /** The radius, one of those searched. */
  radius: number
  /** Votes over the number of perimeter samples (1 for a complete circle). */
  score: number
}

/**
 * The circle Hough accumulators for the given radii: each nonzero pixel votes for every centre at distance $r$ from it,
 * on $\max(8, \lceil 2\pi r \rceil)$ evenly spaced directions rounded to pixels, each distinct pixel offset voting
 * once; votes are normalised by the number of distinct offsets, so a complete circle scores about 1. Centres outside
 * the image get no votes. Throws `DomainError` for a radius that is not positive.
 *
 * @param edges The edge image, $h \times w$: every nonzero pixel votes.
 * @param radii The radii to search, in pixels.
 * @returns One $h \times w$ accumulator per radius, in the order of `radii`.
 *
 * @example A ring of radius 4 scores 1 at its centre for that radius
 * const ring = Array.from({ length: 15 }, (_, r) =>
 *   Array.from({ length: 15 }, (_, c) => +(Math.abs(Math.hypot(r - 7, c - 7) - 4) < 0.5)),
 * )
 * const [r3, r4, r5] = houghCircles(ring, [3, 4, 5])
 * print('best score at radius 3, 4, 5:', max(r3), max(r4), max(r5))
 * print('radius 4 at the centre (7, 7):', r4.data[7 * 15 + 7])
 */
export function houghCircles(edges: ImageInput, radii: ArrayLike<number>): Tensor[] {
  const { v, h, w } = readImage(edges, 'houghCircles')
  const out: Tensor[] = []
  for (const r of Array.from(radii)) {
    if (!(r > 0)) throw new DomainError('houghCircles', 'houghCircles: radii must be positive')
    const samples = Math.max(8, Math.ceil(2 * Math.PI * r))
    // The distinct pixel offsets of the discretised circle, so no centre gets two votes from one pixel.
    const offsets = new Map<string, [number, number]>()
    for (let k = 0; k < samples; k++) {
      const a = (2 * Math.PI * k) / samples
      const dr = Math.round(r * Math.sin(a))
      const dc = Math.round(r * Math.cos(a))
      offsets.set(`${dr},${dc}`, [dr, dc])
    }
    const offs = [...offsets.values()]
    const acc = new Float64Array(h * w)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        if (!v[y * w + x]) continue
        for (const [dr, dc] of offs) {
          const cy = y + dr
          const cx = x + dc
          if (cy >= 0 && cy < h && cx >= 0 && cx < w) acc[cy * w + cx]++
        }
      }
    const norm = offs.length
    out.push(
      fromData(
        acc.map((a) => a / norm),
        [h, w],
      ),
    )
  }
  return out
}

/**
 * Circles from the accumulators of `houghCircles`, strongest first: the best cell over all radii is taken, then every
 * cell within `minDistance` (default the radius) of its centre is suppressed at all radii; stops below `threshold`
 * (default 0.5) or at `count`. Throws `DomainError` unless there is one radius per accumulator.
 *
 * @param accumulators The accumulators of `houghCircles`, one per radius; not modified.
 * @param radii The radius of each accumulator.
 * @param options The threshold, the most circles and the suppression distance.
 * @param options.threshold The least normalised score a circle may have.
 * @param options.count The most circles to return (default all).
 * @param options.minDistance The distance in pixels from a found centre within which other centres are suppressed
 *   (default the found circle's radius).
 * @returns The circles, strongest first.
 *
 * @example Find the ring's centre and radius
 * const ring = Array.from({ length: 15 }, (_, r) =>
 *   Array.from({ length: 15 }, (_, c) => +(Math.abs(Math.hypot(r - 7, c - 7) - 4) < 0.5)),
 * )
 * const radii = [3, 4, 5]
 * print(houghCirclePeaks(houghCircles(ring, radii), radii))
 */
export function houghCirclePeaks(
  accumulators: readonly Tensor[],
  radii: ArrayLike<number>,
  { threshold = 0.5, count, minDistance }: { threshold?: number; count?: number; minDistance?: number } = {},
): HoughCircle[] {
  const rs = Array.from(radii)
  if (rs.length !== accumulators.length)
    throw new DomainError('houghCirclePeaks', 'houghCirclePeaks: one radius per accumulator')
  if (!rs.length) return []
  const [h, w] = accumulators[0].shape
  const acc = accumulators.map((a) => Float64Array.from(a.data))
  const out: HoughCircle[] = []
  while (count === undefined || out.length < count) {
    let bk = -1
    let bi = -1
    for (let k = 0; k < acc.length; k++)
      for (let i = 0; i < acc[k].length; i++) if (bk < 0 || acc[k][i] > acc[bk][bi]) [bk, bi] = [k, i]
    if (bk < 0 || !(acc[bk][bi] >= threshold)) break
    const row = Math.floor(bi / w)
    const col = bi % w
    out.push({ row, col, radius: rs[bk], score: acc[bk][bi] })
    const d = minDistance ?? rs[bk]
    for (let k = 0; k < acc.length; k++)
      for (let r = Math.max(0, Math.floor(row - d)); r <= Math.min(h - 1, Math.ceil(row + d)); r++)
        for (let c = Math.max(0, Math.floor(col - d)); c <= Math.min(w - 1, Math.ceil(col + d)); c++)
          if (Math.hypot(r - row, c - col) <= d) acc[k][r * w + c] = 0
  }
  return out
}
