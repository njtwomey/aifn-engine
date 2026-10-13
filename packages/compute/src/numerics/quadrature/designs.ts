/**
 * Space-filling designs on the unit cube $[0, 1)^d$ and measures of how evenly they fill it: Latin hypercubes (McKay,
 * Beckman and Conover, 1979), the random shift of Cranley and Patterson (1976), scrambled Sobol points (Matoušek's
 * random linear scramble with a digital shift, as scipy's `Sobol(scramble=True)`, or Owen's nested uniform
 * scrambling, Owen 1995 and 1998), scrambled Halton points (random digit permutations, Owen 2017, as scipy's
 * `Halton(scramble=True)`), the exact star discrepancy and the largest gap of a one-dimensional projection.
 *
 * Each randomised design draws from the stream it is given, a number of words fixed by $n$ and $d$ (the Sobol and
 * Halton scrambles by $d$ alone, so the first $n$ points of a larger design are the design of $n$ points), and is
 * unbiased for integration: every point is uniform on the cube. The scrambles keep the structure of the sequence
 * they randomise. For $n = 2^m$ the scrambled Sobol points are a $(t, m, d)$-net in base 2 with the $t$ of the
 * unscrambled points, so every elementary box $\prod_j [a_j 2^{-k_j}, (a_j + 1) 2^{-k_j})$ of volume $2^{t - m}$
 * holds exactly $2^t$ of them; scrambled Halton points keep the stratification of each coordinate in its base.
 *
 * The star discrepancy of $n$ points $\xvec_i \in [0, 1]^d$ is the largest gap between the fraction of points in an
 * anchored box and its volume,
 *
 * $$D^*_n = \sup_{\tvec \in [0, 1]^d} \left\lvert \frac{\#\{i : \xvec_i \in [\mathbf{0}, \tvec)\}}{n}
 *   - \prod_j t_j \right\rvert,$$
 *
 * the quantity in the Koksma–Hlawka bound. The supremum is attained at, or approached from above at, the corners of
 * the grid whose coordinates are the points' coordinates and 1, so it is computed exactly over that grid.
 */

import { randomBits, permutation, units, philox4x32, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { directions, PRIMES, SOBOL_DIMENSIONS } from './multivariate'

/**
 * Check a design's size: $n \ge 0$ and $d \ge 1$ integers, $d$ at most `maxD`.
 *
 * @param n The number of points.
 * @param d The number of dimensions.
 * @param maxD The most dimensions the design supports.
 * @param where The caller's name for error messages.
 */
function checkSize(n: number, d: number, maxD: number, where: string): void {
  if (!(Number.isInteger(n) && n >= 0)) throw new DomainError(where, `${where}: n must be a non-negative integer`)
  if (!(Number.isInteger(d) && d >= 1)) throw new DomainError(where, `${where}: d must be a positive integer`)
  if (d > maxD) throw new ShapeError(where, `${where}: at most ${maxD} dimensions, got ${d}`)
}

/**
 * A Latin hypercube sample of $n$ points in $[0, 1)^d$ (McKay, Beckman and Conover, 1979): each axis is cut into $n$
 * equal slices and each slice of each axis holds exactly one point. Coordinate $j$ of point $i$ is
 * $(\pi_j(i) + u_{ij})/n$ for independent random permutations $\pi_j$ of $0, \dots, n - 1$ and uniforms $u_{ij}$
 * ($\tfrac12$ when `centred`), as scipy's `LatinHypercube` (`scramble=True`, or `False` for centred points). Every
 * one-dimensional projection is stratified; pairs of coordinates are not.
 *
 * @param s The stream; advanced by the $d$ permutations and then by the $n d$ uniforms (none when `centred`).
 * @param n The number of points.
 * @param d The number of dimensions.
 * @param options How the point sits in its cell.
 * @param options.centred Put each point at the centre of its cell instead of uniformly inside it (default false).
 * @returns The design, an $n \times d$ matrix with one point per row.
 *
 * @example Each of the four slices of each axis holds one point
 * const x = latinHypercube(stream(1), 4, 2)
 * print('points =', x)
 * print('slices of x1 =', toFlat(x).filter((_, k) => k % 2 === 0).map((v) => Math.floor(4 * v)))
 */
export function latinHypercube(s: Stream, n: Size, d: Size, { centred = false }: { centred?: boolean } = {}): Tensor {
  checkSize(n, d, Infinity, 'latinHypercube')
  const out = new Float64Array(n * d)
  for (let j = 0; j < d; j++) {
    const p = permutation(s, n).data
    for (let i = 0; i < n; i++) out[i * d + j] = p[i]
  }
  const u = centred ? null : units(s, n * d)
  for (let k = 0; k < n * d; k++) out[k] = (out[k] + (u === null ? 0.5 : u[k])) / n
  return fromData(out, [n, d])
}

/**
 * The points shifted by one uniform vector $\uvec$ modulo 1, $\xvec_i \mapsto (\xvec_i + \uvec) \bmod 1$ (Cranley and
 * Patterson, 1976): the simplest randomisation of a quasi-random design. Each shifted point is uniform on the cube, so
 * an average over the shifted design is unbiased; the gaps between points (on the torus) are kept.
 *
 * @param s The stream; advanced by the $d$ uniforms of the shift.
 * @param points The design, $n \times d$, in $[0, 1)^d$. Not modified.
 * @returns The shifted design, $n \times d$, in $[0, 1)^d$.
 *
 * @example A shifted Sobol design keeps its spacing on each axis
 * const x = randomShift(stream(2), sobol(4, 2))
 * print('shifted =', x)
 * print('sorted x1 =', toFlat(x).filter((_, k) => k % 2 === 0).sort())
 */
export function randomShift(s: Stream, points: MatrixLike): Tensor {
  const { data, m: n, n: d } = dense.toMatrixF64(points, 'randomShift')
  const u = units(s, d)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      const v = data[i * d + j] + u[j]
      data[i * d + j] = v >= 1 ? v - 1 : v
    }
  return fromData(data, [n, d])
}

const TWO_POW_32 = 2 ** 32

/**
 * The parity (sum modulo 2) of the bits of a 32-bit word.
 *
 * @param w The word.
 * @returns 0 or 1.
 */
function parity(w: number): number {
  w ^= w >>> 16
  w ^= w >>> 8
  w ^= w >>> 4
  w ^= w >>> 2
  w ^= w >>> 1
  return w & 1
}

/**
 * The 32-bit integers of Sobol points $i = $ `skip`, ..., `skip` $+ n - 1$ in Gray-code order, from given direction
 * numbers and a starting word per dimension (a digital shift).
 *
 * @param n The number of points.
 * @param V The direction numbers of each dimension, as `directions` returns them (index 1 to 32).
 * @param start The word each dimension starts from: 0 for the plain sequence, the shift for a shifted one.
 * @param skip The number of leading points to leave out.
 * @returns The points' words, $n d$ values row-major.
 */
function sobolWords(n: number, V: readonly Uint32Array[], start: Uint32Array, skip: number): Uint32Array {
  const d = V.length
  const out = new Uint32Array(n * d)
  const X = Uint32Array.from(start)
  for (let i = 0; i < skip + n; i++) {
    if (i > 0) {
      // The index (1-based) of the lowest zero bit of i − 1.
      let c = 1
      let value = i - 1
      while (value & 1) {
        value >>>= 1
        c++
      }
      for (let j = 0; j < d; j++) X[j] = (X[j] ^ V[j][c]) >>> 0
    }
    if (i >= skip) out.set(X, (i - skip) * d)
  }
  return out
}

/** Options of `scrambledSobol`. */
export type ScrambledSobolOptions = {
  /**
   * The scramble: `lms` (default), Matoušek's (1998) random linear matrix scramble with a digital shift, as scipy's
   * `Sobol(scramble=True)`; or `owen`, Owen's (1995) nested uniform scrambling, a random flip of each digit given the
   * digits before it.
   */
  scramble?: 'lms' | 'owen'
  /** The number of leading points of the scrambled sequence to leave out (default 0). */
  skip?: Size
}

/**
 * $n$ scrambled Sobol points in $[0, 1)^d$, $d \le 21$: the Sobol sequence of `sobol` (Joe and Kuo's direction numbers,
 * Gray-code order) randomised so that each point is uniform on the cube while the net structure is kept. For
 * $n = 2^m$ the points are a $(t, m, d)$-net in base 2: an elementary box of volume $2^{t - m}$ holds $2^t$ points.
 *
 * With `scramble: 'lms'` each dimension's generating matrix $\Cmat_j$ (32 binary digits) becomes $\Lmat_j\Cmat_j$ for
 * a random lower-triangular binary $\Lmat_j$ with unit diagonal, and the digits are then added (exclusive or) to a
 * random shift $\evec_j$: $\xvec \mapsto \Lmat_j\Cmat_j \mathbf{a}_i \oplus \evec_j$ (Matoušek, 1998), the scramble
 * scipy's `Sobol(scramble=True)` applies; the points lie on the grid of spacing $2^{-32}$. With `scramble: 'owen'`
 * digit $k$ of each coordinate is flipped by a random bit that depends on the digits before it (nested uniform
 * scrambling; Owen, 1995, 1998), the flips drawn by hashing the digits with a key from the stream, and the digits
 * beyond the 32nd are filled with random bits, so each coordinate is uniform to double precision. scipy's draws
 * cannot be reproduced, only their laws.
 *
 * @param s The stream; advanced by 33 words per dimension (`lms`) or 4 (`owen`), whatever $n$ is.
 * @param n The number of points; a power of 2 keeps the net's balance.
 * @param d The number of dimensions, at most 21.
 * @param options The scramble and the points to skip, as `ScrambledSobolOptions`.
 * @returns The design, an $n \times d$ matrix with one point per row.
 *
 * @example Eight scrambled points: one in each eighth of each axis
 * const x = scrambledSobol(stream(3), 8, 2)
 * print('points =', x)
 * print('eighths of x2 =', toFlat(x).filter((_, k) => k % 2 === 1).map((v) => Math.floor(8 * v)))
 *
 * @example Owen's scrambling of the same sequence
 * print('owen =', scrambledSobol(stream(3), 4, 2, { scramble: 'owen' }))
 */
export function scrambledSobol(s: Stream, n: Size, d: Size, options: ScrambledSobolOptions = {}): Tensor {
  const { scramble = 'lms', skip = 0 } = options
  checkSize(n, d, SOBOL_DIMENSIONS, 'scrambledSobol')
  if (!(Number.isInteger(skip) && skip >= 0))
    throw new DomainError('scrambledSobol', 'scrambledSobol: skip must be a non-negative integer')
  const out = new Float64Array(n * d)
  if (scramble === 'lms') {
    const words = randomBits(s, 33 * d)
    const V: Uint32Array[] = []
    const shift = new Uint32Array(d)
    for (let j = 0; j < d; j++) {
      const w = words.subarray(33 * j, 33 * j + 33)
      // Row r of L (output digit r + 1, most significant first) as a mask over input digits: unit diagonal, random
      // entries left of it.
      const rows = new Uint32Array(32)
      for (let r = 0; r < 32; r++)
        rows[r] = ((r === 0 ? 0 : (w[r] & (0xffffffff << (32 - r))) >>> 0) | (1 << (31 - r))) >>> 0
      const v = directions(j)
      const scrambled = new Uint32Array(33)
      for (let c = 1; c <= 32; c++) {
        let y = 0
        for (let r = 0; r < 32; r++) if (parity(rows[r] & v[c])) y |= 1 << (31 - r)
        scrambled[c] = y >>> 0
      }
      V.push(scrambled)
      shift[j] = w[32]
    }
    const X = sobolWords(n, V, shift, skip)
    for (let k = 0; k < n * d; k++) out[k] = X[k] / TWO_POW_32
    return fromData(out, [n, d])
  }
  if (scramble !== 'owen')
    throw new DomainError('scrambledSobol', `scrambledSobol: scramble must be 'lms' or 'owen', got ${scramble}`)
  const keys = randomBits(s, 4 * d)
  const X = sobolWords(
    n,
    Array.from({ length: d }, (_, j) => directions(j)),
    new Uint32Array(d),
    skip,
  )
  const block = new Uint32Array(4)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      const x = X[i * d + j]
      const [k0, k1, k2, k3] = keys.subarray(4 * j, 4 * j + 4)
      let y = 0
      // The flip of digit k + 1 is a hash of the k digits before it (the prefix x >>> (32 − k)) and of k.
      for (let k = 0; k < 32; k++) {
        const prefix = k === 0 ? 0 : x >>> (32 - k)
        philox4x32(prefix, k, k2, k3, k0, k1, block)
        y |= (((x >>> (31 - k)) ^ block[0]) & 1) << (31 - k)
      }
      // Digits beyond the 32nd: random given all 32 before them.
      philox4x32(x, 32, k2, k3, k0, k1, block)
      out[i * d + j] = ((y >>> 0) + (block[0] >>> 11) / 2 ** 21) / TWO_POW_32
    }
  return fromData(out, [n, d])
}

/**
 * $n$ scrambled Halton points in $[0, 1)^d$, $d \le 25$: coordinate $j$ of point $i$ is the radical inverse of $i$ in
 * the $j$-th prime base $b$ with each digit put through a random permutation of $0, \dots, b - 1$, one permutation per
 * digit position and dimension, leading zeros included:
 *
 * $$x_{ij} = \sum_{k \ge 1} \pi_{jk}(a_k(i))\, b^{-k}, \qquad i = \sum_{k \ge 1} a_k(i)\, b^{k - 1},$$
 *
 * over the $\lceil 54/\log_2 b \rceil - 1$ positions a double resolves. This is Owen's (2017) random digit
 * permutation, as scipy's `Halton(scramble=True)`; it breaks the correlation between coordinates in large bases that
 * makes plain Halton points line up, and keeps each coordinate's stratification: the first $b^m$ points hold one
 * point in each interval $[a b^{-m}, (a + 1) b^{-m})$ of coordinate $j$.
 *
 * @param s The stream; advanced by the permutations, a number of words fixed by $d$.
 * @param n The number of points.
 * @param d The number of dimensions, at most 25.
 * @param options The points to skip.
 * @param options.skip The index of the first point (default 0).
 * @returns The design, an $n \times d$ matrix with one point per row.
 *
 * @example Nine points: one in each ninth of the base-3 axis
 * const x = scrambledHalton(stream(4), 9, 2)
 * print('points =', x)
 * print('ninths of x2 =', toFlat(x).filter((_, k) => k % 2 === 1).map((v) => Math.floor(9 * v)))
 */
export function scrambledHalton(s: Stream, n: Size, d: Size, { skip = 0 }: { skip?: Size } = {}): Tensor {
  checkSize(n, d, PRIMES.length, 'scrambledHalton')
  if (!(Number.isInteger(skip) && skip >= 0))
    throw new DomainError('scrambledHalton', 'scrambledHalton: skip must be a non-negative integer')
  const out = new Float64Array(n * d)
  for (let j = 0; j < d; j++) {
    const b = PRIMES[j]
    const count = Math.ceil(54 / Math.log2(b)) - 1
    const perms = Array.from({ length: count }, () => permutation(s, b).data)
    const terms = new Float64Array(count)
    for (let i = 0; i < n; i++) {
      let rest = i + skip
      let scale = 1 / b
      for (let k = 0; k < count; k++) {
        const digit = rest % b
        rest = (rest - digit) / b
        terms[k] = perms[k][digit] * scale
        scale /= b
      }
      // Smallest terms first.
      let x = 0
      for (let k = count - 1; k >= 0; k--) x += terms[k]
      out[i * d + j] = x
    }
  }
  return fromData(out, [n, d])
}

/**
 * The sorted distinct values of a column and 1, for the grid of candidate box corners.
 *
 * @param data The points, row-major.
 * @param n The number of points.
 * @param d The number of dimensions.
 * @param j The column.
 * @returns The distinct values of column $j$ in ascending order, with 1 appended when it is not among them.
 */
function corners(data: Float64Array, n: number, d: number, j: number): number[] {
  const v = Array.from({ length: n }, (_, i) => data[i * d + j]).sort((a, b) => a - b)
  const out: number[] = []
  for (const x of v) if (out.length === 0 || out[out.length - 1] !== x) out.push(x)
  if (out[out.length - 1] !== 1) out.push(1)
  return out
}

/** The most box corners times points `starDiscrepancy` counts by enumeration in three or more dimensions. */
const MAX_WORK = 2e8

/**
 * The exact star discrepancy $D^*_n$ of $n$ points in $[0, 1]^d$: the largest difference between the fraction of the
 * points in an anchored box $[\mathbf{0}, \tvec)$ and the box's volume. It is the maximum over the grid of corners
 * $\tvec$ whose coordinates are the points' coordinates or 1 of $\prod_j t_j - A(\tvec)/n$, $A$ counting the points
 * in the open box, and of $\bar A(\tvec)/n - \prod_j t_j$, $\bar A$ counting those in the closed box
 * $[\mathbf{0}, \tvec]$ (the limit from above). In one dimension that is a pass over the sorted points, in two a
 * sweep over the grid in $\mathcal{O}(n^2)$ time; in three or more every corner is counted, in
 * $\mathcal{O}(n^{d + 1})$ time, and a design too large for that throws `DomainError`. This is the $L^\infty$ star
 * discrepancy, not the $L^2$ one scipy's `qmc.discrepancy` computes. Throws `DomainError` for a coordinate outside
 * $[0, 1]$.
 *
 * @param points The design, $n \times d$ with $n \ge 1$, coordinates in $[0, 1]$. Not modified.
 * @returns $D^*_n$, in $(0, 1]$.
 *
 * @example Evenly spaced points against a Latin hypercube
 * print('midpoints =', starDiscrepancy([[0.125], [0.375], [0.625], [0.875]]))
 * print('Sobol, 16 points =', starDiscrepancy(sobol(16, 2)))
 * print('Latin hypercube, 16 points =', starDiscrepancy(latinHypercube(stream(5), 16, 2)))
 */
export function starDiscrepancy(points: MatrixLike): number {
  const where = 'starDiscrepancy'
  const { data, m: n, n: d } = dense.toMatrixF64(points, where)
  if (n === 0 || d === 0) throw new ShapeError(where, `${where}: need at least one point and one dimension`)
  if (!data.every((v) => v >= 0 && v <= 1)) throw new DomainError(where, `${where}: coordinates must be in [0, 1]`)
  const grids = Array.from({ length: d }, (_, j) => corners(data, n, d, j))
  let best = 0
  if (d === 1) {
    const x = Float64Array.from(data).sort()
    let open = 0
    for (const a of grids[0]) {
      while (open < n && x[open] < a) open++
      let closed = open
      while (closed < n && x[closed] <= a) closed++
      best = Math.max(best, a - open / n, closed / n - a)
    }
    return best
  }
  if (d === 2) {
    // Sweep the first coordinate's corners; counts over the second come from histograms by the rank of y among its
    // corners, so a point of rank r lies in the open box up to corner k when r < k and in the closed one when r <= k.
    const [cx, cy] = grids
    const rank = (y: number) => {
      let lo = 0
      let hi = cy.length - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (cy[mid] < y) lo = mid + 1
        else hi = mid
      }
      return lo
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => data[2 * a] - data[2 * b])
    const ry = order.map((i) => rank(data[2 * i + 1]))
    const open = new Float64Array(cy.length)
    const closed = new Float64Array(cy.length)
    let next = 0
    for (const a of cx) {
      // The points with x < a are those with x <= the previous corner.
      open.set(closed)
      while (next < n && data[2 * order[next]] <= a) closed[ry[next++]]++
      let below = 0
      let within = 0
      for (let k = 0; k < cy.length; k++) {
        within += closed[k]
        const volume = a * cy[k]
        best = Math.max(best, volume - below / n, within / n - volume)
        below += open[k]
      }
    }
    return best
  }
  let cells = 1
  for (const g of grids) cells *= g.length
  if (cells * n > MAX_WORK)
    throw new DomainError(where, `${where}: ${cells} box corners for ${n} points is too many to count exactly`)
  const index = new Array<number>(d).fill(0)
  for (let c = 0; c < cells; c++) {
    let volume = 1
    for (let j = 0; j < d; j++) volume *= grids[j][index[j]]
    let open = 0
    let closed = 0
    for (let i = 0; i < n; i++) {
      let inOpen = true
      let inClosed = true
      for (let j = 0; j < d && inClosed; j++) {
        const x = data[i * d + j]
        const t = grids[j][index[j]]
        if (x > t) inClosed = false
        if (x >= t) inOpen = false
      }
      open += +inOpen
      closed += +inClosed
    }
    best = Math.max(best, volume - open / n, closed / n - volume)
    for (let j = 0; j < d; j++) {
      if (++index[j] < grids[j].length) break
      index[j] = 0
    }
  }
  return best
}

/**
 * The largest gap between neighbouring values on an interval, the ends included: for values $x_{(1)} \le \dots \le
 * x_{(n)}$ in $[l, h]$, the largest of $x_{(1)} - l$, $x_{(i + 1)} - x_{(i)}$ and $h - x_{(n)}$. Read off one
 * coordinate of a design it says how wide a band of that axis no trial explores; $n$ evenly spaced values leave
 * $(h - l)/n$ at best, and $n$ uniform draws leave $\tfrac{h - l}{n + 1}\sum_{k=1}^{n+1} \tfrac1k$ on average. Throws
 * `DomainError` for a value outside $[l, h]$ or for $l > h$.
 *
 * @param values The values, $n \ge 0$ of them, in any order. Not modified.
 * @param options The interval.
 * @param options.lo Its lower end $l$ (default 0).
 * @param options.hi Its upper end $h$ (default 1).
 * @returns The largest gap; $h - l$ when there are no values.
 *
 * @example Sixteen multiples of 1/16 against sixteen uniform draws
 * print('grid =', maxGap(Array.from({ length: 16 }, (_, i) => i / 16)))
 * print('uniform draws =', maxGap(uniform(stream(6), 0, 1, { shape: [16] })))
 */
export function maxGap(values: VectorLike, { lo = 0, hi = 1 }: { lo?: number; hi?: number } = {}): number {
  const v = dense.toF64(values, 'maxGap').sort()
  if (!(lo <= hi)) throw new DomainError('maxGap', `maxGap: need lo <= hi, got ${lo}, ${hi}`)
  if (!v.every((x) => x >= lo && x <= hi)) throw new DomainError('maxGap', `maxGap: values must lie in [${lo}, ${hi}]`)
  let best = 0
  let previous = lo
  for (const x of v) {
    best = Math.max(best, x - previous)
    previous = x
  }
  return Math.max(best, hi - previous)
}
