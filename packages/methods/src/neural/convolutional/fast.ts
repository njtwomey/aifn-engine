/**
 * Winograd's minimal filtering for convolutional layers, its float32 error, and a cost model comparing the direct
 * sum, im2col, FFT and Winograd convolution of one layer.
 *
 * Winograd's $F(m, r)$ gives $m$ outputs of an $r$-tap correlation from $n = m + r - 1$ multiplications,
 * $\yvec = \Amat^\top[(\Gmat\gvec) \odot (\Bmat^\top\dvec)]$, nested in two dimensions as
 * $\Ymat = \Amat^\top[(\Gmat\gvec\Gmat^\top) \odot (\Bmat^\top\Dmat\Bmat)]\Amat$ (Winograd, 1980, "Arithmetic
 * Complexity of Computations"; Lavin and Gray, 2016, "Fast Algorithms for Convolutional Neural Networks", §3 and §4).
 * The transforms come from the Toom-Cook construction: evaluate at $n - 1$ finite points and at infinity, multiply
 * pointwise, interpolate, then transpose the linear-convolution algorithm into a correlation. With Vandermonde
 * matrices $\Vmat_m$ ($n \times m$), $\Vmat_r$ ($n \times r$) and $\Vmat$ ($n \times n$) over the points,
 * $\Amat^\top = \Vmat_m^\top$, $\Gmat = \Vmat_r$ and $\Bmat^\top = \Vmat^{-\top}$; row $i$ of $\Bmat^\top$ is scaled
 * by $f_i = \prod_{j \ne i}(a_i - a_j)$ and row $i$ of $\Gmat$ by $1/f_i$, which makes $\Bmat^\top$ an integer matrix
 * for integer points (the convention of Lavin's wincnn). Lavin and Gray print other signs for some rows of $F(2, 3)$;
 * every sign choice gives the same result.
 *
 * The cost model of `convolutionAlgorithmCosts` counts real multiplications, a model stated here rather than a
 * measurement: direct $k^2 C C' H W$; FFT with transforms of $P = N_h N_w$ points ($N$ the next power of two at least
 * $H + k - 1$) at $P \log_2 P$ real multiplications per real 2-D transform, and pointwise products over the Hermitian
 * half at 4 real multiplications each; Winograd's $n^2$ elementwise products per tile per channel pair, plus one
 * multiplication per transform-matrix entry outside $\{0, \pm 1\}$ per use. im2col itself is
 * `im2col` of `aifn-compute/nn/functional`. Matrices here are plain arrays of rows.
 */

import { units, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { polyFromRoots } from 'aifn-compute/numerics/polynomial'

/** A dense matrix as an array of rows. */
export type Rows = number[][]

/**
 * The Winograd transforms of $F(m, r)$: $\Amat^\top$ ($m \times n$), $\Gmat$ ($n \times r$) and $\Bmat^\top$
 * ($n \times n$).
 */
export type WinogradTransforms = {
  /** Outputs per tile $m$. */
  m: number
  /** Filter taps $r$. */
  r: number
  /** Tile size $n = m + r - 1$. */
  n: number
  /** The $n - 1$ finite interpolation points. */
  points: number[]
  /** The output transform $\Amat^\top$, $m \times n$. */
  AT: Rows
  /** The filter transform $\Gmat$, $n \times r$. */
  G: Rows
  /** The data transform $\Bmat^\top$, $n \times n$. */
  BT: Rows
}

/**
 * The default interpolation points $0, \pm 1, \pm 2, \pm 1/2, \pm 3, \pm 1/3, \dots$, the first `count` of them
 * (Lavin's choice for $F(4, 3)$ and $F(6, 3)$).
 *
 * @param count How many points.
 * @returns The points, in that order.
 *
 * @example The five points of F(4, 3)
 * print('points =', winogradPoints(5))
 */
export function winogradPoints(count: number): number[] {
  const points = [0]
  for (let j = 1; points.length < count; j++) {
    points.push(j, -j)
    if (j > 1) points.push(1 / j, -1 / j)
  }
  return points.slice(0, count)
}

/**
 * Coefficients, constant term first, of $\prod_k (x - a_k)$, by `polyFromRoots` (which lists the highest power first).
 *
 * @param roots The roots $a_k$ (possibly none).
 * @returns The $1 + $ (number of roots) coefficients, constant term first.
 */
function polynomialFromRoots(roots: readonly number[]): number[] {
  return roots.length === 0 ? [1] : toFlat(polyFromRoots([...roots])).reverse()
}

/**
 * The Toom-Cook transforms of Winograd's minimal filtering algorithm $F(m, r)$, from $n - 1$ distinct finite points
 * and the point at infinity. Throws `DomainError` when $m$ or $r$ is not a positive integer, or the points are too
 * few or not distinct.
 *
 * @param m The number of outputs per tile.
 * @param r The number of filter taps.
 * @param points The $n - 1 = m + r - 2$ finite points; default `winogradPoints(m + r - 2)`.
 * @returns $\Amat^\top$, $\Gmat$ and $\Bmat^\top$ with $\yvec = \Amat^\top[(\Gmat\gvec) \odot (\Bmat^\top\dvec)]$
 *   for a tile $\dvec$ of $n$ inputs and a filter $\gvec$ of $r$ taps.
 *
 * @example F(2, 3): four multiplications for two outputs of a 3-tap filter
 * const t = winogradTransforms(2, 3)
 * print('AT =', t.AT)
 * print('G  =', t.G)
 * print('BT =', t.BT)
 */
export function winogradTransforms(m: number, r: number, points?: readonly number[]): WinogradTransforms {
  const where = 'winogradTransforms'
  if (!Number.isInteger(m) || !Number.isInteger(r) || m < 1 || r < 1) {
    throw new DomainError(where, `${where}: m and r must be positive integers`)
  }
  const n = m + r - 1
  const a = [...(points ?? winogradPoints(n - 1))]
  if (a.length !== n - 1) throw new DomainError(where, `${where}: F(${m}, ${r}) needs ${n - 1} finite points`)
  if (new Set(a).size !== a.length) throw new DomainError(where, `${where}: the points must be distinct`)
  const f = a.map((ai, i) => a.reduce((acc, aj, j) => (j === i ? acc : acc * (ai - aj)), 1))
  // Aᵀ = V_mᵀ: column i evaluates the output polynomial at point i; the last column takes its leading coefficient.
  const AT: Rows = Array.from({ length: m }, (_, row) => [...a.map((ai) => ai ** row), row === m - 1 ? 1 : 0])
  // G = V_r with row i divided by f_i; the last row is the filter's leading tap.
  const G: Rows = [
    ...a.map((ai, i) => Array.from({ length: r }, (_, col) => ai ** col / f[i])),
    Array.from({ length: r }, (_, col) => (col === r - 1 ? 1 : 0)),
  ]
  // Bᵀ = V⁻ᵀ with row i multiplied by f_i: the coefficients of Π_{j≠i}(x − a_j), padded to n; the last row is
  // Π_j (x − a_j), the polynomial that vanishes at every finite point.
  const BT: Rows = [
    ...a.map((_, i) => {
      const c = polynomialFromRoots(a.filter((_, j) => j !== i))
      return [...c, ...new Array<number>(n - c.length).fill(0)]
    }),
    polynomialFromRoots(a),
  ]
  return { m, r, n, points: a, AT, G, BT }
}

/** Arithmetic in float64, or every result rounded to float32 as a single-precision kernel would. */
type Round = (v: number) => number
/**
 * No rounding: float64 arithmetic.
 *
 * @param v A value.
 * @returns `v` unchanged.
 */
const exact: Round = (v) => v
/** Rounding to the nearest float32, after every operation of a single-precision kernel. */
const single: Round = Math.fround

/**
 * The product $\Amat\Bmat$ of two row arrays, rounding every product and partial sum by `round`.
 *
 * @param A The left factor.
 * @param B The right factor.
 * @param round The rounding applied after every operation.
 * @returns The product, as rows.
 */
function multiply(A: Rows, B: Rows, round: Round): Rows {
  return A.map((row) => B[0].map((_, j) => row.reduce((acc, aik, k) => round(acc + round(aik * B[k][j])), 0)))
}

/**
 * The transpose of a row array.
 *
 * @param A The matrix.
 * @returns $\Amat^\top$, as rows.
 */
const transpose = (A: Rows): Rows => A[0].map((_, j) => A.map((row) => row[j]))

/**
 * One two-dimensional Winograd tile $F(m \times m, r \times r)$: the $m \times m$ outputs of the cross-correlation
 * of an $n \times n$ data tile with an $r \times r$ filter,
 * $\Ymat = \Amat^\top[(\Gmat\gvec\Gmat^\top) \odot (\Bmat^\top\Dmat\Bmat)]\Amat$.
 *
 * @param d The data tile $\Dmat$, $n \times n$ rows.
 * @param g The filter, $r \times r$ rows.
 * @param t The transforms of $F(m, r)$, from `winogradTransforms`.
 * @param precision `float64`, or `float32` to round every product and sum (and the matrices) to single precision.
 * @returns The $m \times m$ outputs.
 *
 * @example Winograd against the direct sum on one tile
 * const t = winogradTransforms(2, 3)
 * const d = [[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]]
 * const g = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
 * print('winograd =', winogradTile(d, g, t))
 * print('direct   =', directTile(d, g, 2))
 */
export function winogradTile(
  d: Rows,
  g: Rows,
  t: WinogradTransforms,
  precision: 'float64' | 'float32' = 'float64',
): Rows {
  const round = precision === 'float32' ? single : exact
  const r = (M: Rows) => M.map((row) => row.map(round))
  const AT = r(t.AT)
  const G = r(t.G)
  const BT = r(t.BT)
  const U = multiply(multiply(G, r(g), round), transpose(G), round)
  const V = multiply(multiply(BT, r(d), round), transpose(BT), round)
  const M = U.map((row, i) => row.map((u, j) => round(u * V[i][j])))
  return multiply(multiply(AT, M, round), transpose(AT), round)
}

/**
 * The $m \times m$ outputs of the direct cross-correlation of a tile with an $r \times r$ filter, the reference
 * for `winogradTile`.
 *
 * @param d The data tile, at least $(m + r - 1) \times (m + r - 1)$ rows.
 * @param g The filter, $r \times r$ rows.
 * @param m The outputs per side.
 * @param precision `float64`, or `float32` to round every product and sum to single precision.
 * @returns The $m \times m$ outputs.
 *
 * @example A 3 x 3 vertical-edge filter on a 4 x 4 tile
 * const d = [[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]]
 * print('outputs =', directTile(d, [[1, 0, -1], [1, 0, -1], [1, 0, -1]], 2))
 */
export function directTile(d: Rows, g: Rows, m: number, precision: 'float64' | 'float32' = 'float64'): Rows {
  const round = precision === 'float32' ? single : exact
  const r = g.length
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let acc = 0
      for (let u = 0; u < r; u++)
        for (let v = 0; v < r; v++) acc = round(acc + round(round(g[u][v]) * round(d[i + u][j + v])))
      return acc
    }),
  )
}

/**
 * The mean relative error of float32 Winograd $F(m \times m, r \times r)$ and of the float32 direct sum against the
 * float64 direct sum, over random tiles and filters with entries uniform on $[-1, 1]$: per trial
 * $\max \lvert Y - Y_{\text{exact}} \rvert / \max \lvert Y_{\text{exact}} \rvert$, averaged over trials.
 *
 * @param m The outputs per tile side.
 * @param r The filter size.
 * @param s The random stream; advanced in place ($n^2 + r^2$ draws per trial, tile first).
 * @param trials The number of random tiles.
 * @returns The mean relative errors of Winograd and of the direct sum, both in float32.
 *
 * @example The error grows with the tile
 * for (const m of [2, 4, 6]) print(`F(${m}, 3):`, winogradRelativeError(m, 3, stream(m)))
 */
export function winogradRelativeError(
  m: number,
  r: number,
  s: Stream,
  trials = 16,
): { winograd: number; direct: number } {
  const t = winogradTransforms(m, r)
  const n = t.n
  let winograd = 0
  let direct = 0
  const matrix = (rows: number, values: Float64Array) =>
    Array.from({ length: rows }, (_, i) => Array.from({ length: rows }, (_, j) => 2 * values[i * rows + j] - 1))
  for (let trial = 0; trial < trials; trial++) {
    const d = matrix(n, units(s, n * n))
    const g = matrix(r, units(s, r * r))
    const truth = directTile(d, g, m)
    const scale = Math.max(...truth.flat().map(Math.abs))
    const error = (Y: Rows) => Math.max(...Y.flatMap((row, i) => row.map((y, j) => Math.abs(y - truth[i][j])))) / scale
    winograd += error(winogradTile(d, g, t, 'float32'))
    direct += error(directTile(d, g, m, 'float32'))
  }
  return { winograd: winograd / trials, direct: direct / trials }
}

/**
 * The entries of a matrix outside $\{0, 1, -1\}$: each costs one multiplication per use of the matrix.
 *
 * @param A The matrix.
 * @returns How many entries are not $0$ or $\pm 1$.
 */
const constants = (A: Rows): number => A.flat().filter((v) => v !== 0 && Math.abs(v) !== 1).length

/** One convolutional layer for the cost model: stride 1, same padding, output $H \times W$. */
export type AlgorithmCostSpec = {
  /** Rows $H$ of the input and output maps. */
  height: number
  /** Columns $W$ of the input and output maps. */
  width: number
  /** Input channels $C$. */
  inChannels: number
  /** Output channels $C'$. */
  outChannels: number
  /** Kernel size $k$ ($k \times k$ taps). */
  kernel: number
  /** Images per batch $N$ (default 1): input and output transforms are per image, kernel transforms per batch. */
  batch?: number
  /** Winograd outputs per tile side $m$ (default 2). */
  tile?: number
  /** The kernels' transforms are computed once and kept (inference), so they cost nothing per batch. Default false. */
  precomputedKernels?: boolean
}

/** Real multiplications per batch for each algorithm, and the extra memory each needs, in numbers stored. */
export type AlgorithmCosts = {
  /** The direct sum. */
  direct: { multiplies: number }
  /** im2col and one matrix product: the same multiplications, and the column matrix as workspace. */
  im2col: { multiplies: number; workspace: number }
  /** FFT convolution over the zero-padded map: transforms, pointwise products, transform size and workspace. */
  fft: { multiplies: number; transforms: number; pointwise: number; size: [number, number]; workspace: number }
  /** Winograd: elementwise products, transforms, tile count, tile size $n$ and workspace. */
  winograd: {
    multiplies: number
    elementwise: number
    transforms: number
    tiles: number
    tileSize: number
    workspace: number
  }
}

/**
 * The smallest power of two at least `v` (1 for `v` below 1).
 *
 * @param v The size.
 * @returns $2^{\lceil \log_2 v \rceil}$.
 */
const nextPowerOfTwo = (v: number): number => 2 ** Math.ceil(Math.log2(Math.max(v, 1)))

/**
 * Real multiplications of one stride-1, same-padded layer by the direct sum, im2col with a matrix product, FFT
 * convolution over the whole map, and Winograd $F(m \times m, k \times k)$, with the workspace each needs (a model:
 * see the file's introduction).
 *
 * @param spec The layer, batch, Winograd tile and whether the kernel transforms are kept.
 * @returns The multiplications per batch and the workspace in stored numbers (real; a complex number counts two).
 *
 * @example A 3 x 3 layer, 64 to 64 channels, on 56 x 56
 * const c = convolutionAlgorithmCosts({ height: 56, width: 56, inChannels: 64, outChannels: 64, kernel: 3 })
 * print('direct   =', c.direct.multiplies)
 * print('fft      =', c.fft.multiplies, 'at size', c.fft.size)
 * print('winograd =', c.winograd.multiplies, ', saving on products', c.direct.multiplies / c.winograd.elementwise)
 */
export function convolutionAlgorithmCosts(spec: AlgorithmCostSpec): AlgorithmCosts {
  const { height: H, width: W, inChannels: C, outChannels: Cout, kernel: k } = spec
  const batch = spec.batch ?? 1
  const m = spec.tile ?? 2
  const kernels = spec.precomputedKernels ? 0 : 1
  const pairs = C * Cout

  const direct = batch * H * W * k * k * pairs

  // im2col: the same products, through a [C′, k²C] × [k²C, HW] matrix product; the column matrix is the workspace.
  const im2colWorkspace = batch * k * k * C * H * W

  // FFT over the whole zero-padded map: forward transforms of every input channel and kernel, pointwise products over
  // the Hermitian half of the spectrum, inverse transforms of every output channel.
  const Nh = nextPowerOfTwo(H + k - 1)
  const Nw = nextPowerOfTwo(W + k - 1)
  const P = Nh * Nw
  const half = Nh * (Math.floor(Nw / 2) + 1)
  const transformCount = batch * C + kernels * pairs + batch * Cout
  const fftTransforms = transformCount * P * Math.log2(P)
  const fftPointwise = batch * pairs * 4 * half
  const fftWorkspace = 2 * half * (batch * C + pairs + batch * Cout)

  // Winograd F(m × m, k × k): ⌈H/m⌉⌈W/m⌉ tiles of n × n inputs, n = m + k − 1.
  const t = winogradTransforms(m, k)
  const n = t.n
  const tiles = Math.ceil(H / m) * Math.ceil(W / m)
  const elementwise = batch * tiles * pairs * n * n
  const dataTransform = 2 * n * constants(t.BT)
  const kernelTransform = (k + n) * constants(t.G)
  const outputTransform = (n + m) * constants(t.AT)
  const winogradTransformCost =
    batch * tiles * C * dataTransform + kernels * pairs * kernelTransform + batch * tiles * Cout * outputTransform
  const winogradWorkspace = n * n * (batch * tiles * C + pairs + batch * tiles * Cout)

  return {
    direct: { multiplies: direct },
    im2col: { multiplies: direct, workspace: im2colWorkspace },
    fft: {
      multiplies: fftTransforms + fftPointwise,
      transforms: fftTransforms,
      pointwise: fftPointwise,
      size: [Nh, Nw],
      workspace: fftWorkspace,
    },
    winograd: {
      multiplies: elementwise + winogradTransformCost,
      elementwise,
      transforms: winogradTransformCost,
      tiles,
      tileSize: n,
      workspace: winogradWorkspace,
    },
  }
}
