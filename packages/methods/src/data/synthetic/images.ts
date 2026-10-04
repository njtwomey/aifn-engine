/**
 * Test images (greyscale, row-major [height, width] tensors with values in [0, 1], row 0 at the top) and small binary
 * pattern sets: a checkerboard, gradients, a shapes image, a 5 × 7 digit font with noisy copies, and bars and stripes.
 */

import { normal, type Stream, child, uniform } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkCount, labels, matrix, type Dataset } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A `size`-pixel checkerboard of `tile`-pixel squares, `low` and `high` valued, starting with `high` at the top left. */
export function checkerboardImage({
  size = 64,
  tile = 8,
  low = 0,
  high = 1,
}: { size?: number; tile?: number; low?: number; high?: number } = {}): Tensor {
  const out = new Float64Array(size * size)
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) out[r * size + c] = (Math.floor(r / tile) + Math.floor(c / tile)) % 2 ? low : high
  return matrix(out, size, size)
}

/**
 * A linear ramp from 0 to 1 across the image in the direction `angle` (radians, 0 = left to right, π/2 = bottom to
 * top); `kind: 'radial'` ramps from 0 at the centre to 1 at the corners.
 */
export function gradientImage({
  size = 64,
  angle = 0,
  kind = 'linear',
}: { size?: number; angle?: number; kind?: 'linear' | 'radial' } = {}): Tensor {
  const out = new Float64Array(size * size)
  const c = (size - 1) / 2
  const [dx, dy] = [Math.cos(angle), Math.sin(angle)]
  // The projection of the corners onto the direction bounds the ramp, so values span exactly [0, 1].
  const reach = c * (Math.abs(dx) + Math.abs(dy)) || 1
  for (let r = 0; r < size; r++)
    for (let col = 0; col < size; col++) {
      const x = col - c
      const y = c - r
      out[r * size + col] =
        kind === 'radial' ? Math.hypot(x, y) / (Math.SQRT2 * c || 1) : 0.5 + (x * dx + y * dy) / (2 * reach)
    }
  return matrix(out, size, size)
}

/**
 * A shapes test image (the site's image-processing notes): on a dark background (0.15), a bright rectangle, a grey
 * disk, a triangle and a checkerboard patch, so straight edges, curved edges, corners and texture all appear. With a
 * stream and `noise`, Gaussian noise of that standard deviation is added.
 */
export function shapesImage(options: { size?: number; noise?: number; stream?: Stream } = {}): Tensor {
  const { size = 64, noise = 0, stream } = options
  if (noise > 0 && !stream) throw new DomainError('shapesImage', 'shapesImage: noise needs a stream')
  const k = size / 64
  const out = new Float64Array(size * size)
  for (let row = 0; row < size; row++)
    for (let c = 0; c < size; c++) {
      // Drawn in the original's 64-pixel coordinates with row 0 at the bottom, then flipped so row 0 is the top.
      const r = (size - 1 - row) / k
      const x = c / k
      let v = 0.15
      if (r >= 8 && r <= 27 && x >= 6 && x <= 25) v = 0.9
      if ((r - 46) ** 2 + (x - 16) ** 2 <= 11 ** 2) v = 0.6
      if (r >= 6 && r <= 28 && Math.abs(x - 48) <= (28 - r) * 0.55) v = 0.7
      if (r >= 36 && r <= 59 && x >= 36 && x <= 59)
        v = (Math.floor((r - 36) / 6) + Math.floor((x - 36) / 6)) % 2 ? 0.9 : 0.1
      out[row * size + c] = noise > 0 ? v + noise * normal(stream!) : v
    }
  return matrix(out, size, size)
}

/** A geometric test scene and the ground truth it was drawn from. */
export interface GeometricScene {
  /** The greyscale image [size, size] (row 0 at the top). */
  image: Tensor
  /** 1 on pixels whose 4-neighbourhood holds a different clean value (the true edges), else 0. */
  edges: Tensor
  /** The square's four corners (row, column; sub-pixel). */
  corners: { row: number; col: number }[]
  /** The two drawn lines, x cos θ + y sin θ = ρ with x the column and y the row. */
  lines: { angle: number; distance: number }[]
  /** The disc. */
  circles: { row: number; col: number; radius: number }[]
}

/**
 * A scene with known geometry for testing edge, corner, line and circle detectors: on a background of 0.2, a square
 * of 0.85 rotated by `rotation` radians, a disc of 0.55 and two bright one-pixel lines; with a stream and `noise`,
 * Gaussian noise of that standard deviation is added. The clean image's edges, the square's corners, the lines'
 * normal-form parameters and the disc are returned with it. The noise is drawn from `stream`.
 */
export function geometricScene(
  stream: Stream,
  options: { size?: number; noise?: number; rotation?: number } = {},
): GeometricScene {
  const { size = 96, noise = 0, rotation = 0.3 } = options
  const k = size / 96
  const sq = { row: 32 * k, col: 30 * k, half: 15 * k }
  const disc = { row: 66 * k, col: 64 * k, radius: 16 * k }
  const lines = [
    { angle: -0.35, distance: 0 },
    { angle: 1.25, distance: 0 },
  ]
  // Each line passes through a chosen point: ρ = x cos θ + y sin θ there.
  const through = [
    [78 * k, 14 * k],
    [12 * k, 70 * k],
  ]
  lines.forEach((l, i) => (l.distance = through[i][1] * Math.cos(l.angle) + through[i][0] * Math.sin(l.angle)))
  const [c, s] = [Math.cos(rotation), Math.sin(rotation)]
  const clean = new Float64Array(size * size)
  for (let r = 0; r < size; r++)
    for (let x = 0; x < size; x++) {
      let v = 0.2
      const dr = r - sq.row
      const dc = x - sq.col
      if (Math.abs(c * dc + s * dr) <= sq.half && Math.abs(-s * dc + c * dr) <= sq.half) v = 0.85
      if ((r - disc.row) ** 2 + (x - disc.col) ** 2 <= disc.radius ** 2) v = 0.55
      for (const l of lines) if (Math.abs(x * Math.cos(l.angle) + r * Math.sin(l.angle) - l.distance) < 0.5) v = 1
      clean[r * size + x] = v
    }
  const edges = new Float64Array(size * size)
  for (let r = 0; r < size; r++)
    for (let x = 0; x < size; x++) {
      const v = clean[r * size + x]
      const differs = [
        [r - 1, x],
        [r + 1, x],
        [r, x - 1],
        [r, x + 1],
      ].some(([a, b]) => a >= 0 && a < size && b >= 0 && b < size && clean[a * size + b] !== v)
      edges[r * size + x] = differs ? 1 : 0
    }
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ].map(([u, w]) => ({
    col: sq.col + sq.half * (c * u - s * w),
    row: sq.row + sq.half * (s * u + c * w),
  }))
  const image = noise > 0 ? clean.map((v) => v + noise * normal(stream)) : clean
  return { image: matrix(image, size, size), edges: matrix(edges, size, size), corners, lines, circles: [disc] }
}

// A 5 × 7 bitmap font for the digits, row by row from the top (the classic HD44780 LCD glyphs).
const FONT: readonly string[][] = [
  ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  ['11111', '00010', '00100', '00010', '00001', '10001', '01110'],
  ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
]

/** The ten digit glyphs as a [10, 7, 5] tensor of 0s and 1s (row 0 at the top). */
export function digitGlyphs(): Tensor {
  const out = new Float64Array(10 * 35)
  FONT.forEach((glyph, d) => glyph.forEach((row, r) => [...row].forEach((ch, c) => (out[d * 35 + r * 5 + c] = +ch))))
  return fromData(out, [10, 7, 5])
}

/**
 * Noisy digits: `perClass` copies of each 5 × 7 glyph, each pixel flipped with probability `flip` and then blurred by
 * Gaussian noise of standard deviation `noise`. x is n × 35 (rows of the image concatenated), y the digit.
 */
export function digits(s: Stream, options: { perClass?: number; flip?: number; noise?: number } = {}): Dataset {
  const { perClass = 20, flip = 0.05, noise = 0.1 } = options
  checkCount(perClass, 'digits')
  const glyphs = digitGlyphs().data
  const n = 10 * perClass
  const x = new Float64Array(n * 35)
  const y = new Int32Array(n)
  for (let d = 0; d < 10; d++)
    for (let k = 0; k < perClass; k++) {
      const row = d * perClass + k
      const r = child(s, 'digit', d, k)
      for (let p = 0; p < 35; p++) {
        let v = glyphs[d * 35 + p]
        if (uniform(r) < flip) v = 1 - v
        x[row * 35 + p] = noise > 0 ? v + noise * normal(r) : v
      }
      y[row] = d
    }
  return {
    kind: 'dataset',
    x: matrix(x, n, 35),
    y: labels(y),
    meta: {
      name: 'digits',
      description: `${perClass} noisy copies of each 5 × 7 digit glyph (pixel flip probability ${flip}, noise sd ${noise}).`,
      task: 'images',
      featureNames: Array.from({ length: 35 }, (_, p) => `pixel ${Math.floor(p / 5)},${p % 5}`),
      labelNames: Array.from({ length: 10 }, (_, d) => String(d)),
      key: s.key,
    },
  }
}

/**
 * Every bars-and-stripes pattern on a size × size grid (MacKay, 2003, "Information Theory, Inference, and Learning
 * Algorithms", §43): each subset of columns switched on (bars) or of rows (stripes), with the all-off and all-on
 * patterns counted once. Returns a [2^{size+1} − 2, size·size] tensor of 0s and 1s.
 */
export function barsAndStripes({ size = 4 }: { size?: number } = {}): Tensor {
  const patterns: number[][] = []
  const seen = new Set<string>()
  for (const kind of ['bars', 'stripes'])
    for (let mask = 0; mask < 2 ** size; mask++) {
      const p = Array.from({ length: size * size }, (_, k) => {
        const [r, c] = [Math.floor(k / size), k % size]
        return (mask >> (kind === 'bars' ? c : r)) & 1
      })
      const key = p.join('')
      if (seen.has(key)) continue
      seen.add(key)
      patterns.push(p)
    }
  return matrix(Float64Array.from(patterns.flat()), patterns.length, size * size)
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'checkerboardImage',
    name: 'Checkerboard image',
    summary: 'A greyscale checkerboard of square tiles.',
    task: 'images',
    output: 'image',
    knobs: space({
      size: int(4, 512, { default: 64 }),
      tile: int(1, 64, { default: 8 }),
      low: real(0, 1, { default: 0 }),
      high: real(0, 1, { default: 1 }),
    }),
    truth: false,
    random: false,
  },
  checkerboardImage,
)

dataset(
  {
    key: 'gradientImage',
    name: 'Gradient image',
    summary: 'A linear ramp in a direction, or a radial one from the centre.',
    task: 'images',
    output: 'image',
    knobs: space({
      size: int(4, 512, { default: 64 }),
      angle: real(-Math.PI, Math.PI, { default: 0 }),
      kind: oneOf(['linear', 'radial']),
    }),
    truth: false,
    random: false,
  },
  gradientImage,
)

dataset(
  {
    key: 'shapesImage',
    name: 'Shapes image',
    summary: 'A rectangle, a disk, a triangle and a checkerboard patch on a dark background.',
    task: 'images',
    output: 'image',
    knobs: space({ size: int(16, 512, { default: 64 }) }),
    truth: false,
    random: false,
  },
  shapesImage,
)

dataset(
  {
    key: 'geometricScene',
    name: 'Geometric scene',
    summary: 'A rotated square, a disc and two lines, with their true edges, corners, lines and circle.',
    task: 'images',
    output: 'scene',
    knobs: space({
      size: int(32, 512, { default: 96 }),
      noise: real(0, 0.5, { default: 0 }),
      rotation: real(-Math.PI / 4, Math.PI / 4, { default: 0.3 }),
    }),
    truth: true,
    random: true,
    notes: ['edge-detection', 'feature-detection-and-descriptors', 'hough-transform'],
  },
  geometricScene,
)

dataset(
  {
    key: 'digitGlyphs',
    name: 'Digit glyphs',
    summary: 'The ten digits of a 5 × 7 pixel font.',
    task: 'images',
    output: 'patterns',
    knobs: space({}),
    truth: false,
    random: false,
  },
  digitGlyphs,
)

dataset(
  {
    key: 'digits',
    name: 'Noisy digits',
    summary: 'Copies of the 5 × 7 digit glyphs with flipped pixels and Gaussian noise, labelled by digit.',
    task: 'images',
    output: 'dataset',
    knobs: space({
      perClass: int(1, 200, { default: 20 }),
      flip: real(0, 0.5, { default: 0.05 }),
      noise: real(0, 1, { default: 0.1 }),
    }),
    truth: false,
    random: true,
  },
  digits,
)

dataset(
  {
    key: 'barsAndStripes',
    name: 'Bars and stripes',
    summary: 'Every bars-and-stripes pattern on a square grid.',
    task: 'images',
    output: 'patterns',
    knobs: space({ size: int(2, 6, { default: 4 }) }),
    truth: false,
    random: false,
  },
  barsAndStripes,
)
