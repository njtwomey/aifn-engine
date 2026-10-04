/**
 * Paired views of the same objects, for contrastive and multi-view learning (a toy CLIP): each object is seen twice,
 * once as a small colour image and once as a noisy "caption" of its attributes, so which image goes with which caption
 * is known exactly. Attribute combinations can be held out, so that a model trained on the rest can be tested on
 * combinations it never saw (zero-shot generalisation).
 */

import { child, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DatasetInfo, Key } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { checkCount, generatorRecipe, labels, matrix, type Recipe } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── The paired-views shape ───────────────────────────────────────────────────────────────────────────────────────────

/** One categorical attribute of the objects: its name and its levels, in the order of the integer codes. */
export type Attribute = { readonly name: string; readonly levels: readonly string[] }

/** One view of the objects: its name, the features of each row and, for an image, its layout. */
export type View = {
  readonly name: string
  /** One name per column. */
  readonly featureNames: readonly string[]
  /** For an image view: [channels, height, width], channel-major (column c·h·w + r·w + k is channel c, row r, col k). */
  readonly image?: readonly [number, number, number]
}

/** What is known about each object: the pairing, its attributes, and whether its combination is held out. */
export type PairTruth = {
  /** The object of each row (int32, length n): row i of `a` and row i of `b` are the same object, pair[i]. */
  readonly pair: Tensor
  /** Each attribute's code per row (int32, length n), by attribute name. */
  readonly attributes: Readonly<Record<string, Tensor>>
  /**
   * The class of each row (int32, length n): its combination of attributes, an index into `prototypes`.
   */
  readonly combination: Tensor
  /** 1 for rows whose combination is held out (int32, length n). */
  readonly heldOut: Tensor
}

/**
 * Paired views: `a` [n, dA] and `b` [n, dB] describe the same n objects row by row. `prototypes` holds the clean
 * B-side description of every combination of attributes (the class names a zero-shot classifier encodes), and `truth`
 * the pairing and attributes.
 */
export interface PairedViews {
  readonly kind: 'pairs'
  readonly a: Tensor
  readonly b: Tensor
  readonly truth: PairTruth
  /** The noise-free B-side view of every combination [K, dB], row k for combination k. */
  readonly prototypes: Tensor
  readonly meta: {
    readonly name: string
    readonly description: string
    readonly views: readonly [View, View]
    readonly attributes: readonly Attribute[]
    /** A name per combination, e.g. "large red triangle", row k of `prototypes`. */
    readonly combinationNames: readonly string[]
    /** Each combination's attribute codes [K][attributes]. */
    readonly combinationCodes: readonly (readonly number[])[]
    /** The held-out combinations, by index. */
    readonly heldOutCombinations: readonly number[]
    readonly key?: Key
    readonly recipe?: Recipe
  }
}

// ── Shapes and captions ──────────────────────────────────────────────────────────────────────────────────────────────

/** The shapes, in the order of their codes. */
export const PAIRED_SHAPES = ['circle', 'square', 'triangle', 'cross'] as const
/** The size bins. */
export const PAIRED_SIZES = ['small', 'large'] as const
/** The colours. */
export const PAIRED_COLOURS = ['red', 'green', 'blue'] as const
/**
 * The RGB value of each colour, in [0, 1]. Each colour lights every channel a little, so a shape leaves its outline in
 * all three channels whatever its colour, as in a real image.
 */
export const PAIRED_RGB: readonly (readonly [number, number, number])[] = [
  [0.9, 0.25, 0.2],
  [0.2, 0.75, 0.3],
  [0.25, 0.35, 0.95],
]

/**
 * The shape–colour pairs held out, in the order `heldOut` takes them: the first `heldOut` are never drawn for training
 * (every size of each is held out).
 */
const HOLD_ORDER: readonly (readonly [number, number])[] = [
  [2, 0], // red triangle
  [1, 2], // blue square
  [0, 1], // green circle
  [3, 0], // red cross
  [2, 2], // blue triangle
  [3, 1], // green cross
]

/** Options of `pairedShapes`. */
export type PairedShapesOptions = {
  /** Objects (default 600). */
  n?: number
  /** Image side in pixels (default 12). */
  size?: number
  /** Standard deviation of the Gaussian pixel noise (default 0.05). */
  pixelNoise?: number
  /** Standard deviation of the Gaussian noise on every caption entry (default 0.15). */
  captionNoise?: number
  /** How much position and rotation vary, from 0 (centred, upright) to 1 (anywhere that fits, any angle); default 0.3. */
  jitter?: number
  /** How many shape–colour combinations to hold out (0 … 6, default 2: red triangles and blue squares). */
  heldOut?: number
  /**
   * Which objects to draw: `seen` (default) only combinations that are not held out, `heldOut` only held-out ones,
   * `all` every combination with equal probability.
   */
  include?: 'seen' | 'heldOut' | 'all'
}

/** Whether point (x, y), relative to the shape's centre and in its rotated frame, is inside the shape of radius r. */
function inside(shape: number, x: number, y: number, r: number, theta: number): boolean {
  const [c, s] = [Math.cos(theta), Math.sin(theta)]
  const u = c * x + s * y
  const v = -s * x + c * y
  switch (shape) {
    case 0:
      return x * x + y * y <= r * r
    case 1:
      return Math.max(Math.abs(u), Math.abs(v)) <= 0.78 * r
    case 2: {
      // An equilateral triangle with circumradius r: inside every edge, at distance r/2 from the centre.
      for (let k = 0; k < 3; k++) {
        const a = theta + Math.PI / 2 + (2 * Math.PI * k) / 3
        if (-(Math.cos(a) * x + Math.sin(a) * y) > r / 2) return false
      }
      return true
    }
    default: {
      const w = 0.3 * r
      return (Math.abs(u) <= w && Math.abs(v) <= r) || (Math.abs(v) <= w && Math.abs(u) <= r)
    }
  }
}

/** Every combination of shape, size and colour, in code order (shape slowest, colour fastest). */
function combinations(): number[][] {
  const out: number[][] = []
  for (let sh = 0; sh < PAIRED_SHAPES.length; sh++)
    for (let sz = 0; sz < PAIRED_SIZES.length; sz++)
      for (let co = 0; co < PAIRED_COLOURS.length; co++) out.push([sh, sz, co])
  return out
}

/**
 * Paired shapes: each object is a shape (circle, square, triangle or cross), a size (small or large) and a colour
 * (red, green or blue), drawn at a random position and rotation. View `a` is the object rendered as a size × size
 * colour image (three channels: the shape's anti-aliased coverage times its colour's RGB value, plus Gaussian pixel
 * noise); view `b` is its caption, the one-hot codes of shape, size and colour (4 + 2 + 3 = 9 entries) plus Gaussian
 * noise. Position, exact size and rotation appear only in the image. The first `heldOut` shape–colour pairs (red
 * triangles, blue squares, …) are left out of `include: 'seen'` data and are the only ones in `include: 'heldOut'`.
 */
export function pairedShapes(s: Stream, options: PairedShapesOptions = {}): PairedViews {
  const {
    n = 600,
    size = 12,
    pixelNoise = 0.05,
    captionNoise = 0.15,
    jitter = 0.3,
    heldOut = 2,
    include = 'seen',
  } = options
  checkCount(n, 'pairedShapes')
  if (!(Number.isInteger(size) && size >= 4))
    throw new DomainError('pairedShapes', `pairedShapes: size must be an integer ≥ 4`)
  if (!(Number.isInteger(heldOut) && heldOut >= 0 && heldOut <= HOLD_ORDER.length))
    throw new DomainError('pairedShapes', `pairedShapes: heldOut must be an integer in 0 … ${HOLD_ORDER.length}`)
  const combos = combinations()
  const held = HOLD_ORDER.slice(0, heldOut)
  const isHeld = (c: readonly number[]) => held.some(([sh, co]) => c[0] === sh && c[2] === co)
  const heldIdx = combos.flatMap((c, k) => (isHeld(c) ? [k] : []))
  const pool = combos
    .map((_, k) => k)
    .filter((k) => (include === 'all' ? true : include === 'heldOut' ? isHeld(combos[k]) : !isHeld(combos[k])))
  if (pool.length === 0) throw new DomainError('pairedShapes', 'pairedShapes: no combinations to draw (heldOut is 0)')

  const C = PAIRED_COLOURS.length
  const dA = C * size * size
  const dB = PAIRED_SHAPES.length + PAIRED_SIZES.length + C
  const a = new Float64Array(n * dA)
  const b = new Float64Array(n * dB)
  const shape = new Int32Array(n)
  const sizeBin = new Int32Array(n)
  const colour = new Int32Array(n)
  const combination = new Int32Array(n)
  const heldOutRow = new Int32Array(n)
  // 3 × 3 subpixel samples per pixel give an anti-aliased coverage.
  const sub = [-1 / 3, 0, 1 / 3]
  const px = 2 / size
  for (let i = 0; i < n; i++) {
    const r = child(s, 'object', i)
    const k = pool[Math.min(pool.length - 1, Math.floor(uniform(r) * pool.length))]
    const [sh, sz, co] = combos[k]
    const radius = sz === 0 ? 0.45 + 0.1 * uniform(r) : 0.75 + 0.15 * uniform(r)
    // Position and rotation vary by up to `jitter` of their full range (the room left inside the image, a full turn).
    const room = jitter * Math.max(0, 0.95 - radius)
    const cx = room * (2 * uniform(r) - 1)
    const cy = room * (2 * uniform(r) - 1)
    const theta = jitter * Math.PI * (2 * uniform(r) - 1)
    const noise = child(r, 'noise')
    for (let row = 0; row < size; row++)
      for (let col = 0; col < size; col++) {
        // Pixel centres on [−1, 1]², row 0 at the top.
        const x0 = -1 + (col + 0.5) * px
        const y0 = 1 - (row + 0.5) * px
        let hits = 0
        for (const du of sub)
          for (const dv of sub) if (inside(sh, x0 + du * px - cx, y0 + dv * px - cy, radius, theta)) hits++
        for (let c = 0; c < C; c++) {
          const v = (PAIRED_RGB[co][c] * hits) / 9
          a[i * dA + c * size * size + row * size + col] = pixelNoise > 0 ? v + pixelNoise * normal(noise) : v
        }
      }
    const caption = child(r, 'caption')
    const hot = [sh, PAIRED_SHAPES.length + sz, PAIRED_SHAPES.length + PAIRED_SIZES.length + co]
    for (let j = 0; j < dB; j++)
      b[i * dB + j] = (hot.includes(j) ? 1 : 0) + (captionNoise > 0 ? captionNoise * normal(caption) : 0)
    shape[i] = sh
    sizeBin[i] = sz
    colour[i] = co
    combination[i] = k
    heldOutRow[i] = isHeld(combos[k]) ? 1 : 0
  }

  const prototypes = new Float64Array(combos.length * dB)
  combos.forEach(([sh, sz, co], k) => {
    prototypes[k * dB + sh] = 1
    prototypes[k * dB + PAIRED_SHAPES.length + sz] = 1
    prototypes[k * dB + PAIRED_SHAPES.length + PAIRED_SIZES.length + co] = 1
  })
  const pixelNames: string[] = []
  for (const c of PAIRED_COLOURS)
    for (let row = 0; row < size; row++) for (let col = 0; col < size; col++) pixelNames.push(`${c} ${row},${col}`)
  const heldNames = held.map(([sh, co]) => `${PAIRED_COLOURS[co]} ${PAIRED_SHAPES[sh]}s`)
  return {
    kind: 'pairs',
    a: matrix(a, n, dA),
    b: matrix(b, n, dB),
    truth: {
      pair: labels(Int32Array.from({ length: n }, (_, i) => i)),
      attributes: { shape: labels(shape), size: labels(sizeBin), colour: labels(colour) },
      combination: labels(combination),
      heldOut: labels(heldOutRow),
    },
    prototypes: fromData(prototypes, [combos.length, dB]),
    meta: {
      name: 'paired shapes',
      description:
        `${n} shapes seen twice: as a ${size} × ${size} colour image (pixel noise sd ${pixelNoise}) and as a caption ` +
        `of shape, size and colour one-hots (noise sd ${captionNoise}); ` +
        (heldNames.length ? `${heldNames.join(' and ')} held out (${include}).` : 'no combination held out.'),
      views: [
        { name: 'image', featureNames: pixelNames, image: [C, size, size] },
        {
          name: 'caption',
          featureNames: [...PAIRED_SHAPES, ...PAIRED_SIZES, ...PAIRED_COLOURS],
        },
      ],
      attributes: [
        { name: 'shape', levels: PAIRED_SHAPES },
        { name: 'size', levels: PAIRED_SIZES },
        { name: 'colour', levels: PAIRED_COLOURS },
      ],
      combinationNames: combos.map(([sh, sz, co]) => `${PAIRED_SIZES[sz]} ${PAIRED_COLOURS[co]} ${PAIRED_SHAPES[sh]}`),
      combinationCodes: combos,
      heldOutCombinations: heldIdx,
      key: s.key,
      recipe: generatorRecipe('pairedShapes', s.key, { n, size, pixelNoise, captionNoise, jitter, heldOut, include }),
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'pairedShapes',
    name: 'Paired shapes (image and caption)',
    summary:
      'Coloured shapes seen as a small image and as a noisy attribute caption, paired row by row, with held-out shape–colour combinations.',
    task: 'embedding',
    output: 'pairs',
    knobs: space({
      n: int(1, 5000, { default: 600 }),
      size: int(4, 32, { default: 12 }),
      pixelNoise: real(0, 1, { default: 0.05 }),
      captionNoise: real(0, 1, { default: 0.15 }),
      jitter: real(0, 1, { default: 0.3 }),
      heldOut: int(0, 6, { default: 2 }),
      include: oneOf(['seen', 'heldOut', 'all']),
    }),
    truth: true,
    random: true,
    notes: ['clip', 'contrastive-learning'],
  },
  pairedShapes,
)
