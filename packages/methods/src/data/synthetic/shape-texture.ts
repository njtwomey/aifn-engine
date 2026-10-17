/**
 * Small greyscale images of filled shapes, each shape's interior painted with a texture, for classification
 * experiments that need two separable cues: the outline (shape) and the local pattern (texture).
 *
 * A shape is a disc, square, cross, bar, triangle or ring of a given centre and radius; a texture is solid, horizontal
 * or vertical stripes, a checkerboard or a grid of dots, all of period 2 or 3 pixels, so a $3 \times 3$ filter detects
 * the texture anywhere while the shape needs a receptive field of the object's size. `shapeTextureDataset` pairs
 * textures with shapes in one of three ways: matched (texture $i$ always with shape $i$, so both cues predict the
 * label), random (texture independent of shape, as in Stylized-ImageNet) or conflict (texture $i$ never with shape
 * $i$, the cue-conflict test of Geirhos et al., 2019, "ImageNet-trained CNNs are biased towards texture", §2.2).
 * Images are row-major with row 0 at the top, values in $[0, 1]$ before noise; noise is Gaussian from a stream and
 * never clipped. Image $k$ of shape class $c$ draws from `child(s, c, k)` and its noise from that stream's child
 * `'noise'`, so the images are fixed by the stream whatever the other options.
 */

import { child, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { definer, type FunctionInfo } from 'aifn-compute/foundation/registry'

/** The outline of an object. */
export type ShapeName = 'disc' | 'square' | 'cross' | 'bar' | 'triangle' | 'ring'
/** The pattern painted inside an outline. */
export type TextureName = 'solid' | 'horizontal' | 'vertical' | 'checker' | 'dots'

/** Every outline, in order. */
export const SHAPE_NAMES: readonly ShapeName[] = ['disc', 'square', 'cross', 'bar', 'triangle', 'ring']
/** Every texture, in order. */
export const TEXTURE_NAMES: readonly TextureName[] = ['solid', 'horizontal', 'vertical', 'checker', 'dots']

/** One object in a scene: its outline, its texture, its centre and its radius. */
export type ShapeItem = {
  /** The outline. */
  shape: ShapeName
  /** The texture painted inside it. */
  texture: TextureName
  /** Centre column, in pixels from the left edge (pixel $c$ has its centre at $c + 0.5$). */
  x: number
  /** Centre row, in pixels from the top edge. */
  y: number
  /** Half the object's extent, in pixels. */
  radius: number
  /** Offset of the texture's pattern in pixels (default 0); shifts stripes and checks without changing them. */
  phase?: number
  /** The value of the texture's dark parts, in $[0, 1]$ (default 0.3); the bright parts are 1. */
  low?: number
}

/**
 * Whether the point $(d_x, d_y)$, relative to the centre in pixels with $d_y$ pointing down, lies inside a shape of
 * radius $r$. The arms of the cross and the bar are $\max(1, 0.34r)$ wide on each side; the square's half-side is
 * $0.82r$; the triangle points up; the ring's hole has radius $r/2$.
 *
 * @param shape The outline.
 * @param dx Horizontal offset from the centre, in pixels.
 * @param dy Vertical offset from the centre (down is positive).
 * @param r The radius.
 * @returns True inside the outline.
 *
 * @example The centre is inside every shape except the ring
 * for (const s of SHAPE_NAMES) print(s, insideShape(s, 0, 0, 4))
 */
export function insideShape(shape: ShapeName, dx: number, dy: number, r: number): boolean {
  const ax = Math.abs(dx)
  const ay = Math.abs(dy)
  const arm = Math.max(1, 0.34 * r)
  switch (shape) {
    case 'disc':
      return dx * dx + dy * dy <= r * r
    case 'square':
      return ax <= 0.82 * r && ay <= 0.82 * r
    case 'cross':
      return (ax <= arm && ay <= r) || (ay <= arm && ax <= r)
    case 'bar':
      return ax <= r && ay <= arm
    case 'triangle':
      // Apex up: the half-width grows linearly from 0 at the top (dy = -r) to r at the base (dy = r).
      return dy >= -r && dy <= 0.8 * r && ax <= (dy + r) / 2
    case 'ring': {
      const d2 = dx * dx + dy * dy
      return d2 <= r * r && d2 >= 0.5 * 0.5 * r * r
    }
  }
}

/**
 * The value of a texture at pixel (row, col): 1 on its bright parts, `low` on its dark parts. Stripes and checks have
 * period 2, dots period 3.
 *
 * @param texture The texture.
 * @param row The pixel's row.
 * @param col The pixel's column.
 * @param phase The pattern's offset in pixels, added to both the row and the column.
 * @param low The dark value.
 * @returns The texture's value.
 *
 * @example One row of each texture
 * for (const t of TEXTURE_NAMES) print(t, [0, 1, 2, 3, 4, 5].map((c) => textureValue(t, 0, c)))
 */
export function textureValue(texture: TextureName, row: number, col: number, phase = 0, low = 0.3): number {
  const r = row + phase
  const c = col + phase
  const mod = (a: number, n: number) => ((a % n) + n) % n
  switch (texture) {
    case 'solid':
      return 1
    case 'horizontal':
      return mod(r, 2) === 0 ? 1 : low
    case 'vertical':
      return mod(c, 2) === 0 ? 1 : low
    case 'checker':
      return mod(r + c, 2) === 0 ? 1 : low
    case 'dots':
      return mod(r, 3) === 0 && mod(c, 3) === 0 ? 1 : low
  }
}

/**
 * A `size` $\times$ `size` image of objects on a constant background; a later object paints over an earlier one.
 * With a stream and positive `noise`, Gaussian noise of that standard deviation is added to every pixel. Throws
 * `DomainError` for noise without a stream.
 *
 * @param size The image's width and height in pixels.
 * @param items The objects, painted in order.
 * @param options The background, the noise and its stream.
 * @param options.background The value outside every object.
 * @param options.noise The noise's standard deviation; 0 for none.
 * @param options.stream The stream the noise is drawn from (needed when `noise` is positive).
 * @returns The image, `size * size` values, row-major, row 0 at the top.
 *
 * @example A checkered disc and a striped bar on an 8 x 8 image
 * const img = renderShapes(8, [
 *   { shape: 'disc', texture: 'checker', x: 3, y: 3, radius: 2.5 },
 *   { shape: 'bar', texture: 'vertical', x: 5, y: 6.5, radius: 2.5 },
 * ])
 * for (let r = 0; r < 8; r++) print(Array.from(img.slice(8 * r, 8 * r + 8)).join(' '))
 */
export function renderShapes(
  size: number,
  items: readonly ShapeItem[],
  options: { background?: number; noise?: number; stream?: Stream } = {},
): Float64Array {
  const { background = 0, noise = 0, stream } = options
  const img = new Float64Array(size * size).fill(background)
  for (const it of items) {
    const r0 = Math.max(0, Math.floor(it.y - it.radius - 1))
    const r1 = Math.min(size - 1, Math.ceil(it.y + it.radius + 1))
    const c0 = Math.max(0, Math.floor(it.x - it.radius - 1))
    const c1 = Math.min(size - 1, Math.ceil(it.x + it.radius + 1))
    for (let row = r0; row <= r1; row++)
      for (let col = c0; col <= c1; col++)
        if (insideShape(it.shape, col + 0.5 - it.x, row + 0.5 - it.y, it.radius))
          img[row * size + col] = textureValue(it.texture, row, col, it.phase ?? 0, it.low ?? 0.3)
  }
  if (noise > 0) {
    if (!stream) throw new DomainError('renderShapes', 'renderShapes: noise needs a stream')
    const z = toFlat(normals(stream, size * size, 0, noise))
    for (let i = 0; i < img.length; i++) img[i] += z[i]
  }
  return img
}

/** How `shapeTextureDataset` chooses each image's texture from its shape. */
export type ShapeTexturePairing = 'matched' | 'random' | 'conflict'

/** Options of `shapeTextureDataset`. */
export type ShapeTextureOptions = {
  /** The image's width and height in pixels (default 16). */
  size?: number
  /** Images per shape (default 40). */
  perClass?: number
  /** The shape classes (default disc, square, cross). */
  shapes?: readonly ShapeName[]
  /** The textures (default horizontal, vertical, checker). */
  textures?: readonly TextureName[]
  /** How textures follow shapes (default `matched`). */
  pairing?: ShapeTexturePairing
  /** With matched pairing, the probability of redrawing an image's texture uniformly (default 0). */
  restyle?: number
  /** The range of radii in pixels (default `[3.5, 5.5]`). */
  radius?: readonly [number, number]
  /** The noise's standard deviation (default 0.05). */
  noise?: number
  /** The background value (default 0). */
  background?: number
}

/** A labelled set of shape-texture images. */
export type ShapeTextureData = {
  /** The images, `[n, 1, size, size]`. */
  x: Tensor
  /** The shape class of each image, an index into `shapes` ($n$ values). */
  shape: Int32Array
  /** The texture class of each image, an index into `textures` ($n$ values). */
  texture: Int32Array
  /** The objects drawn, one per image. */
  items: ShapeItem[]
}

/**
 * `perClass` images of each shape, one object per image at a uniform random centre that keeps it inside the frame,
 * with a uniform random radius in `radius` and a random texture phase. The texture follows `pairing`: `matched` gives
 * shape $i$ the texture $i$ (shapes and textures must be equally many); `random` draws it uniformly; `conflict` draws
 * it uniformly from the textures other than texture $i$. With `restyle` $p > 0$ (matched pairing only), each image's
 * texture is redrawn uniformly with probability $p$. Throws `DomainError` when matched or conflict pairing has
 * unequal numbers of shapes and textures, or conflict fewer than 2 textures.
 *
 * @param s The stream; image $k$ of shape $c$ draws from `child(s, c, k)`.
 * @param options The size, images per class, shapes, textures, pairing, restyle probability, radii, noise and
 *   background.
 * @returns The images, both labels, and the objects drawn.
 *
 * @example Conflict pairing: no image's texture matches its shape
 * const d = shapeTextureDataset(stream(2), { perClass: 4, pairing: 'conflict', noise: 0 })
 * print('shapes   =', d.shape)
 * print('textures =', d.texture)
 * print('x shape  =', d.x.shape)
 */
export function shapeTextureDataset(s: Stream, options: ShapeTextureOptions = {}): ShapeTextureData {
  const where = 'shapeTextureDataset'
  const {
    size = 16,
    perClass = 40,
    shapes = ['disc', 'square', 'cross'],
    textures = ['horizontal', 'vertical', 'checker'],
    pairing = 'matched',
    restyle = 0,
    radius = [3.5, 5.5],
    noise = 0.05,
    background = 0,
  } = options
  if (pairing !== 'random' && shapes.length !== textures.length)
    throw new DomainError(where, `${where}: matched and conflict pairings need as many textures as shapes`)
  if (pairing === 'conflict' && textures.length < 2) throw new DomainError(where, `${where}: conflict needs 2 textures`)
  const n = shapes.length * perClass
  const x = new Float64Array(n * size * size)
  const shapeLabel = new Int32Array(n)
  const textureLabel = new Int32Array(n)
  const items: ShapeItem[] = []
  const T = textures.length
  let i = 0
  for (let c = 0; c < shapes.length; c++)
    for (let k = 0; k < perClass; k++, i++) {
      const r = child(s, c, k)
      const u = toFlat(uniform(r, 0, 1, { shape: [6] }) as Tensor)
      const rad = radius[0] + (radius[1] - radius[0]) * u[0]
      const lo = Math.min(rad, size / 2)
      const cx = lo + (size - 2 * lo) * u[1]
      const cy = lo + (size - 2 * lo) * u[2]
      let t: number
      if (pairing === 'matched') t = u[3] < restyle ? Math.min(T - 1, Math.floor(u[4] * T)) : c
      else if (pairing === 'random') t = Math.min(T - 1, Math.floor(u[4] * T))
      else t = (c + 1 + Math.min(T - 2, Math.floor(u[4] * (T - 1)))) % T
      const item: ShapeItem = {
        shape: shapes[c],
        texture: textures[t],
        x: cx,
        y: cy,
        radius: rad,
        phase: Math.floor(u[5] * 6),
      }
      x.set(renderShapes(size, [item], { background, noise, stream: child(r, 'noise') }), i * size * size)
      shapeLabel[i] = c
      textureLabel[i] = t
      items.push(item)
    }
  return { x: fromData(x, [n, 1, size, size]), shape: shapeLabel, texture: textureLabel, items }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'shapeTextureDataset',
    name: 'Shape and texture images',
    summary: 'Textured shapes whose texture matches, ignores or conflicts with the shape: two cues for a small CNN.',
    role: 'simulation',
    random: true,
    notes: ['feature-visualisation', 'class-activation-mapping', 'shape-and-texture-bias'],
    cite: ['geirhos2019'],
  },
  shapeTextureDataset,
)
fn(
  {
    key: 'renderShapes',
    name: 'Textured shapes image',
    summary: 'An image of textured shapes on a constant background, with optional Gaussian noise.',
    role: 'simulation',
    notes: ['class-activation-mapping'],
  },
  renderShapes,
)
