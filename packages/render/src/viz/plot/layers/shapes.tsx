/**
 * Pictures on a Plot's axes: `Pixels`, an image in pixel coordinates (RGB, or one value per pixel on a colour scale),
 * and `Shapes`, filled vector outlines with holes (glyphs, regions, the mass under a curve). They replaced compat's
 * `ImagePlot` and `GlyphPlot`, whose lines, guides and handles are now ordinary layers in the same Plot.
 */
import { chrome, interpolateColors, scaleStops, seriesColor, type Mode } from '../../../design/palette'
import type { Range } from '../../viewport'
import { defineLayer, extentOf, type CommonProps } from '../layer'
import type { CustomApi } from './marks'
import { stretchedImage } from './raster'

// ── Pixels ───────────────────────────────────────────────────────────────────────────────────────────────────────────

export type PixelsProps = CommonProps & {
  /** Image size in pixels. Pixel (row r, column c) is centred at (x, y) = (c, r): use `useAxis({ inverse: true })` on y so row 0 is at the top. */
  width: number
  height: number
  /** Colour pixels, row-major RGB in [0, 1]: `rgb[3 * (r * width + c) + k]` for channel k. */
  rgb?: ArrayLike<number>
  /** Or one value per pixel, row-major, on the `sequential` (default) or `diverging` scale over `range`. */
  values?: ArrayLike<number>
  scale?: 'sequential' | 'diverging'
  range?: Range
}

/** The last canvas per pixel array, keyed by what colours it, so a zoom or a patched overlay reuses it. */
const pictures = new WeakMap<object, { key: string; canvas: HTMLCanvasElement | null }>()

/**
 * One canvas per layer and image size, redrawn in place: an animation that passes a new pixel array each frame
 * (a Player over frames) costs one `putImageData`, not a new canvas, an `ImageData` and a colour table per frame.
 * `owner` is the array last drawn into it; its cache entry is dropped when the canvas is redrawn, so stepping back to
 * an earlier frame draws that frame again rather than showing the newer pixels.
 */
const surfaces = new Map<string, { canvas: HTMLCanvasElement; image: ImageData; owner: object | null }>()

/** 256 colours along a scale as RGB bytes, built once per scale and theme. */
const tables = new Map<string, Uint8Array>()
function colourTable(scale: 'sequential' | 'diverging', mode: Mode): Uint8Array {
  const key = `${scale}|${mode}`
  let table = tables.get(key)
  if (!table) {
    const stops = scaleStops(scale, mode)
    table = new Uint8Array(256 * 3)
    for (let i = 0; i < 256; i++)
      table.set((interpolateColors(stops, i / 255).match(/\d+/g) ?? []).slice(0, 3).map(Number), 3 * i)
    tables.set(key, table)
  }
  return table
}

function picture(p: PixelsProps, mode: Mode, id: string): HTMLCanvasElement | null {
  const data = p.rgb ?? p.values
  if (!data || typeof document === 'undefined') return null
  const key = `${p.width}x${p.height}|${p.rgb ? 'rgb' : `${p.scale ?? 'sequential'}|${p.range}`}|${mode}`
  const hit = pictures.get(data as object)
  if (hit?.key === key) return hit.canvas
  const size = `${id}|${p.width}x${p.height}`
  let surface = surfaces.get(size)
  if (!surface) {
    const canvas = document.createElement('canvas')
    canvas.width = p.width
    canvas.height = p.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    surface = { canvas, image: ctx.createImageData(p.width, p.height), owner: null }
    surfaces.set(size, surface)
  }
  if (surface.owner) pictures.delete(surface.owner)
  const px = surface.image.data
  const n = p.width * p.height
  if (p.rgb) {
    const rgb = p.rgb
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) px[4 * i + k] = 255 * rgb[3 * i + k] // the clamped array rounds and clamps
      px[4 * i + 3] = 255
    }
  } else {
    const values = p.values!
    const [lo, hi] = p.range ?? (extentOf(values) as Range | undefined) ?? [0, 1]
    const table = colourTable(p.scale ?? 'sequential', mode)
    const scale = hi > lo ? 255 / (hi - lo) : 0
    for (let i = 0; i < n; i++) {
      const t = scale ? Math.round((values[i] - lo) * scale) : 128
      const c = 3 * (t < 0 ? 0 : t > 255 ? 255 : t)
      px[4 * i] = table[c]
      px[4 * i + 1] = table[c + 1]
      px[4 * i + 2] = table[c + 2]
      px[4 * i + 3] = 255
    }
  }
  surface.canvas.getContext('2d')!.putImageData(surface.image, 0, 0)
  surface.owner = data as object
  pictures.set(data as object, { key, canvas: surface.canvas })
  return surface.canvas
}

/** An image whose pixels are unit cells centred on integer (column, row) coordinates; drawn without smoothing. */
export const Pixels = defineLayer<PixelsProps>({
  kind: 'Pixels',
  slotted: () => false,
  legend: () => [],
  covers: true,
  canvas: () => true,
  extent: (p) => ({ x: [-0.5, p.width - 0.5], y: [-0.5, p.height - 0.5], tight: true }),
  build: (p, ctx) => {
    const image = picture(p, ctx.mode, ctx.id)
    return {
      series: [
        {
          id: `${ctx.id}:pixels`,
          name: '__pixels',
          type: 'custom',
          silent: true,
          clip: true,
          data: [[-0.5, -0.5]],
          encode: { x: 0, y: 1 },
          tooltip: { show: false },
          renderItem: (_: unknown, api: CustomApi) => {
            if (!image) return null
            // Corners of pixel (0, 0) and (w − 1, h − 1): an inverted axis flips the stretch, so row 0 lands on top.
            const [x0, y0] = api.coord([-0.5, -0.5])
            const [x1, y1] = api.coord([p.width - 0.5, p.height - 0.5])
            const el = stretchedImage(image, x0, y0, x1, y1)
            return { ...el, style: { ...el.style, imageSmoothing: false } }
          },
          z: 0,
        },
      ],
    }
  },
})

// ── Shapes ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One filled shape: closed contours of [x, y] points, filled with the nonzero rule (an opposite winding is a hole). */
export type FilledShape = {
  contours: readonly (readonly (readonly [number, number])[])[]
  /** 'ink' (default), 'muted', or a categorical palette slot. */
  tone?: 'ink' | 'muted' | number
  /** Fill opacity, e.g. a ghost of a reference shape drawn behind. */
  opacity?: number
}

export type ShapesProps = CommonProps & { shapes: readonly FilledShape[] }

/** Filled vector shapes as SVG paths, so counters stay open and edges sharp. Pair with equal axes for glyphs. */
export const Shapes = defineLayer<ShapesProps>({
  kind: 'Shapes',
  slotted: () => false,
  legend: (p) => (p.name ? [p.name] : []),
  extent: (p) => {
    const pts = p.shapes.flatMap((s) => s.contours.flat())
    return { x: extentOf(pts.map((q) => q[0])), y: extentOf(pts.map((q) => q[1])) }
  },
  build: (p, ctx) => {
    const c = chrome(ctx.mode)
    const fill = (tone: FilledShape['tone']) =>
      tone === undefined || tone === 'ink' ? c.ink : tone === 'muted' ? c.muted : seriesColor(ctx.mode, tone)
    return {
      series: [
        {
          id: `${ctx.id}:shapes`,
          name: p.name ?? '__shapes',
          type: 'custom',
          silent: true,
          clip: true,
          progressive: 0,
          data: p.shapes.map((_, i) => [i]),
          tooltip: { show: false },
          renderItem: (_: unknown, api: CustomApi) => {
            const s = p.shapes[api.value(0)]
            const d = s.contours
              .map(
                (contour) =>
                  `M${contour
                    .map((q) =>
                      api
                        .coord([q[0], q[1]])
                        .map((v) => v.toFixed(2))
                        .join(' '),
                    )
                    .join('L')}Z`,
              )
              .join('')
            return {
              type: 'path',
              shape: { pathData: d },
              style: { fill: fill(s.tone), opacity: s.opacity ?? 1, stroke: 'none' },
            }
          },
          z: 2,
        },
      ],
    }
  },
})
