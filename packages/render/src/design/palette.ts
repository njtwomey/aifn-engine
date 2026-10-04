/**
 * Data colours, owned by the lab. Seeded from the site's design/palette.json (copied, not imported): when the site's
 * palette changes, copy the new values here.
 *
 * Rules:
 * - Categorical slots are assigned in fixed order by entity, never by rank, never cycled.
 * - Scatter-type charts use at most three categorical slots unless marker shape also encodes the group.
 * - Continuous scales are theme-aware and start from the surface: zero or the least value (the "empty" part of a
 *   figure) is drawn in the surface colour, so it is the quietest part of the figure in both themes, and strength grows
 *   towards the strong hue. Sequential: one hue (blue), surface → dark blue in light mode, surface → bright saturated
 *   blue in dark mode, lightness monotone in each. Signed values use the diverging scale, the exception to that rule:
 *   a pale neutral at zero in both themes (never a dark band), darkening to strong blue below and strong red above.
 *   The categorical slots are for categories only, never ramps.
 * - Highlighted marks (centroids, optima, handles) use the ink colour, not a palette slot.
 */

export type Mode = 'light' | 'dark'

export const palette = {
  categorical: {
    light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
  },
  // Sequential: from the surface to the strong hue. Light mode darkens, dark mode brightens, so high values always
  // stand out from the surface. The first stop is the theme's surface colour.
  sequential: {
    light: ['#fcfcfb', '#e3eefb', '#c4dbf7', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
    dark: ['#1a1a19', '#182232', '#172d4d', '#163a6d', '#174a94', '#1c5bbb', '#2a6fde', '#3f87f5', '#62a2ff'],
  },
  // Diverging: two ramps out from a pale neutral midpoint at zero, darkening towards a strong blue (below) and red
  // (above), the same in both themes. The owner asked for no dark band at zero: a dark-mode midpoint at the surface
  // colour reads as a black stripe between the two hues, so the midpoint stays pale on the dark surface too.
  diverging: {
    negative: {
      light: ['#0d366b', '#1c5cab', '#3987e5', '#86b6ef', '#cde2fb'],
      dark: ['#0d366b', '#1c5cab', '#3987e5', '#86b6ef', '#cde2fb'],
    },
    midpoint: { light: '#f0efec', dark: '#f0efec' },
    positive: {
      light: ['#fbd5d2', '#f09b98', '#e34948', '#b3302f', '#7a1f1e'],
      dark: ['#fbd5d2', '#f09b98', '#e34948', '#b3302f', '#7a1f1e'],
    },
  },
  chrome: {
    light: {
      surface: '#fcfcfb',
      ink: '#0b0b0b',
      inkSecondary: '#52514e',
      muted: '#898781',
      grid: '#e1e0d9',
      axis: '#c3c2b7',
      // Outcome colours, matching lab.css's --destructive and --success: a failed and a successful episode.
      destructive: '#e7000b',
      success: '#00a63e',
    },
    dark: {
      surface: '#1a1a19',
      ink: '#ffffff',
      inkSecondary: '#c3c2b7',
      muted: '#898781',
      grid: '#2c2c2a',
      axis: '#383835',
      destructive: '#ff6467',
      success: '#05df72',
    },
  },
} as const

export type Chrome = (typeof palette.chrome)[Mode]

/** A filled × (ECharts path symbol): the conventional mark for poles, removed or rejected points. */
const CROSS = 'path://M0,2L2,0L5,3L8,0L10,2L7,5L10,8L8,10L5,7L2,10L0,8L3,5Z'

/**
 * Marker shapes by group, so group identity never relies on colour alone. Shape 4 is a cross (×), which pages use for
 * poles and discarded points.
 */
export const MARKER_SHAPES = ['circle', 'rect', 'triangle', 'diamond', CROSS, 'roundRect', 'arrow', 'circle'] as const

/** The number of categorical slots. */
export const SLOTS = palette.categorical.light.length

export function categorical(mode: Mode): readonly string[] {
  return palette.categorical[mode]
}

/** The colour of categorical slot `slot`. Slots never cycle: past the last one is an error. */
export function seriesColor(mode: Mode, slot: number): string {
  const colours = palette.categorical[mode]
  if (!(slot >= 0 && slot < colours.length)) throw new Error(`categorical slot ${slot} out of range (0–${SLOTS - 1})`)
  return colours[slot]
}

export type SequentialScale = {
  (mode?: Mode): readonly string[]
  readonly [index: number]: string
  readonly length: number
  [Symbol.iterator](): Iterator<string>
  slice(start?: number, end?: number): string[]
  map<U>(callbackfn: (value: string, index: number, array: readonly string[]) => U): U[]
}

/** The sequential ramp in a theme, from the surface (low) to the strong hue (high). Can also be indexed directly as an array. */
const defaultSequential = palette.sequential.light
const rawSequentialFn = (mode: Mode = 'light'): readonly string[] => palette.sequential[mode]
export const sequential: SequentialScale = new Proxy(rawSequentialFn as unknown as SequentialScale, {
  get(target, prop, receiver) {
    if (typeof prop === 'string' && /^\d+$/.test(prop)) {
      return defaultSequential[Number(prop)]
    }
    if (prop === 'length') {
      return defaultSequential.length
    }
    if (prop === Symbol.iterator) {
      return () => defaultSequential[Symbol.iterator]()
    }
    // oxlint-disable-next-line typescript/no-explicit-any
    if (prop in defaultSequential) {
      // oxlint-disable-next-line typescript/no-explicit-any
      const val = (defaultSequential as any)[prop]
      return typeof val === 'function' ? val.bind(defaultSequential) : val
    }
    return Reflect.get(target, prop, receiver)
  },
})

/** The diverging ramp in a theme: strong blue (low), the neutral midpoint (zero), strong red (high). */
export function diverging(mode: Mode): string[] {
  const d = palette.diverging
  return [...d.negative[mode], d.midpoint[mode], ...d.positive[mode]]
}

export function chrome(mode: Mode): Chrome {
  return palette.chrome[mode]
}

/** The red, green and blue channels of `#rrggbb` or `rgb(r, g, b)` / `rgba(…)`. */
function rgb(color: string): number[] {
  return color.startsWith('#')
    ? [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16))
    : (color.match(/[\d.]+/g) ?? ['0', '0', '0']).slice(0, 3).map(Number)
}

/** `color` blended towards `base`, keeping the fraction `t` of `color`: an opaque muted colour that tiles seamlessly. */
export function mute(color: string, base: string, t: number): string {
  const [a, b] = [rgb(color), rgb(base)]
  return `rgb(${a.map((c, j) => Math.round(b[j] + (c - b[j]) * t)).join(', ')})`
}

/**
 * The colour at fraction t ∈ [0, 1] along evenly spaced stops, interpolated linearly in RGB as ECharts' continuous
 * visualMap does, so a mark coloured here matches a heatmap cell of the same value.
 */
export function interpolateColors(stops: readonly string[], t: number): string {
  const u = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1) * (stops.length - 1)
  const i = Math.min(Math.floor(u), stops.length - 2)
  const [a, b] = [rgb(stops[i]), rgb(stops[i + 1])]
  return `rgb(${a.map((c, j) => Math.round(c + (b[j] - c) * (u - i))).join(', ')})`
}

/**
 * A two-stop scale from the strong blue (low) to the strong red (high) of the diverging palette, with no pale
 * midpoint: for thin marks drawn on the surface (the arrows of a vector field), where a pale stop would vanish.
 */
export function coolWarm(mode: Mode): readonly string[] {
  return [palette.diverging.negative[mode][2], palette.diverging.positive[mode][2]]
}

/** The stops of a continuous scale in a theme. */
export function scaleStops(scale: 'sequential' | 'diverging', mode: Mode): readonly string[] {
  return scale === 'diverging' ? diverging(mode) : sequential(mode)
}
