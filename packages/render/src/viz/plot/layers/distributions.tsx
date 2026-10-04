/** Probability layers drawn from aifn objects: a histogram of samples, a density, a mass function, a support band. */
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { Univariate } from 'aifn-compute/probability/distributions'
import { histogram, type BinRule } from 'aifn-compute/probability/stats'
import { chrome } from '@render/design/palette'
import { LINE_WIDTH } from '../../theme'
import type { Range } from '../../viewport'
import type { AxisInterval } from '../axis'
import { defineLayer, extentOf, oriented, orientedExtent, type CommonProps, type Orient } from '../layer'
import { distributionRange, evaluate, supportOf } from '../probability'
import { barSeries, polygonSeries, type CustomApi, type CustomParams } from './marks'

type Bar = [number, number, number, number]

// ── Histogram ────────────────────────────────────────────────────────────────────────────────────────────────────────

export type HistogramProps = CommonProps & {
  /** The samples. */
  values: ArrayLike<number>
  /** `aifn-compute/probability/stats` bin rule: a count, `{ width }`, `sturges`, `freedman-diaconis` or explicit edges. */
  bins?: BinRule
  /** The binned range (default the samples' min and max). */
  range?: [number, number]
  /** `density` (default: integrates to 1, comparable with a density curve) or `count`. */
  normalize?: 'density' | 'count'
  orient?: Orient
  /** Fill opacity (default 0.45; bars are outlined in their colour, so they read at any opacity). */
  opacity?: number
}

/** Histograms by samples array and options, so a histogram is binned once however often the Plot rebuilds it. */
const binned = new WeakMap<object, Map<string, Bar[]>>()

function histogramBars(p: HistogramProps): Bar[] {
  const key = JSON.stringify([p.bins ?? 'freedman-diaconis', p.range, p.normalize ?? 'density'])
  let byKey = binned.get(p.values as object)
  const hit = byKey?.get(key)
  if (hit) return hit
  const finite = Array.from(p.values).filter(Number.isFinite)
  if (!finite.length) return []
  const h = histogram(finite, { bins: p.bins ?? 'freedman-diaconis', range: p.range })
  const edges = toFlat(h.edges)
  const heights = toFlat(p.normalize === 'count' ? h.counts : h.density)
  const bars = heights.map((v, i): Bar => [edges[i], edges[i + 1], 0, v])
  if (!byKey) binned.set(p.values as object, (byKey = new Map()))
  byKey.set(key, bars)
  return bars
}

/**
 * A histogram of samples, binned by `aifn-compute/probability/stats` (Freedman–Diaconis by default), as touching bars outlined
 * in their colour; `orient="y"` lays it along the y axis (e.g. beside a map, for its output).
 */
export const Histogram = defineLayer<HistogramProps>({
  kind: 'Histogram',
  extent: (p) => {
    const bars = histogramBars(p)
    return orientedExtent(
      p.orient,
      extentOf(
        bars.map((b) => b[0]),
        bars.map((b) => b[1]),
      ),
      extentOf(
        [0],
        bars.map((b) => b[3]),
      ),
    )
  },
  build: (p, ctx) => {
    const name = p.name ?? 'histogram'
    const bars = histogramBars(p)
    return {
      series: [
        barSeries(ctx.id, name, bars, p.orient, { color: ctx.color, opacity: p.opacity ?? 0.45, outline: ctx.color }),
      ],
      hover:
        p.orient === 'y'
          ? []
          : [{ label: name, color: ctx.color, x: bars.map((b) => (b[0] + b[1]) / 2), y: bars.map((b) => b[3]) }],
      data: {
        kind: 'histogram',
        name,
        edges: [...bars.map((b) => b[0]), bars.at(-1)?.[1]],
        heights: bars.map((b) => b[3]),
      },
    }
  },
})

// ── Density ──────────────────────────────────────────────────────────────────────────────────────────────────────────

export type DensityProps = CommonProps & {
  /** A univariate continuous distribution from `aifn-compute/probability/distributions`. */
  dist: Univariate
  /** Where to evaluate it (default its 0.002–0.998 quantiles, padded, inside its support). */
  range?: Range
  /** Points along the range (default 400). */
  n?: number
  orient?: Orient
  /** Fill under the curve at this opacity (default none). */
  fill?: number
  dashed?: boolean
}

const densities = new WeakMap<object, Map<string, { x: number[]; y: number[] }>>()

function densityCurve(p: DensityProps): { x: number[]; y: number[] } {
  const [lo, hi] = p.range ?? distributionRange(p.dist)
  const n = p.n ?? 400
  const key = `${lo},${hi},${n}`
  let byKey = densities.get(p.dist)
  const hit = byKey?.get(key)
  if (hit) return hit
  // A bounded support's ends are left out when open, so a density that is infinite there does not spike the axis.
  const s = supportOf(p.dist)
  const eps = (hi - lo) * 1e-6
  const a = s && lo <= s.lower ? s.lower + (s.lowerOpen ? eps : 0) : lo
  const b = s && hi >= s.upper ? s.upper - (s.upperOpen ? eps : 0) : hi
  const x = Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
  const curve = { x, y: evaluate(p.dist, x) }
  if (!byKey) densities.set(p.dist, (byKey = new Map()))
  byKey.set(key, curve)
  return curve
}

/** The density of an aifn distribution as a curve (optionally filled), along x or, with `orient="y"`, along y. */
export const Density = defineLayer<DensityProps>({
  kind: 'Density',
  legend: (p) => [p.name ?? p.dist.name],
  extent: (p) => {
    const c = densityCurve(p)
    return orientedExtent(p.orient, extentOf(c.x), extentOf([0], c.y))
  },
  build: (p, ctx) => {
    const name = p.name ?? p.dist.name
    const c = densityCurve(p)
    const pts = c.x.map((v, i) => oriented(p.orient, v, c.y[i]))
    const series: Record<string, unknown>[] = []
    if (p.fill) {
      const base = [oriented(p.orient, c.x.at(-1)!, 0), oriented(p.orient, c.x[0], 0)]
      series.push(
        polygonSeries(
          `${ctx.id}:fill`,
          name,
          [...pts.filter((q) => q.every(Number.isFinite)), ...base],
          ctx.color,
          p.fill,
        ),
      )
    }
    series.push({
      id: ctx.id,
      name,
      type: 'line',
      data: pts.map((q) => (q.every(Number.isFinite) ? q : [null, null])),
      connectNulls: false,
      showSymbol: false,
      clip: true,
      lineStyle: { width: LINE_WIDTH, color: ctx.color, type: p.dashed ? 'dashed' : 'solid' },
      itemStyle: { color: ctx.color },
      emphasis: { focus: 'none' },
      z: 3,
    })
    return {
      series,
      hover: p.orient === 'y' ? [] : [{ label: name, color: ctx.color, x: c.x, y: c.y }],
      data: { kind: 'density', name, x: c.x, y: c.y },
    }
  },
})

// ── Mass ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

export type MassProps = CommonProps & {
  /** A discrete distribution from `aifn-compute/probability/distributions`. */
  dist: Univariate
  /** Integers to show (default its 0.002–0.998 quantiles, inside its support). */
  range?: Range
  orient?: Orient
  /** Bar width in data units (default 0.6). */
  width?: number
  opacity?: number
}

function massPoints(p: MassProps): { k: number[]; p: number[] } {
  const [lo, hi] = p.range ?? distributionRange(p.dist)
  const s = supportOf(p.dist)
  const a = Math.ceil(Math.max(lo, s?.lower ?? -Infinity))
  const b = Math.floor(Math.min(hi, s?.upper ?? Infinity))
  const k: number[] = []
  for (let v = a; v <= b && k.length < 5000; v++) k.push(v)
  return { k, p: evaluate(p.dist, k) }
}

/** The mass function of a discrete aifn distribution as a bar at each integer. */
export const Mass = defineLayer<MassProps>({
  kind: 'Mass',
  legend: (p) => [p.name ?? p.dist.name],
  extent: (p) => {
    const m = massPoints(p)
    const w = (p.width ?? 0.6) / 2
    return orientedExtent(
      p.orient,
      extentOf(
        m.k.map((v) => v - w),
        m.k.map((v) => v + w),
      ),
      extentOf([0], m.p),
    )
  },
  build: (p, ctx) => {
    const name = p.name ?? p.dist.name
    const m = massPoints(p)
    const w = (p.width ?? 0.6) / 2
    const bars = m.k.map((k, i): Bar => [k - w, k + w, 0, Number.isFinite(m.p[i]) ? m.p[i] : 0])
    return {
      series: [barSeries(ctx.id, name, bars, p.orient, { color: ctx.color, opacity: p.opacity ?? 0.9 })],
      hover: p.orient === 'y' ? [] : [{ label: name, color: ctx.color, x: m.k, y: m.p }],
      data: { kind: 'mass', name, k: m.k, p: m.p },
    }
  },
})

// ── SupportBand ──────────────────────────────────────────────────────────────────────────────────────────────────────

export type SupportBandProps = CommonProps & {
  /** The support: an interval (e.g. `supportInterval(d.support)` from aifn-compute/probability/bijectors) or a distribution. */
  interval?: AxisInterval
  dist?: Univariate
  orient?: Orient
  /** Shade the region outside the support (default true). */
  shade?: boolean
}

const intervalOf = (p: SupportBandProps): AxisInterval | undefined => p.interval ?? (p.dist && supportOf(p.dist))

/**
 * Where a distribution lives, on the axis: a band along the plot's edge over the support, ending in a filled dot at a
 * closed end, a hollow dot at an open end, and running off the plot where the support is unbounded; the region
 * outside the support is shaded.
 */
export const SupportBand = defineLayer<SupportBandProps>({
  kind: 'SupportBand',
  legend: () => [],
  needsBox: true,
  extent: (p) => {
    const s = intervalOf(p)
    if (!s) return undefined
    const ends = [s.lower, s.upper].filter(Number.isFinite)
    return orientedExtent(p.orient, extentOf(ends), undefined)
  },
  build: (p, ctx) => {
    const s = intervalOf(p)
    const box = ctx.box
    if (!s || !box) return { series: [] }
    const c = chrome(ctx.mode)
    const color = ctx.color
    const vertical = p.orient === 'y'
    const [lo, hi] = vertical ? box.y : box.x
    const shade = p.shade !== false
    const [other0] = vertical ? box.x : box.y
    return {
      series: [
        {
          id: ctx.id,
          name: p.name ?? '__support',
          type: 'custom',
          data: [[vertical ? other0 : lo, vertical ? lo : other0]],
          encode: { x: 0, y: 1 },
          silent: true,
          clip: false,
          tooltip: { show: false },
          itemStyle: { color },
          renderItem: (params: CustomParams, api: CustomApi) => {
            const sys = params.coordSys
            // Pixel position of a value along the band's axis, clamped to the plot.
            const at = (v: number) => {
              const clamped = Math.min(Math.max(v, lo), hi)
              return vertical ? api.coord([other0, clamped])[1] : api.coord([clamped, other0])[0]
            }
            const a = at(s.lower)
            const b = at(s.upper)
            const children: Record<string, unknown>[] = []
            const rect = (from: number, to: number) =>
              vertical
                ? { x: sys.x, y: Math.min(from, to), width: sys.width, height: Math.abs(to - from) }
                : { x: Math.min(from, to), y: sys.y, width: Math.abs(to - from), height: sys.height }
            if (shade) {
              const [start, end] = vertical ? [sys.y + sys.height, sys.y] : [sys.x, sys.x + sys.width]
              if (s.lower > lo)
                children.push({ type: 'rect', shape: rect(start, a), style: { fill: c.muted, opacity: 0.12 } })
              if (s.upper < hi)
                children.push({ type: 'rect', shape: rect(b, end), style: { fill: c.muted, opacity: 0.12 } })
            }
            // The band: 4 px along the bottom (or left) edge, inside the plot.
            const band = vertical
              ? { x1: sys.x + 3, y1: a, x2: sys.x + 3, y2: b }
              : { x1: a, y1: sys.y + sys.height - 3, x2: b, y2: sys.y + sys.height - 3 }
            children.push({ type: 'line', shape: band, style: { stroke: color, lineWidth: 4, opacity: 0.8 } })
            const end = (v: number, open: boolean, pixel: number) => {
              if (!Number.isFinite(v) || v < lo || v > hi) return
              const [cx, cy] = vertical ? [sys.x + 3, pixel] : [pixel, sys.y + sys.height - 3]
              children.push({
                type: 'circle',
                shape: { cx, cy, r: 4.5 },
                style: { fill: open ? c.surface : color, stroke: color, lineWidth: 2 },
              })
            }
            end(s.lower, !!s.lowerOpen, a)
            end(s.upper, !!s.upperOpen, b)
            return { type: 'group', children }
          },
          z: 0,
        },
      ],
    }
  },
})
