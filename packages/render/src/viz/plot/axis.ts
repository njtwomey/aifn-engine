/**
 * The axis model (DESIGN.md §5). One `AxisModel` per axis, shared by every `Plot` that draws it: the plots report their
 * layers' data extents to it, and it answers with the one range they all draw. Fitting, holding, zoom, pan, the
 * drag freeze and equal units live here, so panels that share an axis cannot disagree.
 *
 * Fit policies, chosen by the options:
 * - fit (default): the union of the layers' extents, rounded out to whole ticks, refitted whenever the data changes;
 * - hold: `hold: 'initial'` keeps the first fit, `hold: 'union'` grows it to take in new data but never shrinks it;
 *   `key` refits (e.g. a new family or dataset, not a new parameter value);
 * - fixed: `range: [lo, hi]`; either end may be left undefined to fit that end;
 * - support: `support` (an aifn `Interval`): bounded ends are hard limits (the range ends there, and zoom and pan stay
 *   inside), unbounded ends fit the data (a density layer's extent is its 0.002 and 0.998 quantiles).
 */
import { useLayoutEffect, useMemo, useState, useSyncExternalStore } from 'react'
import { niceRange } from './ticks'
import { panRange, validRange, zoomRange, type Range } from '../viewport'

/** An interval of the real line, as `aifn-compute/probability/bijectors` gives it (`supportInterval(d.support)`). */
export type AxisInterval = { lower: number; upper: number; lowerOpen?: boolean; upperOpen?: boolean }

export type AxisOptions = {
  /** The axis name, drawn beside the axis and on its toolbar button. */
  label?: string
  /** Hold the range while the data changes: `initial` keeps the first fit, `union` grows it. Absent: always refit. */
  hold?: 'initial' | 'union'
  /** Refit a held axis when this changes. */
  key?: unknown
  /** Fixed ends; an undefined end fits the data. Both ends given: the axis is fixed (still zoomable). */
  range?: readonly [number | undefined, number | undefined]
  /** Bounded ends of this interval are hard limits of the axis; unbounded ends fit the data. */
  support?: AxisInterval
  log?: boolean
  /** A categorical axis: category k sits at position k, labelled with its name; the range is [−0.5, K − 0.5]. */
  categories?: readonly string[]
  /** Equal units: this axis takes the same length per unit as `equal` in every Plot that draws both. */
  equal?: AxisModel
  /** Round a fitted range out to whole ticks (default true). */
  nice?: boolean
  /** Zoom, pan and the toolbar button (default true, false for categorical axes). */
  zoom?: boolean
  /** Tick, tooltip and readout labels. */
  format?: (v: number) => string
  /** Whole-number ticks only (steps, counts, iterations), at a regular 1-2-5 spacing. */
  integer?: boolean
  /** Values increase downwards (y) or leftwards (x), e.g. image rows with row 0 at the top. */
  inverse?: boolean
}

type Listener = () => void

/** A data extent; a third element `true` marks it tight (a raster's grid): the axis ends there exactly, unrounded. */
export type Extent = readonly [number, number] | readonly [number, number, true]

let nextId = 0

export class AxisModel {
  readonly id = `axis${++nextId}`
  options: AxisOptions
  /** Extents reported by each source (a Plot), in data units. */
  private extents = new Map<string, Extent>()
  /** The zoomed range, or undefined when the axis shows its base (fitted, held or fixed) range. */
  private view: Range | undefined
  private held: { key: unknown; range: Range } | undefined
  /** Set while a handle is dragged on any Plot drawing this axis: the axis holds still under the pointer. */
  private frozen: Range | undefined
  private holdPending = false
  /** Axes whose `equal` option names this one: zooming either zooms both. */
  readonly linked = new Set<AxisModel>()
  private listeners = new Set<Listener>()
  private version = 0

  constructor(options: AxisOptions) {
    this.options = options
  }

  /** The latest options, set by `useAxis` on every render of the component that owns the axis. */
  configure(options: AxisOptions) {
    this.options = options
  }

  // ── Store protocol (useSyncExternalStore) ──────────────────────────────────────────────────────────────────────────

  subscribe = (listener: Listener) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }
  getVersion = () => this.version
  private emit() {
    this.version++
    for (const l of [...this.listeners]) l()
  }

  // ── Ranges ─────────────────────────────────────────────────────────────────────────────────────────────────────────

  get log(): boolean {
    return !!this.options.log && !this.options.categories
  }
  get categorical(): boolean {
    return !!this.options.categories
  }
  get zoomable(): boolean {
    return this.options.zoom ?? !this.options.categories
  }
  /** The axis this one keeps equal units with, in either direction. */
  get partner(): AxisModel | undefined {
    return this.options.equal ?? [...this.linked][0]
  }

  /** The union of every source's extent, with `source`'s own replaced by `extent` (its value in this render). */
  private dataExtent(source?: string, extent?: Extent): { range: Range; tight: [boolean, boolean] } | undefined {
    let lo = Infinity
    let hi = -Infinity
    const all: Extent[] = []
    for (const [key, r] of this.extents) if (key !== source) all.push(r)
    if (extent) all.push(extent)
    for (const r of all) {
      lo = Math.min(lo, r[0])
      hi = Math.max(hi, r[1])
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return undefined
    // An end is tight when a tight extent (a raster's grid) sets it: the axis ends exactly there, not at a tick.
    const tight: [boolean, boolean] = [
      all.some((r) => r[2] === true && r[0] === lo),
      all.some((r) => r[2] === true && r[1] === hi),
    ]
    return { range: [lo, hi], tight }
  }

  /** The fitted range: the data's extent under the fit policy (fixed ends, support limits, nice rounding). */
  fit(source?: string, extent?: Extent): Range | undefined {
    const { categories, range, support, nice = true } = this.options
    if (categories) return [-0.5, Math.max(categories.length, 1) - 0.5]
    const log = this.log
    const [f0, f1] = range ?? []
    if (f0 !== undefined && f1 !== undefined) return [f0, f1]
    const found = this.dataExtent(source, extent)
    let data = found?.range
    if (data && log) data = data[0] > 0 ? data : data[1] > 0 ? [data[1] / 10, data[1]] : undefined
    const lower = support && Number.isFinite(support.lower) ? support.lower : undefined
    const upper = support && Number.isFinite(support.upper) ? support.upper : undefined
    if (!data) {
      if (lower !== undefined && upper !== undefined) return [lower, upper]
      return undefined
    }
    let [lo, hi] = data
    if (lo === hi) [lo, hi] = log ? [lo / 10, hi * 10] : [lo - 1, hi + 1]
    if (nice) {
      const [nlo, nhi] = niceRange([lo, hi], log)
      if (!found?.tight[0]) lo = nlo
      if (!found?.tight[1]) hi = nhi
    }
    // Hard limits: a bounded end of the support is where the axis ends, whatever the rounding did.
    if (lower !== undefined) lo = lower
    if (upper !== undefined) hi = upper
    if (f0 !== undefined) lo = f0
    if (f1 !== undefined) hi = f1
    return validRange([lo, hi], log) ? [lo, hi] : undefined
  }

  /** The unzoomed range: fixed, held or fitted. */
  base(source?: string, extent?: Extent): Range | undefined {
    const fitted = this.fit(source, extent)
    const { hold, key } = this.options
    if (!hold || !this.held || this.held.key !== key) return fitted
    if (hold === 'union' && fitted) return union(this.held.range, fitted)
    return this.held.range
  }

  /** The range to draw now: frozen during a drag, else zoomed, else the base range. */
  resolve(source?: string, extent?: Extent): Range | undefined {
    return this.frozen ?? this.view ?? this.base(source, extent)
  }

  /** Whether the axis shows a zoomed range. */
  get zoomed(): boolean {
    return this.view !== undefined
  }

  // ── Reports from plots ─────────────────────────────────────────────────────────────────────────────────────────────

  /** A Plot's extent on this axis (undefined withdraws it). Called from layout effects. */
  report(source: string, extent: Extent | undefined) {
    const before = this.resolve()
    const old = this.extents.get(source)
    if (extent) {
      if (old && old[0] === extent[0] && old[1] === extent[1] && old[2] === extent[2]) return this.takeHold()
      // If data extent changed (points added or dropped) and this axis is not held, auto-scale by clearing manual zoom
      if (!this.options.hold) this.view = undefined
      this.extents.set(source, extent)
    } else if (old) {
      if (!this.options.hold) this.view = undefined
      this.extents.delete(source)
    } else return
    this.takeHold()
    if (!same(before, this.resolve())) this.emit()
  }

  /**
   * Take (or grow) the held range once every Plot of this commit has reported, so the first hold covers all of them:
   * layout effects of one commit run together, and the microtask runs after them.
   */
  private takeHold() {
    const { hold } = this.options
    if (!hold || this.holdPending) return
    this.holdPending = true
    queueMicrotask(() => {
      this.holdPending = false
      const { key } = this.options
      const fitted = this.fit()
      if (!fitted) return
      const before = this.held
      if (!this.held || this.held.key !== key) this.held = { key, range: fitted }
      else if (this.options.hold === 'union') {
        const grown = union(this.held.range, fitted)
        if (!same(grown, this.held.range)) this.held = { key, range: grown }
      }
      if (before !== this.held) this.emit()
    })
  }

  // ── Zoom, pan, reset, freeze ───────────────────────────────────────────────────────────────────────────────────────

  /** Clamp a range inside the support's bounded ends, keeping its span where it fits. */
  private clamp(r: Range): Range {
    const s = this.options.support
    if (!s) return r
    let [lo, hi] = r
    const span = hi - lo
    if (Number.isFinite(s.lower) && lo < s.lower) [lo, hi] = [s.lower, s.lower + span]
    if (Number.isFinite(s.upper) && hi > s.upper) [lo, hi] = [s.upper - span, s.upper]
    if (Number.isFinite(s.lower)) lo = Math.max(lo, s.lower)
    return [lo, hi]
  }

  /** Show `r` (a typed range or a zoom); undefined returns to the base range. */
  setView(r: Range | undefined) {
    const next = r && validRange(r, this.log) ? this.clamp(r) : undefined
    if (same(next, this.view)) return
    this.view = next
    this.emit()
  }

  /** Zoom by `factor` (above 1 out, below 1 in) about `about`; an equal-units partner zooms by the same factor. */
  zoom(factor: number, about?: number, withPartner = true) {
    const r = this.resolve()
    if (r && validRange(r, this.log)) this.setView(zoomRange(r, factor, this.log, about))
    if (withPartner) this.partner?.zoom(factor, undefined, false)
  }

  pan(fraction: number) {
    const r = this.resolve()
    if (r && validRange(r, this.log)) this.setView(panRange(r, fraction, this.log))
  }

  /** Back to the fitted range: no zoom, and a held axis refits to the data now. */
  reset() {
    this.view = undefined
    const fitted = this.fit()
    if (this.options.hold && fitted) this.held = { key: this.options.key, range: fitted }
    this.emit()
  }

  /** Freeze at `r` while a handle is dragged; `undefined` releases (and refits). */
  setFrozen(r: Range | undefined) {
    if (same(r, this.frozen)) return
    this.frozen = r
    this.emit()
  }

  /** Tick label text for a value on this axis. */
  label(v: number): string {
    const { categories, format } = this.options
    if (categories) return Number.isInteger(v) && v >= 0 && v < categories.length ? categories[v] : ''
    return format ? format(v) : ''
  }
}

const union = (a: Range, b: Range): Range => [Math.min(a[0], b[0]), Math.max(a[1], b[1])]
const same = (a: Range | undefined, b: Range | undefined) => a === b || (!!a && !!b && a[0] === b[0] && a[1] === b[1])

/**
 * One axis model for the life of the component, with the latest options. Pass the same model to every Plot that
 * shares the axis. Changing `key` refits a held axis; changing a fixed `range` or `support` takes effect at once.
 */
export function useAxis(options: AxisOptions = {}): AxisModel {
  const [model] = useState(() => new AxisModel(options))
  // The latest options, read by every Plot rendering in this pass (they render after the component that owns the axis).
  model.configure(options)
  const partner = options.equal
  useLayoutEffect(() => {
    if (!partner) return
    partner.linked.add(model)
    return () => void partner.linked.delete(model)
  }, [partner, model])
  return model
}

/** Re-render when any of these axes changes (its range, zoom, hold or freeze). */
export function useAxisVersion(axes: readonly (AxisModel | undefined)[]): string {
  const ids = axes.map((a) => a?.id ?? '').join(',')
  // Stable while the set of axes is (models are identified by id): a new subscribe function would resubscribe on
  // every render.
  const [subscribe, snapshot] = useMemo(
    () => [
      (l: () => void) => {
        const offs = axes.map((a) => a?.subscribe(l))
        return () => offs.forEach((off) => off?.())
      },
      () => axes.map((a) => a?.getVersion() ?? 0).join(':'),
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the axes' ids
    [ids],
  )
  return useSyncExternalStore(subscribe, snapshot, snapshot)
}
