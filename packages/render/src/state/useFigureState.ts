/**
 * `useFigureState(schema)` (DESIGN.md §4): one declaration of a figure's parameters gives
 *
 * - typed values, on the state itself and under `values` (`state.x0`, `state.input.key`, `state.input.values.mu`,
 *   narrowed by the case's key, `state.reveal.jacobian`);
 * - the control rows (`<Figure state={state}>` draws them; `FigureControls` alone elsewhere);
 * - `state.set(path, value)`, clamped and snapped by the field's definition, from any input;
 * - `state.handle(path)` (or `[px, py]` for a point): a drag handle writing the same value, and `state.bind(path)`, a
 *   `Param` for a `Slider`, a probe or anything else that takes one;
 * - `state.reset()` and `state.isDefault`, which the Figure turns into a reset button;
 * - URL state: once a `Figure` attaches it, non-default values live in the query next to the figure's anchor, decoded
 *   through aifn's `clampReport` on load (dropped keys are listed in `state.dropped` and warned about in dev).
 *
 * The schema may be written inline: values are re-validated against the latest definitions on every render (a slider
 * whose range depends on the data stays in range), and an unchanged field keeps its object identity, so memos that
 * depend on `state.input.values` do not rerun when another field moves.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Handle } from '../viz/handles'
import { niceStep } from '../viz/format'
import { numberBounds } from './number'
import { sliderStep, type ParamDefs, type Values } from './schema'
import {
  fromEntries,
  initialRaw,
  isInitial,
  locate,
  normalise,
  setAt,
  toEntries,
  typedValues,
  type Raw,
  type TypedCache,
} from './store'
import { readFigureEntries, writeFigureEntries } from './url'

/** A numeric value with its range and step: what a slider, a handle or a probe binds to. */
export type Param = { value: number; set: (v: number) => void; min: number; max: number; step: number }

export type HandleOptions = {
  /** Which coordinate a one-value handle sets: a vertical guide (`x`, default) or a horizontal one (`y`). */
  axis?: 'x' | 'y'
  label?: string
  /** Called once when the pointer lets go. */
  onRelease?: () => void
}

export type FigureStateApi<S extends ParamDefs> = {
  /** The typed values (also spread on the state itself). */
  values: Values<S>
  /** The schema as last given. */
  schema: S
  /** Set a field by path (`x0`, `reveal.jacobian`, `input` to a case name, `input.mu`), clamped and snapped. */
  set: (path: string, value: unknown) => void
  /** A drag handle bound to a numeric field (`axis` x or y), or to two of them as a point. */
  handle: (path: string | readonly [string, string], options?: HandleOptions) => Handle
  /** A numeric field as a `Param` (value, set, min, max, step). */
  bind: (path: string) => Param
  /** Every field back to its initial value (and the figure's URL parameters removed). */
  reset: () => void
  /** True when every field is at its initial value. */
  isDefault: boolean
  /** The whole state as plain JSON, and its inverse (invalid parts fall back to initial values). */
  json: Raw
  setJson: (saved: unknown) => void
  /** URL keys that the last decode could not use (unknown, inactive or unparseable). */
  dropped: readonly string[]
  /** Called by `Figure`: bind the state to the figure's anchor, reading the URL once and writing it on change. */
  attach: (figureId: string) => void
}

export type FigureState<S extends ParamDefs> = Values<S> & FigureStateApi<S>

const RESERVED = new Set([
  'values',
  'schema',
  'set',
  'handle',
  'bind',
  'reset',
  'isDefault',
  'json',
  'setJson',
  'dropped',
  'attach',
])

const split = (path: string) => path.split('.')

export function useFigureState<const S extends ParamDefs>(
  schema: S,
  options: {
    /** Saved JSON state (`state.json`) to start from in place of the initial values; invalid parts are reset. */
    initial?: unknown
  } = {},
): FigureState<S> {
  if (import.meta.env.DEV)
    for (const key of Object.keys(schema))
      if (RESERVED.has(key)) throw new Error(`useFigureState: '${key}' is reserved; rename the field`)
  const latest = useRef(schema)
  useEffect(() => {
    latest.current = schema
  })
  const [raw, setRaw] = useState<Raw>(() =>
    options.initial === undefined ? initialRaw(schema) : normalise(schema, options.initial),
  )
  const [link, setLink] = useState<{ id: string; dropped: string[] } | null>(null)
  const [cache] = useState<TypedCache>(() => new Map())

  const set = useCallback((path: string, value: unknown) => {
    setRaw((r) => setAt(latest.current, r, split(path), value))
  }, [])
  const reset = useCallback(() => setRaw(initialRaw(latest.current)), [])
  const setJson = useCallback((saved: unknown) => setRaw(normalise(latest.current, saved)), [])
  const attached = useRef<string | null>(null)
  const attach = useCallback((id: string) => {
    if (attached.current === id) return
    attached.current = id
    const entries = readFigureEntries(id)
    const decoded = entries.length ? fromEntries(latest.current, entries) : null
    if (decoded?.dropped.length && import.meta.env.DEV)
      console.warn(`figure #${id}: URL keys not used: ${decoded.dropped.join(', ')}`)
    if (decoded) setRaw(decoded.raw)
    setLink({ id, dropped: decoded?.dropped ?? [] })
  }, [])

  // Write the non-default values next to the figure's anchor once attached.
  useEffect(() => {
    if (link) writeFigureEntries(link.id, toEntries(latest.current, raw))
  }, [raw, link])

  // Latest-schema validation each render; per-field identity is kept by the cache.
  const values = typedValues(schema, raw, cache) as Values<S>
  const valuesMemo = useStableRecord(values)

  const isDefault = isInitial(schema, raw)
  return useMemo(() => {
    const numeric = (path: string) => {
      const found = locate(schema, raw, split(path))
      if (!found || found.kind !== 'leaf' || (found.def.kind !== 'slider' && found.def.kind !== 'number'))
        throw new Error(`figure state: '${path}' is not a numeric field`)
      return found.def
    }
    const valueAt = (path: string) => {
      let v: unknown = valuesMemo
      for (const part of split(path)) {
        const o = v as Record<string, unknown> & { values?: Record<string, unknown> }
        v = part in o ? o[part] : o.values?.[part]
      }
      return v as number
    }
    const bind = (path: string): Param => {
      const def = numeric(path)
      // A number's strict bounds are approximated by inclusive ones here; its field validates them exactly.
      const { lower: min, upper: max } = def.kind === 'number' ? numberBounds(def) : { lower: def.min, upper: def.max }
      const step =
        def.kind === 'slider' ? sliderStep(def) : (def.step ?? (Number.isFinite(min + max) ? niceStep(min, max) : 1))
      return { value: valueAt(path), set: (v: number) => set(path, v), min, max, step }
    }
    const handle = (path: string | readonly [string, string], options: HandleOptions = {}): Handle => {
      const { label, onRelease } = options
      if (typeof path !== 'string') {
        const [px, py] = path
        numeric(px)
        numeric(py)
        return {
          kind: 'point',
          at: [valueAt(px), valueAt(py)],
          onDrag: ([x, y]) => {
            set(px, x)
            set(py, y)
          },
          label,
          onRelease,
        }
      }
      numeric(path)
      const at = valueAt(path)
      const onDrag = (v: number) => set(path, v)
      return options.axis === 'y'
        ? { kind: 'y', at, onDrag, label, onRelease }
        : { kind: 'x', at, onDrag, label, onRelease }
    }
    const api: FigureStateApi<S> = {
      values: valuesMemo,
      schema,
      set,
      handle,
      bind,
      reset,
      isDefault,
      json: raw,
      setJson,
      dropped: link?.dropped ?? [],
      attach,
    }
    return { ...valuesMemo, ...api } as FigureState<S>
  }, [valuesMemo, schema, set, reset, isDefault, raw, setJson, link, attach])
}

/** The same record object while every entry is identical, so a re-render with unchanged values keeps memos valid. */
function useStableRecord<T extends object>(next: T): T {
  // A box held in state rather than a ref: it is read during render, and written only when the record changes.
  const [box] = useState(() => new Box(next))
  const prev = box.value as Record<string, unknown>
  const same =
    prev === next ||
    (Object.keys(prev).length === Object.keys(next).length &&
      Object.entries(next).every(([k, v]) => Object.is(prev[k], v)))
  const value = same ? (prev as T) : next
  useEffect(() => box.set(value))
  return value
}

/** A mutable cell kept across renders. */
class Box<T> {
  value: T
  constructor(value: T) {
    this.value = value
  }
  set(value: T) {
    this.value = value
  }
}
