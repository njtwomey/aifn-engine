/**
 * Hover and pin, the lab's one selection pattern: hovering an item previews it, a click pins it, a second click on it
 * (or `clear`, e.g. a click on empty space) or Escape unpins. The pin lives in a hidden figure field (`pinField`), so it
 * is kept in the URL and a link reproduces it; the hover is transient. `focus` is what a figure shows: the hovered item,
 * else the pinned one.
 *
 *   const state = useFigureState({ …, pin: pinField() })
 *   const pin = usePinned(state.pin, (v) => state.set('pin', v))
 *   <Node onPointerEnter={() => pin.hover(k)} onPointerLeave={() => pin.hover(null)} onClick={() => pin.toggle(k)} />
 *
 * Items are non-negative integers (an index, a node id); a pair such as a text span packs into one (`packPair`).
 */
import { useEffect, useRef, useState } from 'react'
import { int } from './schema'

/**
 * The field that holds a pin: −1 when nothing is pinned. It draws no control (`onChart`: the chart sets it) and, being
 * active, is kept in the URL (a field whose `when` fails is not written there).
 */
export const pinField = (max = Number.MAX_SAFE_INTEGER) => int(-1, { min: -1, max, onChart: true })

export type Pinned = {
  /** The pinned item, or null. */
  pinned: number | null
  /** The hovered item, or null. */
  hovered: number | null
  /** What to show: the hovered item, else the pinned one. */
  focus: number | null
  hover: (item: number | null) => void
  /** Pin `item`, or unpin it if it is the one pinned. */
  toggle: (item: number) => void
  pin: (item: number) => void
  clear: () => void
}

/**
 * The hover-and-pin state over a pin field's `value` (set through `set`). `valid` rejects a pinned value that no longer
 * names an item (a smaller tree after a change), which then reads as no pin.
 */
export function usePinned(
  value: number,
  set: (value: number) => void,
  options: { valid?: (item: number) => boolean } = {},
): Pinned {
  const [hovered, setHovered] = useState<number | null>(null)
  const pinned = value >= 0 && (options.valid?.(value) ?? true) ? value : null
  const has = pinned !== null
  useEffect(() => {
    if (!has) return
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') set(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- `set` is usually a fresh closure over a stable setter
  }, [has])
  return {
    pinned,
    hovered,
    focus: hovered ?? pinned,
    hover: setHovered,
    toggle: (item) => set(pinned === item ? -1 : item),
    pin: (item) => set(item),
    clear: () => set(-1),
  }
}

/**
 * The field that holds a pin on a named item (`usePinnedName`): its index in the items, −1 when nothing is pinned, −2
 * (the default, so absent from the URL) for the hook's `initial` name.
 */
export const namedPinField = (max = Number.MAX_SAFE_INTEGER) => int(-2, { min: -2, max, onChart: true })

/**
 * A pin on a named item (a word, a document) whose position in `items` changes with the figure's settings (a new
 * representation re-indexes the vocabulary). The field (`namedPinField`) holds the index, so the URL keeps the pin; the
 * name is carried across a re-indexing, and the field follows it. A name missing from the current items reads as no
 * pin and comes back when the items include it again. `name` is the pinned name, also while it is missing.
 */
export function usePinnedName(
  value: number,
  set: (value: number) => void,
  items: readonly string[] | null,
  initial: string | null = null,
): Pinned & { name: string | null } {
  const last = useRef<string | null>(null)
  let name: string | null
  if (value === -2) name = initial
  else if (value < 0) name = null
  // The field names an item: on load (no name yet) or after a pin through this hook (the names agree).
  else if (items && value < items.length && (last.current === null || items[value] === last.current))
    name = items[value]
  // The items were re-indexed under the field: keep the name.
  else name = last.current
  last.current = name
  const index = name !== null && items ? items.indexOf(name) : -1
  useEffect(() => {
    if (value >= 0 && index >= 0 && index !== value) set(index)
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- `set` is usually a fresh closure over a stable setter
  }, [value, index])
  const pins = usePinned(index, (i) => {
    last.current = i >= 0 && items ? items[i] : null
    set(i)
  })
  return { ...pins, name }
}

/** Two non-negative integers below 2²⁶ (e.g. a span's start and end) as one pin item, and back. */
const PAIR = 2 ** 26
export const packPair = (a: number, b: number) => a * PAIR + b
export const unpackPair = (v: number): [number, number] => [Math.floor(v / PAIR), v % PAIR]
