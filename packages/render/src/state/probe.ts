/**
 * Probes (DESIGN.md §6): a point the reader moves, with numbers computed there. One probe is shared by every `Plot`
 * that shows its coordinate (`<Probe probe={p} />` in each), so dragging it on one chart moves it on all of them; a
 * probe with `{ x, y }` is a point (on a heatmap, pressing anywhere moves it: click-to-set). `ProbeReadout` reports
 * the values at the probe.
 *
 *   const state = useFigureState({ x0: slider(-4, 4, 0.5, { onChart: true }) })
 *   const probe = useProbe({ x: state.bind('x0'), label: 'x₀' })
 *   <Plot …><Curve … /><Probe probe={probe} at={f(probe.x)} /></Plot>
 *   <ProbeReadout probe={probe} values={{ 'f(x₀)': f(probe.x), "f′(x₀)": df(probe.x) }} />
 */
import { useCallback, useMemo, useState } from 'react'
import type { Param } from './useFigureState'

/** A probe's position and setters. `x` and `y` are present when the probe has that coordinate. */
export type ProbeModel = {
  x?: number
  y?: number
  setX: (v: number) => void
  setY: (v: number) => void
  /** Set both coordinates (a point probe), clamped and snapped by their params. */
  set: (p: { x?: number; y?: number }) => void
  /** What the probe's x stands for, e.g. 'x₀'. */
  label?: string
  /** What its y stands for (default 'y'). */
  yLabel?: string
}

/** A coordinate: a `Param` (from `state.bind(name)` or `useParam`), or a starting value the probe keeps itself. */
type Coordinate = Param | number

function useCoordinate(c: Coordinate | undefined): { value?: number; set: (v: number) => void } {
  const [own, setOwn] = useState(typeof c === 'number' ? c : 0)
  const param = typeof c === 'object' ? c : undefined
  const setParam = param?.set
  const set = useCallback((v: number) => (setParam ? setParam(v) : setOwn(v)), [setParam])
  if (c === undefined) return { set }
  return { value: param ? param.value : own, set }
}

/** A probe over one or two coordinates. */
export function useProbe({
  x,
  y,
  label,
  yLabel,
}: {
  x?: Coordinate
  y?: Coordinate
  label?: string
  yLabel?: string
}): ProbeModel {
  const cx = useCoordinate(x)
  const cy = useCoordinate(y)
  const { set: setX } = cx
  const { set: setY } = cy
  return useMemo(
    () => ({
      x: cx.value,
      y: cy.value,
      setX,
      setY,
      set: (p: { x?: number; y?: number }) => {
        if (p.x !== undefined) setX(p.x)
        if (p.y !== undefined) setY(p.y)
      },
      label,
      yLabel,
    }),
    [cx.value, cy.value, setX, setY, label, yLabel],
  )
}
