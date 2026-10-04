import { useMemo } from 'react'
import { slider, sliderStep } from '@render/state/schema'
import { useFigureState, type Param } from '@render/state/useFigureState'

export type { Param } from '@render/state/useFigureState'
export type ParamSpec = { min: number; max: number; step?: number }

/**
 * One adjustable value with its range and step, as a one-field figure state: every input bound to it (slider, number
 * field, chart handle, button) calls `set`, which clamps to the range and snaps to the step. Without a step it takes a
 * nice 1-2-5 step from the range. Prefer `useFigureState` for a figure's parameters; this remains for single values.
 */
export function useParam(initial: number, { min, max, step }: ParamSpec): Param {
  const state = useFigureState({ v: slider(min, max, initial, { step }) })
  const resolved = sliderStep({ kind: 'slider', min, max, initial, step })
  const { v: value, set } = state
  return useMemo(
    () => ({ value, set: (x: number) => set('v', x), min, max, step: resolved }),
    [value, set, min, max, resolved],
  )
}
