import { useCallback } from 'react'
import { interpolateColors, scaleStops } from '../design/palette'
import { useTheme } from '../design/theme'

/**
 * The colour at fraction t ∈ [0, 1] of a sequential or diverging scale in the current theme: the colour a `Raster` with
 * that `scale` gives the value at t of its range. For marks that encode an ordered value, e.g. class k of K.
 */
export function useScaleColor(scale: 'sequential' | 'diverging'): (t: number) => string {
  const { resolved: mode } = useTheme()
  return useCallback((t: number) => interpolateColors(scaleStops(scale, mode), t), [scale, mode])
}
