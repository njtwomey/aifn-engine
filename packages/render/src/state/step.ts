/**
 * Snapping and showing a stepped number: the one rule every input of a figure value goes through (sliders, number
 * fields, chart handles, URL values).
 */
import { formatNumber, stepDecimals } from '../viz/format'

/**
 * `v` clamped to [min, max] and snapped to the nearest multiple of `step` (k·step, not min + k·step), so a slider from
 * 0.02 with step 0.05 reaches 0.4 rather than 0.42. The bounds themselves stay reachable when they are off the grid.
 * Results carry no floating-point noise (0.30000000000000004 is 0.3).
 */
export function snapToStep(v: number, min: number, max: number, step: number): number {
  const clamped = Math.min(Math.max(v, min), max)
  if (!(step > 0) || !Number.isFinite(clamped)) return clamped
  const clean = (x: number) => Number(x.toFixed(Math.min(stepDecimals(step) + 2, 20)))
  const k = Math.floor(clamped / step)
  let best = min
  for (const c of [min, max, clean(k * step), clean((k + 1) * step)]) {
    if (c < min || c > max) continue
    if (Math.abs(c - clamped) < Math.abs(best - clamped)) best = c
  }
  return best
}

/**
 * A number as an input field shows it: whole numbers in full (123456, not 1.23e+5), anything else as `formatNumber`
 * does, with up to 8 significant figures for large values so that the field round-trips what it holds.
 */
export function formatField(v: number): string {
  if (!Number.isFinite(v)) return formatNumber(v)
  const a = Math.abs(v)
  if (Number.isInteger(v) && a < 1e15) return String(v)
  if (a >= 1e5 && a < 1e15) return String(Number(v.toPrecision(8)))
  return formatNumber(v)
}
