/** Number formatting shared by axes, tooltips, readouts and fields. */

/** Four significant figures, exponent form for very small or large magnitudes: axes read 0.5, not 0.49999999. */
export function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '∞' : '−∞'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2).replace(/\.?0+e/, 'e')
  return String(Number(v.toPrecision(4)))
}

const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'

/** A log-axis tick label: 10⁻³ rather than 0.001; values between powers of ten fall back to `formatNumber`. */
export function formatPower(v: number): string {
  const e = Math.round(Math.log10(v))
  if (!(Math.abs(v - 10 ** e) <= 1e-9 * 10 ** e)) return formatNumber(v)
  if (e === 0) return '1'
  const digits = [...String(Math.abs(e))].map((d) => SUPERSCRIPT[Number(d)]).join('')
  return `10${e < 0 ? '⁻' : ''}${digits}`
}

/**
 * A nice step for a range: 1, 2 or 5 × 10ᵏ, giving between about 100 and 250 positions across it. Sliders use it when
 * no step is given.
 */
export function niceStep(min: number, max: number, positions = 250): number {
  const raw = (max - min) / positions
  if (!(raw > 0) || !Number.isFinite(raw)) return 1
  const p = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 5, 10].map((m) => m * p).find((c) => c >= raw * (1 - 1e-9))!
}

/** Decimal places needed to write multiples of `step` exactly (0.05 → 2, 1e-4 → 4, 10 → 0). */
export function stepDecimals(step: number): number {
  if (!(step > 0)) return 0
  const text = String(step)
  const exp = text.match(/e-(\d+)$/)
  if (exp) return Number(exp[1]) + (text.split('e')[0].split('.')[1]?.length ?? 0)
  return text.split('.')[1]?.length ?? 0
}
