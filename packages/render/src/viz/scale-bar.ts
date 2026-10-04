/** The colour bar's geometry and ticks, shared by `ScaleBar` and the `Raster` layer. */
/** The colour bar's geometry: gap from the plot, bar width, tick length, gap before the tick labels. */
export const BAR = { gap: 12, width: 10, tick: 4, labelGap: 3 }
/** 3–5 round values (1, 2 or 5 × 10ᵏ apart) inside [lo, hi], for the colour bar's axis; the ends if none fit. */
export function colorBarTicks(lo: number, hi: number): number[] {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return Number.isFinite(lo) ? [lo] : []
  const span = hi - lo
  const unit = 10 ** Math.floor(Math.log10(span / 10))
  const at = (step: number) => {
    const out: number[] = []
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step)
      out.push(Number(v.toPrecision(12)))
    return out
  }
  let best: number[] | null = null
  for (const m of [1, 2, 5, 10, 20, 50, 100]) {
    const ticks = at(m * unit)
    if (ticks.length >= 3 && ticks.length <= 5 && (!best || Math.abs(ticks.length - 4) < Math.abs(best.length - 4)))
      best = ticks
  }
  return best ?? [lo, hi]
}
