/** Contour lines of a field sampled on a rectangular grid, as ECharts series; the marching squares are aifn's. */
import { contourLines } from 'aifn-compute/numerics/geometry'
import { chrome, type Mode } from '@render/design/palette'
import { formatNumber } from './format'

/**
 * Contour lines as ECharts series: one line series (polylines from `aifn-compute/numerics/geometry` split by nulls) and one
 * label per level. Ink by default. Static (no animation), so a redraw does not diff every contour point.
 */
export function contourSeries(
  x: readonly number[],
  y: readonly number[],
  field: readonly (readonly number[])[],
  levels: readonly number[],
  mode: Mode,
  {
    id = '__contours',
    color,
    labels = true,
    width = 1,
  }: { id?: string; color?: string; labels?: boolean; width?: number } = {},
): Record<string, unknown>[] {
  const { ink, surface } = chrome(mode)
  const stroke = color ?? ink
  const lines: (number | null)[][] = []
  const marks: number[][] = []
  for (const level of levels) {
    // Joined polylines, so each level is a few runs of points rather than one null break per segment.
    let longest: Float64Array | null = null
    for (const line of contourLines(x, y, field, level)) {
      const p = line.data as Float64Array
      for (let k = 0; k < p.length; k += 2) lines.push([p[k], p[k + 1]])
      lines.push([null, null])
      if (!longest || p.length > longest.length) longest = p
    }
    // The label sits at the middle point of the level's longest line.
    if (longest) {
      const k = 2 * Math.floor(longest.length / 4)
      marks.push([longest[k], longest[k + 1], level])
    }
  }
  return [
    {
      id,
      name: '__contours',
      type: 'line',
      data: lines,
      connectNulls: false,
      showSymbol: false,
      silent: true,
      clip: true,
      lineStyle: { color: stroke, width },
      tooltip: { show: false },
      animation: false,
      z: 2,
    },
    ...(labels
      ? [
          {
            id: `${id}-labels`,
            name: '__contour-labels',
            type: 'scatter',
            data: marks,
            symbolSize: 0,
            silent: true,
            clip: true,
            label: {
              show: true,
              formatter: (p: { value: number[] }) => formatNumber(p.value[2]),
              color: stroke,
              fontSize: 10,
              backgroundColor: surface,
              padding: [1, 3],
              borderRadius: 2,
            },
            tooltip: { show: false },
            animation: false,
            z: 4,
          },
        ]
      : []),
  ]
}

/**
 * The argmax margin fields of K score fields on one grid (e.g. class probabilities or mixture weights): mₖ(x) =
 * sₖ(x) − max_{j≠k} sⱼ(x), positive exactly where k wins. The level-0 contour of mₖ is the boundary of k's region, so
 * drawing every mₖ at 0 draws the argmax boundaries (each shared boundary appears in two margins). With K = 2, m₁ = s₁ − s₂
 * alone suffices (for probabilities, the s₁ = 0.5 contour).
 */
export function argmaxMargins(fields: readonly (readonly (readonly number[])[])[]): number[][][] {
  const K = fields.length
  if (K === 0) return []
  const rows = fields[0].length
  return fields.map((_, k) =>
    Array.from({ length: rows }, (__, i) =>
      Array.from({ length: fields[0][i].length }, (___, j) => {
        let other = -Infinity
        for (let q = 0; q < K; q++) if (q !== k) other = Math.max(other, fields[q][i][j])
        return fields[k][i][j] - other
      }),
    ),
  )
}
