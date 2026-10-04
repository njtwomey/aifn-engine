import { useId } from 'react'
import { formatNumber } from './format'
import { BAR } from './scale-bar'

/**
 * The colour bar: a vertical ramp beside the plot, as tall as the plot area, low values at the bottom, with a small axis
 * of labelled ticks and the value's name above it. Drawn as SVG over the chart's box, so it follows the plot area.
 */
export function ScaleBar({
  stops,
  lo,
  hi,
  ticks,
  label,
  left,
  top,
  height,
  room,
}: {
  stops: readonly string[]
  lo: number
  hi: number
  ticks: readonly number[]
  label: string
  left: number
  top: number
  height: number
  room: number
}) {
  // useId can contain characters that are not valid in a url(#…) reference.
  const id = `scale${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const at = (v: number) => height * (1 - (hi > lo ? (v - lo) / (hi - lo) : 0.5))
  const tickX = BAR.width + BAR.tick
  // The value's name starts over the bar; a long name is shifted left to stay inside the chart.
  const labelX = Math.min(0, room - label.length * 6.5)
  return (
    <svg
      width={Math.max(room, 1)}
      height={height + 20}
      role="img"
      aria-label={`colour scale: ${label} from ${formatNumber(lo)} to ${formatNumber(hi)}`}
      className="pointer-events-none absolute overflow-visible"
      style={{ left, top: top - 20 }}
    >
      <defs>
        <linearGradient id={id} x1="0" y1="1" x2="0" y2="0">
          {stops.map((c, i) => (
            <stop key={i} offset={i / (stops.length - 1)} stopColor={c} />
          ))}
        </linearGradient>
      </defs>
      <text x={labelX} y={11} className="fill-muted-foreground text-[11px]">
        {label}
      </text>
      <g transform="translate(0 20)">
        <rect
          x={0}
          y={0}
          width={BAR.width}
          height={height}
          fill={`url(#${id})`}
          className="stroke-border"
          strokeWidth={1}
        />
        {ticks
          .filter((v) => v >= Math.min(lo, hi) - 1e-12 && v <= Math.max(lo, hi) + 1e-12)
          .map((v) => (
            <g key={v}>
              <line x1={BAR.width} x2={tickX} y1={at(v)} y2={at(v)} className="stroke-muted-foreground" />
              <text
                x={tickX + BAR.labelGap}
                y={at(v)}
                dominantBaseline="middle"
                className="fill-muted-foreground text-[11px] tabular-nums"
              >
                {formatNumber(v)}
              </text>
            </g>
          ))}
      </g>
    </svg>
  )
}
