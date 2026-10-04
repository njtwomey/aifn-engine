import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Line widths',
  question: 'How do I draw thin and thick lines?',
  explain:
    '`width` sets a stroke in pixels (default 2); one colour (`slot`) keeps the comparison about width alone. `thin` with `muted` draws a fine, translucent line for many overlapping draws such as sample paths, and `emphasis` draws the line that matters in ink on top.',
}

const xs = grid(0, 2 * Math.PI, 160)
const draws = Array.from({ length: 30 }, (_, k) =>
  xs.map((x) => Math.sin(x + 0.15 * Math.sin(3 * k)) * (0.8 + 0.01 * k)),
)

export default function LineWidths() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="Thin and thick lines" purpose="Stroke widths from 0.5 to 6 px, thin draws, and one emphasised line.">
      <Plot x={x} y={y}>
        {draws.map((d, k) => (
          <Curve key={k} x={xs} y={d} thin muted silent />
        ))}
        {[0.5, 1, 2, 4, 6].map((w, i) => (
          <Curve key={w} name={`width ${w}`} x={xs} y={xs.map((v) => Math.cos(v) - 1.5 - 0.6 * i)} width={w} slot={0} />
        ))}
        <Curve name="emphasis" x={xs} y={xs.map(Math.sin)} emphasis width={3} />
      </Plot>
    </Figure>
  )
}
