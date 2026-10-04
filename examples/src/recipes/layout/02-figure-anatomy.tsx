import { Curve, Figure, formatNumber, Plot, Readout, Slider, useAxis, useParam } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Figure anatomy',
  question: 'What goes around a chart: title, controls, readouts and caption?',
  explain:
    'A `Figure` lays out its slots in a fixed order: `title` and `purpose` (its one point), `description`, `controls`, the chart area, `readouts` and `caption` (what to change, what to watch). `defaultSize` picks S, M or L.',
}

const xs = grid(0, 10, 200)

export default function FigureAnatomy() {
  const k = useParam(0.5, { min: 0.1, max: 2 })
  const half = Math.LN2 / k.value
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'N(t)', range: [0, 1] })
  return (
    <Figure
      title="Exponential decay"
      purpose="N(t) = e^(−kt) halves every ln 2 / k."
      description="The curve is the fraction left at time t."
      controls={<Slider label="rate k" param={k} />}
      readouts={<Readout label="half-life" value={formatNumber(half)} />}
      caption="Raise k: the curve falls faster and the half-life shrinks."
      defaultSize="M"
    >
      <Plot x={x} y={y}>
        <Curve name="N(t)" x={xs} y={xs.map((t) => Math.exp(-k.value * t))} />
      </Plot>
    </Figure>
  )
}
