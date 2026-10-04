import { Curve, Figure, Plot, Plots, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Shared axes',
  question: 'How do I make two plots share an axis?',
  explain:
    'Pass the same axis model to both plots: they share its range, zoom and toolbar button. Here x is shared and each plot keeps its own y.',
}

const ts = grid(0, 20, 400)

export default function SharedAxes() {
  const t = useAxis({ label: 't' })
  const pos = useAxis({ label: 'position' })
  const vel = useAxis({ label: 'velocity' })
  return (
    <Figure title="A shared x axis" purpose="A damped oscillator's position and velocity on one time axis.">
      <Plots rows={2} toolbar>
        <Plot x={t} y={pos}>
          <Curve name="x(t)" x={ts} y={ts.map((s) => Math.exp(-0.1 * s) * Math.cos(s))} />
        </Plot>
        <Plot x={t} y={vel}>
          <Curve
            name="v(t)"
            x={ts}
            y={ts.map((s) => -Math.exp(-0.1 * s) * (Math.sin(s) + 0.1 * Math.cos(s)))}
            slot={1}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
