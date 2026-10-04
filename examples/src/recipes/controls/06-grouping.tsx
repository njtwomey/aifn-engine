import { Curve, Figure, Plot, row, slider, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Groups of controls',
  question: 'How do I group controls into numbered, collapsible rows?',
  explain:
    "Each `row(label, fields)` is one labelled row; number the labels (`1 · …`, `2 · …`) when the order is the order to try them. `collapsible` with `defaultCollapsed` tucks away a row of rarely used fields. Values nest under the row's key.",
}

const ts = grid(0, 10, 400)

export default function Grouping() {
  const s = useFigureState({
    signal: row('1 · signal', {
      freq: slider(0.5, 3, 1, { label: 'frequency' }),
      amp: slider(0.2, 2, 1, { label: 'amplitude' }),
    }),
    damping: row('2 · damping', { rate: slider(0, 1, 0.2, { label: 'rate' }) }),
    style: row(
      '3 · advanced',
      { offset: slider(-1, 1, 0, { label: 'offset' }) },
      { collapsible: true, defaultCollapsed: true },
    ),
  })
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'x(t)', hold: 'union' })
  return (
    <Figure title="Numbered rows" purpose="A damped wave set in three groups." state={s}>
      <Plot x={x} y={y}>
        <Curve
          name="x(t)"
          x={ts}
          y={ts.map((t) => s.style.offset + s.signal.amp * Math.exp(-s.damping.rate * t) * Math.sin(s.signal.freq * t))}
        />
      </Plot>
    </Figure>
  )
}
