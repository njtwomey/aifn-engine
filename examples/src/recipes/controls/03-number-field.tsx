import { Curve, Figure, float, int, Plot, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Number field',
  question: 'How do I take a number that a slider cannot hold: an integer count or a rate over decades?',
  explain:
    '`int` and `float` draw a typed field with step buttons either side, a menu of `suggestions` inside it, and a rail along its bottom edge to drag across the range. Bounds are `gt`, `ge`, `lt`, `le`; `scale: "log10"` steps and drags by decades. Typing a value that breaks the type or a bound (try 0.5 steps, or a rate of 2) turns the field red with a message and is not applied; `format` sets how the value is shown.',
}

export default function TypedNumbers() {
  const s = useFigureState({
    steps: int(50, { ge: 1, le: 2000, label: 'steps' }),
    rate: float(0.1, {
      gt: 0,
      lt: 1,
      scale: 'log10',
      suggestions: [0.001, 0.01, 0.1, 0.5],
      format: (v) => v.toExponential(1),
      label: 'rate',
    }),
  })
  const ts = Array.from({ length: s.steps + 1 }, (_, t) => t)
  const x = useAxis({ label: 'step', integer: true })
  const y = useAxis({ label: 'remaining', range: [0, 1] })
  return (
    <Figure title="Typed fields" purpose="(1 − rate)ᵗ over a whole number of steps." state={s}>
      <Plot x={x} y={y}>
        <Curve name="(1 − rate)ᵗ" x={ts} y={ts.map((t) => (1 - s.rate) ** t)} />
      </Plot>
    </Figure>
  )
}
