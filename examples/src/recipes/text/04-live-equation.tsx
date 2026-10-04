import { Curve, Equation, Figure, live, Plot, slider, tex, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Live equation',
  question: 'How do I show an equation with the current values substituted?',
  explain:
    "The `tex` template substitutes numbers into TeX and `live(value, { strong })` highlights them; pass the `Equation` as the figure's `equation` and it sits between the controls and the chart.",
}

const xs = grid(-3, 3, 200)

export default function LiveEquation() {
  const s = useFigureState({ a: slider(-2, 2, 0.5, { label: 'a' }), b: slider(-2, 2, -1, { label: 'b' }) })
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', range: [-8, 8] })
  return (
    <Figure
      title="A line with its equation"
      purpose="The equation shows the slope and intercept as they are."
      state={s}
      equation={
        <Equation>{tex`y = ${live(s.a, { strong: true })}\,x ${s.b < 0 ? '-' : '+'} ${Math.abs(s.b)}`}</Equation>
      }
    >
      <Plot x={x} y={y}>
        <Curve name="y" x={xs} y={xs.map((v) => s.a * v + s.b)} />
      </Plot>
    </Figure>
  )
}
