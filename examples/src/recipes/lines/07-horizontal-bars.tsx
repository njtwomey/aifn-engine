import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Horizontal bars',
  question: 'How do I draw bars sideways?',
  explain:
    '`orient="y"` lays bars along the y axis: `x` is still the position and `y` the length, so put the categories on the y axis.',
}

const tasks = ['parse', 'plan', 'execute', 'report']

export default function HorizontalBars() {
  const x = useAxis({ label: 'seconds', range: [0, undefined] })
  const y = useAxis({ categories: tasks })
  return (
    <Figure title="Horizontal bars" purpose="Time spent in each stage.">
      <Plot x={x} y={y}>
        <Bars name="time" x={[0, 1, 2, 3]} y={[1.2, 3.4, 7.9, 0.8]} orient="y" />
      </Plot>
    </Figure>
  )
}
