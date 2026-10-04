import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Line chart',
  question: 'How do I draw a line chart?',
  explain:
    'A `Plot` takes an x and a y axis model from `useAxis`; each `Curve` inside it is one line through `x` and `y` arrays. Hovering reads the values under the pointer.',
}

const xs = grid(0, 4 * Math.PI, 200)

export default function LineChart() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="A line chart" purpose="sin x and cos x over two periods.">
      <Plot x={x} y={y}>
        <Curve name="sin x" x={xs} y={xs.map(Math.sin)} />
        <Curve name="cos x" x={xs} y={xs.map(Math.cos)} />
      </Plot>
    </Figure>
  )
}
