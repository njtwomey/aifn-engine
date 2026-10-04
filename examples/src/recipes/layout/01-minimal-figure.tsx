import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Minimal figure',
  question: 'What is the smallest complete figure?',
  explain:
    'A `Figure` with a `title`, one `Plot` and one layer. The figure gives the chart its frame: a size the reader can change, an anchor link, a copy-data button and hover readouts.',
}

export default function MinimalFigure() {
  const x = useAxis({})
  const y = useAxis({})
  return (
    <Figure title="The smallest figure" purpose="Three points joined by a line.">
      <Plot x={x} y={y}>
        <Curve x={[0, 1, 2]} y={[1, 3, 2]} />
      </Plot>
    </Figure>
  )
}
