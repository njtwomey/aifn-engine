import { Figure, Histogram, Plot, Rug, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Rug',
  question: 'How do I mark each observation along an axis?',
  explain: '`Rug` draws a tick per value at the edge of the plot, under a histogram or a density.',
}

const r = rng(9)
const values = Array.from({ length: 60 }, () => 2 + r.normal())

export default function RugRecipe() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'count' })
  return (
    <Figure title="A rug under a histogram" purpose="Sixty values, binned and marked one by one.">
      <Plot x={x} y={y}>
        <Histogram name="count" values={values} bins={12} normalize="count" muted />
        <Rug name="values" values={values} />
      </Plot>
    </Figure>
  )
}
