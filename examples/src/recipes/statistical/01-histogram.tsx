import { Figure, Histogram, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Histogram',
  question: 'How do I draw a histogram of samples?',
  explain:
    '`Histogram` bins `values` itself: `bins` is a count, `{ width }`, a rule such as `freedman-diaconis`, or the edges. `normalize="density"` scales it to integrate to one.',
}

const r = rng(2)
const samples = Array.from({ length: 2000 }, () => (r.uniform() < 0.3 ? -2 : 1) + 0.7 * r.normal())

export default function HistogramRecipe() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'density' })
  return (
    <Figure title="A histogram" purpose="2000 draws from a mixture of two normals.">
      <Plot x={x} y={y}>
        <Histogram name="samples" values={samples} bins="freedman-diaconis" normalize="density" />
      </Plot>
    </Figure>
  )
}
