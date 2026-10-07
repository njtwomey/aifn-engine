import { Curve, Figure, Plot, Plots, Points, useAxis } from 'aifn-render'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { anscombe } from 'aifn-methods/data/real'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Anscombe’s quartet',
  question: 'What does Anscombe’s quartet look like?',
  explain:
    '`anscombe()` gives four datasets of eleven points, each with the same means, variances, correlation and least-squares line, y = 3 + 0.5x, drawn in every panel. Only a plot tells them apart: a line, a curve, an outlier, and a single point setting the slope.',
}

const sets = anscombe()
const line = [4, 19]

export default function Anscombe() {
  const x = useAxis({ label: 'x', range: [3, 20] })
  const ys = [
    useAxis({ range: [2, 14] }),
    useAxis({ range: [2, 14] }),
    useAxis({ range: [2, 14] }),
    useAxis({ range: [2, 14] }),
  ]
  return (
    <Figure title="Anscombe’s quartet" purpose="Four datasets with the same summary statistics." defaultSize="L">
      <Plots rows={2} cols={2}>
        {sets.map((d, k) => (
          <Plot key={d.meta.name} title={d.meta.name} x={x} y={ys[k]}>
            <Curve name="y = 3 + 0.5x" x={line} y={line.map((v) => 3 + 0.5 * v)} muted />
            <Points name="points" x={toFlat(d.x)} y={toFlat(d.y!)} slot={0} />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}
