import { Figure, Plot, useAxis, Vectors, type Vector } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Vector field',
  question: 'How do I draw arrows, such as a vector field?',
  explain:
    '`Vectors` draws arrows `{ from, to }`, clipped to the plot with a chevron where they leave it. Use `head` for smaller arrowheads in a dense field and `equal` axes so angles are true.',
}

const at = grid(-2, 2, 11)
// The rotation field (−y, x), scaled to fit between grid points
const field: Vector[] = at.flatMap((a) =>
  at.map((b): Vector => ({ from: [a, b], to: [a - 0.15 * b, b + 0.15 * a], head: 6 })),
)

export default function VectorField() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', equal: x })
  return (
    <Figure title="A vector field" purpose="The rotation field (−y, x).">
      <Plot x={x} y={y}>
        <Vectors vectors={field} />
      </Plot>
    </Figure>
  )
}
