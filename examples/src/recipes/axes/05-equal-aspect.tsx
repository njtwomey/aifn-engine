import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Equal aspect',
  question: 'How do I make a circle look like a circle?',
  explain:
    '`equal: x` on the y axis model gives both axes the same length per unit, whatever the ranges. Use it whenever a shape or an angle matters.',
}

const t = grid(0, 2 * Math.PI, 120)

export default function EqualAspect() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', equal: x })
  return (
    <Figure title="Equal units" purpose="A unit circle and an ellipse with axes 3 and 1.">
      <Plot x={x} y={y}>
        <Curve name="circle" x={t.map(Math.cos)} y={t.map(Math.sin)} />
        <Curve name="ellipse" x={t.map((s) => 3 * Math.cos(s))} y={t.map(Math.sin)} />
      </Plot>
    </Figure>
  )
}
