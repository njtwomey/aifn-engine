import { Figure, Plot, Segmented, useAxis, VectorField, type ArrowStyle } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Vector field',
  question: 'How do I draw a vector field?',
  explain:
    '`VectorField` samples a function `(x, y) => [u, v]` on a grid and draws an arrow at each point. By default every arrow is the same length and its colour shows the magnitude, from blue at the smallest to red at the largest, so direction stays readable where the field is weak; `length="magnitude"` scales the arrows instead. `arrow` picks how each one is drawn. Use `equal` axes so angles are true. For single arrows of your own, use `Vectors`.',
}

// A swirl that also drifts outwards: slow near the origin, fast at the edge
const swirl = (x: number, y: number) => [-y + 0.3 * x, x + 0.3 * y] as const

const STYLES: ArrowStyle[] = ['arrow', 'triangle', 'line', 'dot']

export default function VectorFieldRecipe() {
  const [arrow, setArrow] = useState<ArrowStyle>('arrow')
  const [length, setLength] = useState<'unit' | 'magnitude'>('unit')
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', equal: x })
  return (
    <Figure
      title="A vector field"
      purpose="Unit arrows show the direction; colour shows the magnitude."
      controls={
        <>
          <Segmented label="arrow" value={arrow} onChange={setArrow} options={STYLES} />
          <Segmented label="length" value={length} onChange={setLength} options={['unit', 'magnitude']} />
        </>
      }
      caption="The field (−y + 0.3x, x + 0.3y). With unit lengths, blue arrows are the slowest and red the fastest."
    >
      <Plot x={x} y={y}>
        <VectorField field={swirl} x={[-2, 2]} y={[-2, 2]} n={15} arrow={arrow} length={length} />
      </Plot>
    </Figure>
  )
}
