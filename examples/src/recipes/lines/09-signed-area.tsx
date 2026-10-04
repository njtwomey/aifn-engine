import { Figure, Plot, SignedArea, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Signed area',
  question: 'How do I shade area above and below zero differently?',
  explain:
    '`SignedArea` fills the parts above zero and below zero in the two ends of the diverging scale and labels the net area.',
}

const xs = grid(0, 2 * Math.PI, 200)

export default function SignedAreaRecipe() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'f(x)' })
  return (
    <Figure title="Signed area" purpose="f(x) = sin x + 0.3: more area above zero than below.">
      <Plot x={x} y={y}>
        <SignedArea name="f" x={xs} y={xs.map((v) => Math.sin(v) + 0.3)} />
      </Plot>
    </Figure>
  )
}
