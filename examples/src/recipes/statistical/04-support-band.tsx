import { Beta } from 'aifn-compute/probability/distributions'
import { Density, Figure, Plot, SupportBand, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Support',
  question: 'How do I show where a distribution is defined?',
  explain:
    '`SupportBand` marks a support interval along the axis, with open or closed ends; give it `dist` or an `interval`. `support` on the axis makes those ends hard limits for zoom.',
}

export default function SupportBandRecipe() {
  const x = useAxis({ label: 'p', support: { lower: 0, upper: 1 } })
  const y = useAxis({ label: 'density' })
  return (
    <Figure title="A support band" purpose="Beta(2, 5) lives on [0, 1].">
      <Plot x={x} y={y}>
        <SupportBand dist={Beta(2, 5)} shade />
        <Density name="Beta(2, 5)" dist={Beta(2, 5)} />
      </Plot>
    </Figure>
  )
}
