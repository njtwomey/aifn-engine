import { Normal, StudentT } from 'aifn-compute/probability/distributions'
import { Density, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Probability density',
  question: 'How do I plot the density of a distribution?',
  explain:
    '`Density` takes an `aifn-compute` distribution and evaluates it over `range` (default: where its mass is). `fill` shades under the curve.',
}

export default function DensityRecipe() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'p(x)' })
  return (
    <Figure title="Two densities" purpose="A standard normal against a Student t with 2 degrees of freedom.">
      <Plot x={x} y={y}>
        <Density name="Normal(0, 1)" dist={Normal(0, 1)} range={[-6, 6]} fill={0.15} />
        <Density name="Student t, ν = 2" dist={StudentT(2, 0, 1)} range={[-6, 6]} dashed />
      </Plot>
    </Figure>
  )
}
