import { Poisson } from 'aifn-compute/probability/distributions'
import { Figure, Mass, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Probability mass',
  question: 'How do I plot the mass function of a discrete distribution?',
  explain:
    '`Mass` draws a bar per integer of a discrete `aifn-compute` distribution. An `integer` axis keeps the ticks on whole numbers.',
}

export default function MassFunction() {
  const x = useAxis({ label: 'k', integer: true })
  const y = useAxis({ label: 'P(K = k)' })
  return (
    <Figure title="A Poisson mass function" purpose="Poisson(3.5) on 0 … 12.">
      <Plot x={x} y={y}>
        <Mass name="Poisson(3.5)" dist={Poisson(3.5)} range={[0, 12]} />
      </Plot>
    </Figure>
  )
}
