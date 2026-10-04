import { Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Categorical axis',
  question: 'How do I label an axis with category names?',
  explain:
    '`categories` makes an axis categorical: category k sits at position k and is labelled with its name, so any layer can use it, not only bars.',
}

const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

export default function CategoricalAxis() {
  const x = useAxis({ categories: days })
  const y = useAxis({ label: 'minutes late' })
  return (
    <Figure title="A categorical x axis" purpose="Each day's delays as points over its name.">
      <Plot x={x} y={y}>
        <Points name="trains" x={[0, 0, 1, 2, 2, 2, 3, 4, 4]} y={[2, 5, 1, 7, 3, 12, 0, 4, 6]} />
      </Plot>
    </Figure>
  )
}
