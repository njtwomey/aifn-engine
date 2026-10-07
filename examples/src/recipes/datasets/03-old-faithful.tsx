import { Figure, Plot, Points, useAxis } from 'aifn-render'
import { toRows } from 'aifn-compute/foundation/tensor'
import { oldFaithful } from 'aifn-methods/data/real'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Old Faithful',
  question: 'What does the Old Faithful geyser data look like?',
  explain:
    '`oldFaithful()` gives 272 eruptions as a 272 × 2 `x`: how long each lasted and the wait until the next, in minutes. There are no labels, but two clusters are plain: short eruptions come sooner. The usual first example for a two-component mixture.',
}

const data = oldFaithful()
const rows = toRows(data.x)

export default function OldFaithful() {
  const x = useAxis({ label: 'eruption duration (min)' })
  const y = useAxis({ label: 'waiting time to the next (min)' })
  return (
    <Figure title="Old Faithful" purpose="Duration of each eruption against the wait until the next.">
      <Plot x={x} y={y}>
        <Points name="eruptions" x={rows.map((r) => r[0])} y={rows.map((r) => r[1])} />
      </Plot>
    </Figure>
  )
}
