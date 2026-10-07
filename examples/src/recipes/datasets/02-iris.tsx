import { choice, Figure, Plot, Points, row, useAxis, useFigureState } from 'aifn-render'
import { toRows } from 'aifn-compute/foundation/tensor'
import { iris } from 'aifn-methods/data/real'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Iris',
  question: 'What does Fisher’s Iris data look like?',
  explain:
    '`iris()` gives 150 flowers as a 150 × 4 `x` (sepal and petal length and width, cm) and the species as `y`. Pick any two measurements: the petals separate setosa at once; versicolor and virginica overlap.',
}

const data = iris()
const rows = toRows(data.x)
const names = data.meta.featureNames!
const species = Array.from(data.y!.data as ArrayLike<number>)

export default function Iris() {
  const s = useFigureState({
    axes: row('axes', {
      across: choice(names, 'petal length', { label: 'x' }),
      up: choice(names, 'petal width', { label: 'y' }),
    }),
  })
  const i = names.indexOf(s.axes.across)
  const j = names.indexOf(s.axes.up)
  const x = useAxis({ label: `${s.axes.across} (cm)` })
  const y = useAxis({ label: `${s.axes.up} (cm)` })
  return (
    <Figure title="Iris" purpose="Two of the four measurements of 150 flowers, by species." state={s}>
      <Plot x={x} y={y}>
        <Points x={rows.map((r) => r[i])} y={rows.map((r) => r[j])} group={species} groupNames={data.meta.labelNames} />
      </Plot>
    </Figure>
  )
}
