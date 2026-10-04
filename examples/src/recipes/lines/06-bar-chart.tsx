import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Bar chart',
  question: 'How do I draw a bar chart over categories?',
  explain:
    'A categorical axis (`categories`) puts category k at position k; `Bars` draws a bar at each `x` with height `y`. Two `Bars` layers with a `width` and an offset sit side by side.',
}

const fruit = ['apples', 'pears', 'plums', 'figs']
const at = fruit.map((_, k) => k)

export default function BarChart() {
  const x = useAxis({ categories: fruit })
  const y = useAxis({ label: 'kg sold', range: [0, undefined] })
  return (
    <Figure title="A grouped bar chart" purpose="Sales of four fruits in two shops.">
      <Plot x={x} y={y}>
        <Bars name="north shop" x={at.map((k) => k - 0.2)} y={[12, 7, 9, 3]} width={0.4} />
        <Bars name="south shop" x={at.map((k) => k + 0.2)} y={[8, 10, 4, 6]} width={0.4} />
      </Plot>
    </Figure>
  )
}
