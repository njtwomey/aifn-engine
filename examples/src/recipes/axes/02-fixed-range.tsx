import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Fixed range',
  question: 'How do I fix an axis range, or one end of it?',
  explain:
    "`range: [lo, hi]` fixes both ends (zoom still works); leave an end `undefined` to fit only that end, e.g. `[0, undefined]` for a count that starts at zero. `nice: false` keeps the data's own ends.",
}

const xs = grid(0, 10, 200)

export default function FixedRange() {
  const x = useAxis({ label: 't', range: [0, 10], nice: false })
  const y = useAxis({ label: 'growth', range: [0, undefined] })
  return (
    <Figure title="A fixed range" purpose="x fixed to [0, 10]; y starts at zero and fits the top.">
      <Plot x={x} y={y}>
        <Curve name="logistic" x={xs} y={xs.map((t) => 50 / (1 + Math.exp(-(t - 5))))} />
      </Plot>
    </Figure>
  )
}
