import { Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Scatter plot',
  question: 'How do I draw a scatter plot with groups?',
  explain:
    '`Points` takes `x`, `y` and an optional `group` per point: group k takes palette slot k and marker shape k, and `groupNames` labels the legend.',
}

const r = rng(7)
const centres = [
  [-2, 0],
  [2, 1],
  [0, 3],
]
const group = Array.from({ length: 150 }, (_, i) => i % 3)
const px = group.map((k) => centres[k][0] + 0.8 * r.normal())
const py = group.map((k) => centres[k][1] + 0.8 * r.normal())

export default function Scatter() {
  const x = useAxis({ label: 'x₁' })
  const y = useAxis({ label: 'x₂' })
  return (
    <Figure title="A scatter plot with groups" purpose="Three groups, each with its own colour and marker.">
      <Plot x={x} y={y}>
        <Points x={px} y={py} group={group} groupNames={['a', 'b', 'c']} />
      </Plot>
    </Figure>
  )
}
