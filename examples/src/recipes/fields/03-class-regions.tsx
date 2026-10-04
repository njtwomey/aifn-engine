import { Figure, Plot, Points, Raster, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Class regions',
  question: 'How do I colour regions by class and draw their boundary?',
  explain:
    '`scale="categorical"` colours cell value k in slot k, matching `Points` with `group` k; `boundary` draws the borders between classes in ink.',
}

const centres = [
  [-1, -1],
  [1.2, -0.5],
  [0, 1.3],
]
const xs = grid(-3, 3, 120)
const ys = grid(-3, 3, 120)
const nearest = (a: number, b: number) => {
  const d = centres.map(([cx, cy]) => (a - cx) ** 2 + (b - cy) ** 2)
  return d.indexOf(Math.min(...d))
}
const z = ys.map((b) => xs.map((a) => nearest(a, b)))

export default function ClassRegions() {
  const x = useAxis({ label: 'x₁' })
  const y = useAxis({ label: 'x₂', equal: x })
  return (
    <Figure title="Class regions" purpose="Each cell coloured by its nearest centre.">
      <Plot x={x} y={y}>
        <Raster x={xs} y={ys} z={z} scale="categorical" categoryNames={['a', 'b', 'c']} fillOpacity={0.35} boundary />
        <Points
          name="centres"
          x={centres.map((c) => c[0])}
          y={centres.map((c) => c[1])}
          group={[0, 1, 2]}
          groupNames={['a', 'b', 'c']}
        />
      </Plot>
    </Figure>
  )
}
