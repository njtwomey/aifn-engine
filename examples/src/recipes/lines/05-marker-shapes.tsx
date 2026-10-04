import { Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Colour and shape',
  question: 'How do I show two variables with colour and marker shape?',
  explain:
    'Colour is the `group`; `shape` (a number per point) gives a second variable its own markers, named by `shapeNames`. Scatter charts keep to three colours, so shape carries the rest.',
}

const r = rng(11)
const n = 120
const px = Array.from({ length: n }, () => r.normal())
const py = px.map((v) => 0.6 * v + 0.6 * r.normal())
const group = px.map((v) => (v > 0 ? 1 : 0))
const shape = px.map((_, i) => i % 2)

export default function MarkerShapes() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="Colour and marker shape" purpose="Colour is the sign of x; shape is the fold.">
      <Plot x={x} y={y}>
        <Points
          x={px}
          y={py}
          group={group}
          groupNames={['x < 0', 'x > 0']}
          shape={shape}
          shapeNames={['fold 1', 'fold 2']}
        />
      </Plot>
    </Figure>
  )
}
