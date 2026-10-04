import { Curve, Figure, Plot, Points, Segments, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Segments',
  question: 'How do I draw many short line segments, such as residuals?',
  explain:
    '`Segments` takes a list of `{ from, to }` pairs and draws them thin in one series: residuals, stems, meshes or steps.',
}

const r = rng(5)
const px = Array.from({ length: 25 }, (_, i) => i / 2.5)
const py = px.map((v) => 0.5 * v + 1 + r.normal())
const fit = (v: number) => 0.5 * v + 1

export default function SegmentsRecipe() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="Residuals as segments" purpose="Each point joined to the line by its residual.">
      <Plot x={x} y={y}>
        <Segments segments={px.map((v, i) => ({ from: [v, py[i]], to: [v, fit(v)] }))} muted />
        <Curve name="fit" x={[0, 10]} y={[fit(0), fit(10)]} emphasis />
        <Points name="data" x={px} y={py} />
      </Plot>
    </Figure>
  )
}
