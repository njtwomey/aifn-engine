import { Annotation, Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Annotations',
  question: 'How do I label a point or a line on a chart?',
  explain:
    '`Annotation` with `at` marks a labelled point; with `x` (or `y`) alone it draws a labelled vertical (or horizontal) line, `dashed` if it is a reference.',
}

const xs = grid(-1, 3, 200)
const f = (v: number) => v ** 3 - 3 * v ** 2 + 2

export default function Annotations() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'f(x)' })
  return (
    <Figure title="Annotated extrema" purpose="f(x) = x³ − 3x² + 2 with its turning points marked.">
      <Plot x={x} y={y}>
        <Curve name="f" x={xs} y={xs.map(f)} />
        <Annotation at={[0, f(0)]} text="local max" />
        <Annotation at={[2, f(2)]} text="local min" />
        <Annotation x={1} text="inflection" dashed />
        <Annotation y={0} dashed />
      </Plot>
    </Figure>
  )
}
