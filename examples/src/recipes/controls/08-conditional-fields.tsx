import { choice, Curve, Figure, Plot, row, slider, useAxis, useFigureState, when } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Conditional fields',
  question: 'How do I show a control only when it applies?',
  explain:
    '`row(label, fields)` draws its fields on one labelled row and nests their values (`s.shape.kind`). `when(key, value)` shows a field only while a sibling has that value; a hidden field is left out of the URL.',
}

const xs = grid(-3, 3, 300)

export default function RowsAndConditions() {
  const s = useFigureState({
    shape: row('1 · shape', {
      kind: choice(['gaussian', 'box'], 'gaussian', { label: 'kernel' }),
      width: slider(0.2, 2, 1, { label: 'width' }),
      tail: slider(0, 0.5, 0, { label: 'tail', when: when('kind', 'gaussian') }),
    }),
  })
  const { kind, width, tail } = s.shape
  const k = (v: number) =>
    kind === 'box' ? (Math.abs(v) < width ? 1 : 0) : Math.exp(-((v / width) ** 2)) + tail / (1 + v * v)
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'k(x)', hold: 'union' })
  return (
    <Figure title="A row of controls" purpose="The tail control appears only for the Gaussian kernel." state={s}>
      <Plot x={x} y={y}>
        <Curve name={kind} x={xs} y={xs.map(k)} />
      </Plot>
    </Figure>
  )
}
