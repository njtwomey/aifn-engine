import { Curve, Figure, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Readouts',
  question: 'How do I report numbers under a chart?',
  explain:
    '`readouts` takes `Readout` elements, or a record of labelled groups. `formatNumber` gives every figure the same number format.',
}

const xs = grid(0, 2, 200)

export default function Readouts() {
  const s = useFigureState({ k: slider(0.5, 4, 2, { label: 'k' }) })
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'xᵏ' })
  return (
    <Figure
      title="Grouped readouts"
      purpose="Values of xᵏ at two points and its integral on [0, 2]."
      state={s}
      readouts={{
        values: (
          <>
            <Readout label="f(1)" value={formatNumber(1)} />
            <Readout label="f(2)" value={formatNumber(2 ** s.k)} />
          </>
        ),
        integral: <Readout label="∫₀² f" value={formatNumber(2 ** (s.k + 1) / (s.k + 1))} />,
      }}
    >
      <Plot x={x} y={y}>
        <Curve name="xᵏ" x={xs} y={xs.map((v) => v ** s.k)} />
      </Plot>
    </Figure>
  )
}
