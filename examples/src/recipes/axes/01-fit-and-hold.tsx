import { Curve, Figure, Plot, Plots, slider, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Fit or hold',
  question: 'How do I stop the axes jumping while a slider moves?',
  explain:
    'By default an axis refits to the data on every change. `hold: "union"` keeps a range that only grows; `hold: "initial"` keeps the first fit. A changed `key` refits a held axis.',
}

const xs = grid(-3, 3, 200)

export default function FitAndHold() {
  const state = useFigureState({ a: slider(0.2, 3, 1, { label: 'amplitude a' }) })
  const x = useAxis({ label: 'x' })
  const fitted = useAxis({ label: 'refits' })
  const held = useAxis({ label: 'held', hold: 'union' })
  const ys = xs.map((v) => state.a * Math.sin(2 * v))
  return (
    <Figure
      title="Fitted and held axes"
      purpose="The same curve on an axis that refits and one that holds."
      state={state}
    >
      <Plots cols={2}>
        <Plot x={x} y={fitted} title="refits">
          <Curve name="a sin 2x" x={xs} y={ys} />
        </Plot>
        <Plot x={x} y={held} title="hold: union">
          <Curve name="a sin 2x" x={xs} y={ys} />
        </Plot>
      </Plots>
    </Figure>
  )
}
