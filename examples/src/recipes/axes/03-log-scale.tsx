import { Curve, Figure, Plot, setting, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Log scale',
  question: 'How do I use a logarithmic axis?',
  explain:
    '`log: true` on an axis model. Here a `setting` switches it, and the axis `key` changes with it so the held range refits.',
}

const ns = grid(1, 1000, 200)

export default function LogScale() {
  const state = useFigureState({ log: setting(true, 'log y') })
  const x = useAxis({ label: 'n', log: state.log, key: state.log })
  const y = useAxis({ label: 'cost', log: state.log, key: state.log })
  return (
    <Figure title="Log axes" purpose="Power laws are straight lines on log–log axes." state={state}>
      <Plot x={x} y={y}>
        <Curve name="n" x={ns} y={ns} />
        <Curve name="n log n" x={ns} y={ns.map((n) => n * Math.log2(n + 1))} />
        <Curve name="n²" x={ns} y={ns.map((n) => n * n)} />
      </Plot>
    </Figure>
  )
}
