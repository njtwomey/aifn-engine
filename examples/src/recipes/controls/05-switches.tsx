import { Curve, Figure, Plot, setting, toggle, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Toggle and setting',
  question: 'What is the difference between a toggle and a setting?',
  explain:
    'A `toggle` is a button that reveals an ingredient the figure is about (here the derivative); a `setting` is a plain switch for presentation (here a log axis). Both are booleans in the state.',
}

const xs = grid(0.1, 5, 200)

export default function Switches() {
  const s = useFigureState({
    derivative: toggle(false, 'show f′'),
    log: setting(false, 'log y'),
  })
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'value', log: s.log, key: s.log })
  return (
    <Figure title="A reveal and a setting" purpose="f(x) = x³, with its derivative on request." state={s}>
      <Plot x={x} y={y}>
        <Curve name="f = x³" x={xs} y={xs.map((v) => v ** 3)} />
        {s.derivative && <Curve name="f′ = 3x²" x={xs} y={xs.map((v) => 3 * v * v)} slot={1} />}
      </Plot>
    </Figure>
  )
}
