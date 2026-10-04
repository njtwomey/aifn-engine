import { Curve, Figure, Handle, Plot, slider, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Figure state',
  question: "How do I declare a figure's controls in one place?",
  explain:
    '`useFigureState` takes a schema of fields and returns typed values; `<Figure state>` draws the controls, a reset button and keeps changes in the URL. `onChart` fields draw no control: `state.handle(name)` binds them to a handle.',
}

const xs = grid(-4, 4, 300)

export default function FigureState() {
  const s = useFigureState({
    width: slider(0.2, 2, 1, { label: 'width w' }),
    centre: slider(-3, 3, 0, { onChart: true, label: 'centre' }),
  })
  const x = useAxis({ label: 'x', range: [-4, 4] })
  const y = useAxis({ label: 'bump', range: [0, 1.05] })
  return (
    <Figure
      title="State with a handle"
      purpose="A slider sets the width; the centre is dragged on the chart."
      state={s}
      caption="Drag the vertical line to move the centre."
    >
      <Plot x={x} y={y}>
        <Curve name="bump" x={xs} y={xs.map((v) => Math.exp(-(((v - s.centre) / s.width) ** 2)))} />
        <Handle {...s.handle('centre', { label: 'centre' })} />
      </Plot>
    </Figure>
  )
}
