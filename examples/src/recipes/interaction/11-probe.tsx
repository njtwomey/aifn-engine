import { Curve, Figure, Plot, Plots, Probe, ProbeReadout, slider, useAxis, useFigureState, useProbe } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Probe',
  question: 'How do I read values at a point that the reader moves across several plots?',
  explain:
    "A probe from `useProbe` is one position shared by every plot that draws `<Probe probe>`: dragging it on one moves it on all. `at` marks the curve's value there and `ProbeReadout` reports the numbers.",
}

const xs = grid(-3, 3, 300)
const f = (v: number) => Math.exp((-v * v) / 2)
const df = (v: number) => -v * f(v)

export default function ProbeRecipe() {
  const s = useFigureState({ x0: slider(-3, 3, 0.8, { onChart: true }) })
  const probe = useProbe({ x: s.bind('x0'), label: 'x₀' })
  const x = useAxis({ label: 'x' })
  const [y1, y2] = [useAxis({ label: 'f' }), useAxis({ label: 'f′' })]
  return (
    <Figure
      title="A probe on two plots"
      purpose="f and its derivative read at the same x₀."
      state={s}
      readouts={<ProbeReadout probe={probe} values={{ 'f(x₀)': f(s.x0), 'f′(x₀)': df(s.x0) }} />}
      caption="Drag the probe on either plot."
    >
      <Plots rows={2}>
        <Plot x={x} y={y1}>
          <Curve name="f" x={xs} y={xs.map(f)} />
          <Probe probe={probe} at={f(s.x0)} />
        </Plot>
        <Plot x={x} y={y2}>
          <Curve name="f′" x={xs} y={xs.map(df)} slot={1} />
          <Probe probe={probe} at={df(s.x0)} />
        </Plot>
      </Plots>
    </Figure>
  )
}
