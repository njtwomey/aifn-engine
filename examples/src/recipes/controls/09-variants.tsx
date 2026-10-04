import { Button, Curve, Figure, Plot, slider, useAxis, useFigureState, variants } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Variants',
  question: 'How do I offer a family choice where each option has its own parameters?',
  explain:
    "`variants` takes one case per option, each with its own fields; the value is `{ key, values }`, typed per case. Switching away and back keeps each case's values. A preset is a button that calls `state.set(path, value)` for several fields; `state.reset()` restores the initial values.",
}

const xs = grid(-3, 3, 300)

export default function Variants() {
  // region
  const s = useFigureState({
    curve: variants(
      {
        line: { label: 'line', params: { slope: slider(-2, 2, 1, { label: 'slope' }) } },
        wave: {
          label: 'wave',
          params: { freq: slider(0.5, 4, 1, { label: 'frequency' }), amp: slider(0.2, 2, 1, { label: 'amplitude' }) },
        },
      },
      { label: 'curve', choiceLabel: 'family' },
    ),
  })
  const c = s.curve
  const f = (v: number) => (c.key === 'line' ? c.values.slope * v : c.values.amp * Math.sin(c.values.freq * v))
  // endregion
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'f(x)' })
  return (
    <Figure
      title="A family with its own parameters"
      purpose="Each family shows only its own sliders; presets set several at once."
      state={s}
      // region
      controls={
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => (s.set('curve', 'wave'), s.set('curve.freq', 3), s.set('curve.amp', 0.5))}
          >
            Fast ripple
          </Button>
          <Button size="sm" variant="outline" onClick={() => s.reset()}>
            Reset
          </Button>
        </div>
      }
      // endregion
    >
      <Plot x={x} y={y}>
        <Curve name={c.key} x={xs} y={xs.map(f)} />
      </Plot>
    </Figure>
  )
}
