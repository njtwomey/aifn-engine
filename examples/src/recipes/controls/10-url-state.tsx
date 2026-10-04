import { choice, Curve, Figure, Plot, Readout, slider, toEntries, useAxis, useFigureState } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'State in the URL',
  question: 'How do I make a link that reproduces the figure as the reader left it?',
  explain:
    'A figure with `state` writes its non-default values into the URL next to its anchor, and reads them back on load; the reset button clears them. `toEntries(state.schema, state.json)` lists what is stored.',
}

const xs = grid(0, 10, 300)

export default function UrlState() {
  const s = useFigureState({
    freq: slider(0.5, 3, 1, { label: 'frequency' }),
    shape: choice(['sin', 'cos'], 'sin', { label: 'shape' }),
  })
  const stored = toEntries(s.schema, s.json)
    .map(([k, v]) => `${k}=${v}`)
    .join(' & ')
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'f(t)' })
  return (
    <Figure
      title="State kept in the URL"
      purpose="Change a control, then reload: the figure comes back as it was."
      state={s}
      readouts={<Readout label="in the URL" value={stored || 'nothing (all defaults)'} />}
    >
      <Plot x={x} y={y}>
        <Curve name={s.shape} x={xs} y={xs.map((t) => (s.shape === 'sin' ? Math.sin : Math.cos)(s.freq * t))} />
      </Plot>
    </Figure>
  )
}
