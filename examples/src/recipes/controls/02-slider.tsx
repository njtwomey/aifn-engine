import { Curve, Figure, Plot, Readout, Slider, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Slider',
  question: "How do I set a slider's range, step, arrows and a log scale?",
  explain:
    '`min`, `max` and `step` set the track (a 1-2-5 step by default); arrows either side step once (`steppable={false}` hides them) and the field takes typed values. There is no log slider: slide the exponent and `format` the value. Every move commits at once; put slow work behind `useComputed`, not a debounce.',
}

const xs = grid(0, 1, 200)

export default function SliderRecipe() {
  const [k, setK] = useState(3)
  const [exponent, setExponent] = useState(-1)
  const [moves, setMoves] = useState(0)
  const rate = 10 ** exponent
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', range: [0, 1] })
  return (
    <Figure
      title="Two sliders"
      purpose="An integer slider with arrows and a log-scale rate."
      controls={
        <>
          <Slider label="harmonic k" value={k} min={1} max={8} step={1} onChange={setK} />
          <Slider
            label="rate"
            value={exponent}
            min={-3}
            max={0}
            step={0.1}
            format={(e) => (10 ** e).toPrecision(2)}
            onChange={(e) => (setExponent(e), setMoves((m) => m + 1))}
            steppable={false}
          />
        </>
      }
      readouts={<Readout label="rate changes committed" value={moves} />}
    >
      <Plot x={x} y={y}>
        <Curve
          name="y"
          x={xs}
          y={xs.map((v) => 0.5 + 0.5 * Math.exp(-rate * 20 * v) * Math.sin(2 * Math.PI * k * v))}
        />
      </Plot>
    </Figure>
  )
}
