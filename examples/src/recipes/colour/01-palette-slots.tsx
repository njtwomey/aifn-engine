import { Curve, Figure, Plot, Switch, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Palette slots',
  question: "How do I keep a series' colour when others are toggled?",
  explain:
    'Layers take categorical palette slots in order by default. When series can come and go, pass `slot` explicitly, so each entity keeps its colour; slots are never assigned by rank or cycled.',
}

const xs = grid(0, 1, 100)
const powers = [1, 2, 3, 4]

export default function PaletteSlots() {
  const [shown, setShown] = useState([true, false, true, true])
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'xᵏ', range: [0, 1] })
  return (
    <Figure
      title="Fixed slots"
      purpose="Switch series off: the others keep their colours."
      controls={powers.map((k, i) => (
        <Switch
          key={k}
          label={`x^${k}`}
          checked={shown[i]}
          onChange={(on) => setShown(shown.map((v, j) => (j === i ? on : v)))}
        />
      ))}
    >
      <Plot x={x} y={y}>
        {powers.map((k, i) => shown[i] && <Curve key={k} name={`x^${k}`} x={xs} y={xs.map((v) => v ** k)} slot={i} />)}
      </Plot>
    </Figure>
  )
}
