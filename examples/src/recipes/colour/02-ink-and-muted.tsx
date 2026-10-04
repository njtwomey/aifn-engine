import { Curve, Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid, rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Ink and muted',
  question: 'How do I make one mark stand out and push others back?',
  explain:
    '`emphasis` draws in ink (the truth, a centroid) and takes no palette slot; `muted` draws in the chrome colour for background marks. Neither shifts the slots of the other layers.',
}

const xs = grid(-3, 3, 100)
const r = rng(21)
const px = Array.from({ length: 80 }, () => 6 * r.uniform() - 3)
const py = px.map((v) => Math.tanh(v) + 0.25 * r.normal())

export default function InkAndMuted() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="Ink, slots and muted marks" purpose="Data muted, a fit in slot 0, the truth in ink.">
      <Plot x={x} y={y}>
        <Points name="data" x={px} y={py} muted />
        <Curve name="fit" x={xs} y={xs.map((v) => 0.9 * Math.tanh(1.1 * v))} slot={0} />
        <Curve name="truth" x={xs} y={xs.map(Math.tanh)} emphasis dashed />
      </Plot>
    </Figure>
  )
}
