import { Area, Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Band between curves',
  question: 'How do I shade the region between two curves?',
  explain:
    '`Area` fills from its curve down to `base`: a constant, or a second curve given as an array. Draw the band first so the line sits on top, and give both the same `slot`.',
}

const xs = grid(0, 6, 120)
const mean = xs.map((v) => Math.sin(v))
const spread = xs.map((v) => 0.2 + 0.08 * v)

export default function FillBetween() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'f(x)' })
  return (
    <Figure title="A band between two curves" purpose="A mean with a band that widens with x.">
      <Plot x={x} y={y}>
        <Area
          name="± 2 sd"
          x={xs}
          y={mean.map((m, i) => m + 2 * spread[i])}
          base={mean.map((m, i) => m - 2 * spread[i])}
          slot={0}
          opacity={0.2}
        />
        <Curve name="mean" x={xs} y={mean} slot={0} />
      </Plot>
    </Figure>
  )
}
