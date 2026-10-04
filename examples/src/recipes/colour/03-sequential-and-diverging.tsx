import { Figure, Plot, Plots, Raster, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Sequential and diverging',
  question: 'When do I use a sequential or a diverging colour scale?',
  explain:
    'A magnitude takes `scale="sequential"`: one hue, light to dark. A signed value takes `scale="diverging"`, symmetric about zero, so zero is the pale midpoint.',
}

const xs = grid(-2, 2, 60)
const signed = xs.map((b) => xs.map((a) => a * b * Math.exp(-(a * a + b * b) / 2)))
const magnitude = signed.map((row) => row.map(Math.abs))

export default function SequentialAndDiverging() {
  const [a, b, c, d] = [
    useAxis({ label: 'x' }),
    useAxis({ label: 'y' }),
    useAxis({ label: 'x' }),
    useAxis({ label: 'y' }),
  ]
  return (
    <Figure title="Two colour scales" purpose="|f| on a sequential scale, f on a diverging one." defaultSize="L">
      <Plots cols={2}>
        <Plot x={a} y={b} title="sequential: |f|">
          <Raster x={xs} y={xs} z={magnitude} scale="sequential" valueLabel="|f|" />
        </Plot>
        <Plot x={c} y={d} title="diverging: f">
          <Raster x={xs} y={xs} z={signed} scale="diverging" valueLabel="f" />
        </Plot>
      </Plots>
    </Figure>
  )
}
