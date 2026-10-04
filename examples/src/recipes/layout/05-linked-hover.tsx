import { Curve, Figure, Plot, Plots, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Linked hover',
  question: 'How do I link the hover across plots?',
  explain:
    '`hoverGroup` on `Plots` shares the hovered x between its panels, so one pointer reads all of them; a string name links plots in different grids too.',
}

const xs = grid(0, 2 * Math.PI, 300)

export default function LinkedHover() {
  const x = useAxis({ label: 'θ' })
  const a = useAxis({ label: 'sin θ' })
  const b = useAxis({ label: 'cos θ' })
  return (
    <Figure title="Linked hover" purpose="Hover either panel: both read the same θ.">
      <Plots cols={2} hoverGroup>
        <Plot x={x} y={a}>
          <Curve name="sin θ" x={xs} y={xs.map(Math.sin)} />
        </Plot>
        <Plot x={x} y={b}>
          <Curve name="cos θ" x={xs} y={xs.map(Math.cos)} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
