import { Columns, Curve, Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Side-by-side panels',
  question: 'How do I put titled panels side by side, each with a footer?',
  explain:
    "`Columns` takes `panels`, each a `title`, a `body` that fills the frame's height and an optional `footer`; `widths` sets relative widths.",
}

const t = grid(0, 2 * Math.PI, 100)

export default function ColumnsRecipe() {
  const [a, b, c, d] = [
    useAxis({ label: 'x' }),
    useAxis({ label: 'y' }),
    useAxis({ label: 't' }),
    useAxis({ label: 'r' }),
  ]
  return (
    <Figure title="Two panels" purpose="A curve in the plane and its radius over time.">
      <Columns
        widths={[1, 1.5]}
        panels={[
          {
            title: 'plane',
            body: (
              <Plot x={a} y={b}>
                <Curve
                  name="path"
                  x={t.map((s) => Math.cos(s) * (1 + 0.3 * Math.cos(5 * s)))}
                  y={t.map((s) => Math.sin(s) * (1 + 0.3 * Math.cos(5 * s)))}
                />
              </Plot>
            ),
            footer: 'r(t) = 1 + 0.3 cos 5t',
          },
          {
            title: 'radius',
            body: (
              <Plot x={c} y={d}>
                <Points name="r" x={t} y={t.map((s) => 1 + 0.3 * Math.cos(5 * s))} />
              </Plot>
            ),
          },
        ]}
      />
    </Figure>
  )
}
