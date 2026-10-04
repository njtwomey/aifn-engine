import { Curve, Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Responsive panels',
  question: 'How do I put panels side by side when the figure is wide and stack them when narrow?',
  explain:
    "A Tailwind container query (`@container` and `@2xl:grid-cols-2`) follows the figure's width, not the window's, so the panels stack in a narrow column or a small figure size.",
}

const t = grid(0, 4 * Math.PI, 200)

export default function ResponsivePanels() {
  const [a, b, c, d] = [
    useAxis({ label: 'x' }),
    useAxis({ label: 'y' }),
    useAxis({ label: 't' }),
    useAxis({ label: 'x(t)' }),
  ]
  return (
    <Figure title="Panels that stack" purpose="Pick the S size: the two panels stack." hoverReadout={false}>
      {/* region */}
      <div className="@container">
        <div className="grid grid-cols-1 gap-4 @2xl:grid-cols-2">
          <Plot x={a} y={b} height={260}>
            <Points name="orbit" x={t.map((s) => Math.cos(s))} y={t.map((s) => Math.sin(2 * s))} />
          </Plot>
          <Plot x={c} y={d} height={260}>
            <Curve name="x(t)" x={t} y={t.map(Math.cos)} />
          </Plot>
        </div>
      </div>
      {/* endregion */}
    </Figure>
  )
}
