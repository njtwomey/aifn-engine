import { Bars, Curve, Dashboard, DashboardCell, DashboardRow, Figure, Plot, Raster, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Dashboard',
  question: 'How do I mix plots of different shapes in one figure?',
  explain:
    '`Dashboard` splits the chart area into rows of cells by `ratio`; a cell with `aspect="square"` takes its row\'s height and a matching width. Below `stackBelow` pixels every cell stacks.',
}

const xs = grid(0, 1, 40)
const z = xs.map((b) => xs.map((a) => Math.sin(6 * a) * Math.cos(4 * b)))

export default function DashboardRecipe() {
  const [u, v, w, c, d, e] = [useAxis({}), useAxis({}), useAxis({}), useAxis({}), useAxis({}), useAxis({})]
  return (
    <Figure title="A dashboard" purpose="A wide curve, a square heatmap and a bar chart." defaultSize="L">
      <Dashboard>
        <DashboardRow ratio={1.2}>
          <DashboardCell>
            <Plot x={u} y={v}>
              <Curve name="signal" x={xs} y={xs.map((t) => Math.sin(12 * t) * t)} />
            </Plot>
          </DashboardCell>
          <DashboardCell aspect="square">
            <Plot x={w} y={c}>
              <Raster x={xs} y={xs} z={z} scale="diverging" colorBar={false} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow>
          <DashboardCell>
            <Plot x={d} y={e}>
              <Bars name="share" x={[1, 2, 3, 4, 5]} y={[5, 3, 6, 2, 4]} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
