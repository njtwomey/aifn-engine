import { Figure, Plot, Shapes, useAxis, type FilledShape } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Filled shapes',
  question: 'How do I fill polygons, with holes?',
  explain:
    "`Shapes` fills each shape's `contours` as one SVG path, so an inner contour leaves a hole. `tone` is ink, muted or a palette slot.",
}

const circle = (r: number, cx = 0, cy = 0) =>
  grid(0, 2 * Math.PI, 64).map((t) => [cx + r * Math.cos(t), cy + r * Math.sin(t)] as const)
const shapes: FilledShape[] = [
  { contours: [circle(2), circle(1.2).reverse()], tone: 0 },
  {
    contours: [
      [
        [3, -2],
        [6, -2],
        [4.5, 1],
      ],
    ],
    tone: 1,
  },
  { contours: [circle(0.8, 4.5, 2)], tone: 'muted' },
]

export default function FilledShapes() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', equal: x })
  return (
    <Figure title="Filled shapes" purpose="A ring, a triangle and a disc.">
      <Plot x={x} y={y}>
        <Shapes shapes={shapes} />
      </Plot>
    </Figure>
  )
}
