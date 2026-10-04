import { Curve, Figure, Handle, Plot, useAxis, type Vec2 } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Drag a point',
  question: 'How do I let the reader drag a point on the chart?',
  explain:
    'A `Handle` of `kind="point"` sits `at` a position and calls `onDrag` with the new one; it writes the same state everything else reads. With one handle, pressing anywhere on the plot moves it.',
}

export default function DragAPoint() {
  const [tip, setTip] = useState<Vec2>([2, 1])
  const x = useAxis({ label: 'x', range: [-3, 3] })
  const y = useAxis({ label: 'y', range: [-3, 3], equal: x })
  return (
    <Figure
      title="A draggable vector"
      purpose="Drag the tip; the vector and its length follow."
      caption="Drag the ink point."
    >
      <Plot x={x} y={y}>
        <Curve name="v" x={[0, tip[0]]} y={[0, tip[1]]} live />
        <Handle kind="point" at={tip} onDrag={setTip} label="tip" />
      </Plot>
    </Figure>
  )
}
