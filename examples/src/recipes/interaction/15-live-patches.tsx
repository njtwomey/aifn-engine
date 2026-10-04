import { Curve, Figure, Handle, Plot, Points, useAxis, type Vec2 } from 'aifn-render'
import { useMemo, useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid, rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Fast-moving marks',
  question: 'How do I keep a drag smooth when the chart has many marks?',
  explain:
    'Mark the layers that follow the pointer `live`: they are sent as a patch each frame without redrawing the rest, and never rescale the axes. Memoise the heavy, static layers on values, not on inline arrays.',
}

const r = rng(23)
const cloud = { x: Array.from({ length: 5000 }, () => r.normal()), y: Array.from({ length: 5000 }, () => r.normal()) }
const t = grid(0, 2 * Math.PI, 60)

export default function LivePatches() {
  const [c, setC] = useState<Vec2>([0.5, 0.5])
  // region
  const background = useMemo(() => <Points name="cloud" x={cloud.x} y={cloud.y} muted dense />, [])
  const x = useAxis({ label: 'x', range: [-4, 4] })
  const y = useAxis({ label: 'y', range: [-4, 4], equal: x })
  return (
    <Figure title="A live circle over 5000 points" purpose="Only the circle moves while you drag.">
      <Plot x={x} y={y}>
        {background}
        <Curve name="circle" x={t.map((s) => c[0] + Math.cos(s))} y={t.map((s) => c[1] + Math.sin(s))} emphasis live />
        <Handle kind="point" at={c} onDrag={setC} />
      </Plot>
    </Figure>
  )
  // endregion
}
