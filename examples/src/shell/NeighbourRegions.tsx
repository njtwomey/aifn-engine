import { normal, stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { Figure, Handle, int, Plot, Points, Raster, useAxis, useFigureState, type Vec2 } from 'aifn-render'
import { useState } from 'react'
import { grid } from '@examples/data'

const PER_CLASS = 30
const NAMES = ['a', 'b', 'c']
/** Fixed standard-normal offsets of each class's points from its centre, so the clouds move rigidly with a drag. */
const OFFSETS = NAMES.map((n) => toRows(normal(stream(`home/${n}`), 0, 0.9, { shape: [PER_CLASS, 2] })))
const gx = grid(-4, 4, 72)
const START: Vec2[] = [
  [-1.6, -1],
  [1.6, -0.8],
  [0, 1.6],
]

/** The k-nearest-neighbour vote on the grid: the class most common among the k nearest points of each cell. */
function regions(points: { x: number; y: number; c: number }[], k: number) {
  const d = new Float64Array(points.length)
  const order = points.map((_, i) => i)
  return gx.map((b) =>
    gx.map((a) => {
      points.forEach((p, i) => (d[i] = (p.x - a) ** 2 + (p.y - b) ** 2))
      order.sort((i, j) => d[i] - d[j])
      const votes = [0, 0, 0]
      for (let r = 0; r < k; r++) votes[points[order[r]].c]++
      return votes.indexOf(Math.max(...votes))
    }),
  )
}

/** Three draggable class clouds and the k-nearest-neighbour classifier's regions, recomputed on every move. */
export function NeighbourRegions() {
  const [centres, setCentres] = useState(START)
  const s = useFigureState({ k: int(7, { min: 1, max: 31, label: 'k neighbours' }) })
  const points = centres.flatMap(([cx, cy], c) => OFFSETS[c].map(([u, v]) => ({ x: cx + u, y: cy + v, c })))
  const z = regions(points, s.k)
  const x = useAxis({ label: 'x₁', range: [-4, 4] })
  const y = useAxis({ label: 'x₂', range: [-4, 4], equal: x })
  return (
    <Figure
      title="Move the classes"
      purpose="A k-nearest-neighbour classifier, redrawn as the data moves."
      state={s}
      hoverReadout={false}
      defaultSize="L"
      caption="Drag a class's ink centre; small k gives ragged islands, large k smooth borders."
    >
      <Plot x={x} y={y}>
        <Raster x={gx} y={gx} z={z} scale="categorical" categoryNames={NAMES} fillOpacity={0.3} boundary live />
        <Points
          name="points"
          x={points.map((p) => p.x)}
          y={points.map((p) => p.y)}
          group={points.map((p) => p.c)}
          groupNames={NAMES}
          live
        />
        {centres.map((c, i) => (
          <Handle key={i} kind="point" at={c} onDrag={(q) => setCentres((cs) => cs.map((r, j) => (j === i ? q : r)))} />
        ))}
      </Plot>
    </Figure>
  )
}
