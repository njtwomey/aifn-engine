import { normal, stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { Figure, Handle, int, Plot, Points, Raster, useAxis, useFigureState, type Vec2 } from 'aifn-render'
import { useRef, useState } from 'react'
import { grid } from '@examples/data'
import { useFrames, useOnScreen, useTouched } from './live'

const PER_CLASS = 30
const NAMES = ['a', 'b', 'c']
/** Fixed standard-normal offsets of each class's points from its centre, so the clouds move rigidly. */
const OFFSETS = NAMES.map((n) => toRows(normal(stream(`home/${n}`), 0, 0.9, { shape: [PER_CLASS, 2] })))
const gx = grid(-4, 4, 64)

/** Where the centres drift on their own: three points circling at different speeds, so the clouds meet and part. */
const drift = (t: number): Vec2[] => [
  [-1.4 + 1.1 * Math.cos(0.5 * t), -0.8 + 1.1 * Math.sin(0.5 * t)],
  [1.5 + 1.0 * Math.cos(-0.37 * t + 2), -0.6 + 1.3 * Math.sin(-0.37 * t + 2)],
  [0.1 + 1.4 * Math.sin(0.29 * t), 1.6 + 0.8 * Math.cos(0.43 * t)],
]

/** The k-nearest-neighbour vote on the grid: the class most common among the k nearest points of each cell. */
function regions(px: Float64Array, py: Float64Array, cls: Int8Array, k: number) {
  const n = px.length
  const bestD = new Float64Array(k)
  const bestC = new Int8Array(k)
  return gx.map((b) =>
    gx.map((a) => {
      // Keep the k smallest distances by insertion: k is small, so this beats sorting all n.
      let m = 0
      for (let i = 0; i < n; i++) {
        const d = (px[i] - a) ** 2 + (py[i] - b) ** 2
        if (m === k && d >= bestD[k - 1]) continue
        let j = m < k ? m++ : k - 1
        while (j > 0 && bestD[j - 1] > d) {
          bestD[j] = bestD[j - 1]
          bestC[j] = bestC[j - 1]
          j--
        }
        bestD[j] = d
        bestC[j] = cls[i]
      }
      const votes = [0, 0, 0]
      for (let r = 0; r < m; r++) votes[bestC[r]]++
      return votes.indexOf(Math.max(...votes))
    }),
  )
}

/**
 * Three clouds of points and the k-nearest-neighbour classifier's regions, recomputed every frame. On screen, the
 * clouds drift on their own; dragging a centre takes over, and the drift glides back a few seconds after.
 */
export function NeighbourRegions() {
  const [centres, setCentres] = useState(() => drift(0))
  const s = useFigureState({ k: int(7, { min: 1, max: 31, label: 'k neighbours' }) })
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const { touched, touch } = useTouched(5)
  useFrames(shown && !touched, (t, dt) => {
    const target = drift(t)
    const k = Math.min(1, 2.5 * dt)
    setCentres((cs) => cs.map((c, i) => [c[0] + k * (target[i][0] - c[0]), c[1] + k * (target[i][1] - c[1])]))
  })
  const points = centres.flatMap(([cx, cy], c) => OFFSETS[c].map(([u, v]) => ({ x: cx + u, y: cy + v, c })))
  const z = regions(
    Float64Array.from(points, (p) => p.x),
    Float64Array.from(points, (p) => p.y),
    Int8Array.from(points, (p) => p.c),
    s.k,
  )
  const x = useAxis({ label: 'x₁', range: [-4, 4] })
  const y = useAxis({ label: 'x₂', range: [-4, 4], equal: x })
  return (
    <div ref={box}>
      <Figure
        title="Move the classes"
        purpose="A k-nearest-neighbour classifier, redrawn as the data moves."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        caption="The clouds drift on their own; drag an ink centre to take over. Small k gives ragged islands."
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
            <Handle
              key={i}
              kind="point"
              at={c as Vec2}
              onDrag={(q) => {
                touch()
                setCentres((cs) => cs.map((r, j) => (j === i ? q : r)))
              }}
            />
          ))}
        </Plot>
      </Figure>
    </div>
  )
}
