import { Figure, Plot, Points, Segments, Shapes, StatusText, choice, int, useAxis, useFigureState } from 'aifn-render'
import { useMemo, useRef, useState } from 'react'
import { rng } from '@examples/data'
import { useFrames, useOnScreen } from './live'

type Pt = { x: number; y: number; c: 0 | 1 }
type Box = [number, number, number, number]
/** A leaf: its box [x₀, x₁, y₀, y₁], its points' indices, and the split that made it (null for the root). */
type Leaf = { box: Box; idx: number[]; split: { axis: 0 | 1; at: number } | null }

const BOX: Box = [-2.5, 3.5, -2, 2.5]

/** Two interleaved half-moons with noise: the classes curve round each other, so axis-aligned splits must staircase. */
function moons(seed: number): Pt[] {
  const r = rng(seed)
  return Array.from({ length: 220 }, (_, i) => {
    const c = (i % 2) as 0 | 1
    const t = Math.PI * r.uniform()
    const [x, y] = c === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)]
    return { x: x + 0.22 * r.normal(), y: y + 0.22 * r.normal(), c }
  })
}

const gini = (n0: number, n1: number) => {
  const n = n0 + n1
  return n === 0 ? 0 : 1 - (n0 / n) ** 2 - (n1 / n) ** 2
}

/** The split of a leaf that most reduces the weighted Gini impurity, over every threshold between points on each axis. */
function bestSplit(points: Pt[], idx: number[]) {
  let best = { gain: 0, axis: 0 as 0 | 1, at: 0 }
  const n0 = idx.filter((i) => points[i].c === 0).length
  const parent = idx.length * gini(n0, idx.length - n0)
  for (const axis of [0, 1] as const) {
    const sorted = [...idx].sort((a, b) => (axis ? points[a].y - points[b].y : points[a].x - points[b].x))
    let l0 = 0
    for (let k = 0; k < sorted.length - 1; k++) {
      if (points[sorted[k]].c === 0) l0++
      const v = axis ? points[sorted[k]].y : points[sorted[k]].x
      const w = axis ? points[sorted[k + 1]].y : points[sorted[k + 1]].x
      if (w === v) continue
      const left = k + 1
      const right = sorted.length - left
      const cost = left * gini(l0, left - l0) + right * gini(n0 - l0, right - (n0 - l0))
      if (parent - cost > best.gain + 1e-12) best = { gain: parent - cost, axis, at: (v + w) / 2 }
    }
  }
  return best
}

/** CART grown best-first: each step splits the leaf whose best split most reduces impurity. Every stage is kept. */
function grow(points: Pt[], splits: number): Leaf[][] {
  let leaves: Leaf[] = [{ box: BOX, idx: points.map((_, i) => i), split: null }]
  const stages = [leaves]
  for (let s = 0; s < splits; s++) {
    const options = leaves.map((l) => ({ l, b: bestSplit(points, l.idx) })).filter((o) => o.b.gain > 0)
    if (!options.length) break
    const { l, b } = options.reduce((a, o) => (o.b.gain > a.b.gain ? o : a))
    const [x0, x1, y0, y1] = l.box
    const lo: Box = b.axis ? [x0, x1, y0, b.at] : [x0, b.at, y0, y1]
    const hi: Box = b.axis ? [x0, x1, b.at, y1] : [b.at, x1, y0, y1]
    const value = (i: number) => (b.axis ? points[i].y : points[i].x)
    const split = { axis: b.axis, at: b.at }
    leaves = [
      ...leaves.filter((k) => k !== l),
      { box: lo, idx: l.idx.filter((i) => value(i) < b.at), split },
      { box: hi, idx: l.idx.filter((i) => value(i) >= b.at), split },
    ]
    stages.push(leaves)
  }
  return stages
}

/** Seconds between splits while the tree grows itself. */
const PACE = 0.55

/**
 * A decision tree (CART, Gini impurity) grown best-first on two moons, one split at a time: each leaf is shaded by its
 * majority class, more strongly the purer it is. It grows while on screen, holds, and regrows on new data; a click
 * adds a point of the chosen class and starts the growth again.
 */
export function TreeGrowth() {
  const s = useFigureState({
    splits: int(30, { min: 1, max: 60, label: 'splits' }),
    add: choice(['a', 'b'], 'a', { label: 'a click adds class' }),
  })
  const [seed, setSeed] = useState(3)
  const [extra, setExtra] = useState<Pt[]>([])
  const points = useMemo(() => [...moons(seed), ...extra], [seed, extra])
  const stages = useMemo(() => grow(points, s.splits), [points, s.splits])
  const [stage, setStage] = useState(0)
  const [seen, setSeen] = useState(stages)
  if (seen !== stages) {
    setSeen(stages)
    setStage(0)
  }
  const clock = useRef(0)
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  useFrames(shown, (_t, dt) => {
    clock.current += dt
    const last = stage >= stages.length - 1
    if (clock.current < (last ? 4 : PACE)) return
    clock.current = 0
    if (!last) setStage((k) => k + 1)
    else {
      setExtra([])
      setSeed((x) => x + 1)
    }
  })
  const leaves = stages[Math.min(stage, stages.length - 1)]
  const correct = leaves.reduce((a, l) => {
    const n1 = l.idx.filter((i) => points[i].c === 1).length
    return a + Math.max(n1, l.idx.length - n1)
  }, 0)
  // Every split line so far, each drawn across the box it divided: the edges shared by a leaf and its sibling.
  const edges = leaves
    .filter((l) => l.split)
    .map((l) => {
      const [x0, x1, y0, y1] = l.box
      const { axis, at } = l.split!
      return axis
        ? { from: [x0, at] as const, to: [x1, at] as const }
        : { from: [at, y0] as const, to: [at, y1] as const }
    })
  const x = useAxis({ label: 'x₁', range: [BOX[0], BOX[1]] })
  const y = useAxis({ label: 'x₂', range: [BOX[2], BOX[3]], equal: x })
  return (
    <div ref={box}>
      <Figure
        title="Grow a decision tree"
        purpose="CART splits the plane one axis-aligned cut at a time, always where impurity falls most."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={1.05}
        controlsCollapsed
        readouts={
          <StatusText>
            {`${leaves.length} leaves after ${stage} splits: ${((100 * correct) / points.length).toFixed(0)}% of points in a leaf of their class.`}
          </StatusText>
        }
        caption="Click to add a point of the chosen class and watch the tree regrow around it."
      >
        <Plot x={x} y={y} onPlotClick={([a, b]) => setExtra((e) => [...e, { x: a, y: b, c: s.add === 'a' ? 0 : 1 }])}>
          <Shapes
            shapes={leaves.map((l) => {
              const n1 = l.idx.filter((i) => points[i].c === 1).length
              const n = Math.max(1, l.idx.length)
              const [x0, x1, y0, y1] = l.box
              return {
                contours: [
                  [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                  ] as const,
                ],
                tone: n1 * 2 > l.idx.length ? 1 : 0,
                opacity: 0.08 + 0.4 * ((2 * Math.max(n1, n - n1)) / n - 1),
              }
            })}
            live
          />
          <Segments segments={edges} emphasis live />
          <Points
            name="points"
            x={points.map((p) => p.x)}
            y={points.map((p) => p.y)}
            group={points.map((p) => p.c)}
            groupNames={['a', 'b']}
            live
          />
        </Plot>
      </Figure>
    </div>
  )
}
