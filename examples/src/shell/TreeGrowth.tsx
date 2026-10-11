import { pathToRoot, treeFromNested, type Tree } from 'aifn-compute/graph'
import {
  Columns,
  Figure,
  Plot,
  Points,
  Segments,
  Shapes,
  StatusText,
  TreeView,
  choice,
  int,
  useAxis,
  useChartHeight,
  useFigureState,
} from 'aifn-render'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import { rng } from '@examples/data'
import { useFrames, useNarrow, useOnScreen } from './live'

type Pt = { x: number; y: number; c: 0 | 1 }
type Box = [number, number, number, number]
type Split = { axis: 0 | 1; at: number }
/**
 * A node of the grown tree: its box [x₀, x₁, y₀, y₁], its points' indices, the split that made it (null for the root),
 * its own split and children once it has them, and the stages at which it appeared and was split (Infinity if never).
 */
type Node = {
  box: Box
  idx: number[]
  split: Split | null
  own: Split | null
  children: Node[]
  born: number
  splitAt: number
}

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

/**
 * CART grown best-first: each step splits the leaf whose best split most reduces impurity. Every stage's leaves are
 * kept, and the whole tree, each node marked with the stage it appeared at.
 */
function grow(points: Pt[], splits: number): { stages: Node[][]; root: Node } {
  const node = (box: Box, idx: number[], split: Split | null, born: number): Node => ({
    box,
    idx,
    split,
    own: null,
    children: [],
    born,
    splitAt: Infinity,
  })
  const root = node(
    BOX,
    points.map((_, i) => i),
    null,
    0,
  )
  let leaves = [root]
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
    l.own = split
    l.splitAt = s + 1
    l.children = [
      node(
        lo,
        l.idx.filter((i) => value(i) < b.at),
        split,
        s + 1,
      ),
      node(
        hi,
        l.idx.filter((i) => value(i) >= b.at),
        split,
        s + 1,
      ),
    ]
    leaves = [...leaves.filter((k) => k !== l), ...l.children]
    stages.push(leaves)
  }
  return { stages, root }
}

type Nested = { label: string; node: Node; children: Nested[] }
/** The grown tree as an `aifn-compute/graph` tree, its nodes in pre-order, each carrying its `Node` as `node`. */
function asTree(root: Node): Tree<{ node: Node }> {
  const nest = (n: Node): Nested => ({
    label: n.own ? `$x_${n.own.axis + 1} < ${n.own.at.toFixed(2)}$` : '',
    node: n,
    children: n.children.map(nest),
  })
  return treeFromNested(nest(root)) as unknown as Tree<{ node: Node }>
}

/** The four edges of a box, for outlining it. */
const outline = ([x0, x1, y0, y1]: Box) =>
  [
    [x0, y0, x1, y0],
    [x1, y0, x1, y1],
    [x1, y1, x0, y1],
    [x0, y1, x0, y0],
  ].map(([a, b, c, d]) => ({ from: [a, b] as const, to: [c, d] as const }))

/**
 * The plane and the tree: side by side on a wide screen, the plane above the tree on a phone, where each takes a share
 * of the frame's height.
 */
function Panes({ narrow, plane, tree }: { narrow: boolean; plane: ReactNode; tree: (height: number) => ReactNode }) {
  const height = useChartHeight()
  if (!narrow)
    return (
      <Columns
        widths={[1, 1]}
        panels={[
          { title: 'The plane', body: plane },
          { title: 'The tree', body: tree(height - 24) },
        ]}
      />
    )
  return (
    <div className="flex flex-col gap-2">
      {plane}
      {tree(Math.round(0.42 * height))}
    </div>
  )
}

/** Seconds between splits while the tree grows itself. */
const PACE = 0.55

/**
 * A decision tree (CART, Gini impurity) grown best-first on two moons, one split at a time: each leaf is shaded by its
 * majority class, more strongly the purer it is, and the tree itself grows beside the plane (below it on a phone), laid
 * out for its final shape so that nothing moves as nodes appear. Hovering a node outlines its region. It grows while on
 * screen, holds, and regrows on new data; a click adds a point of the chosen class and starts the growth again.
 */
export function TreeGrowth() {
  const s = useFigureState({
    splits: int(8, { min: 1, max: 40, label: 'splits' }),
    add: choice(['a', 'b'], 'a', { label: 'a click adds class' }),
  })
  const [seed, setSeed] = useState(3)
  const [extra, setExtra] = useState<Pt[]>([])
  const points = useMemo(() => [...moons(seed), ...extra], [seed, extra])
  const { stages, root } = useMemo(() => grow(points, s.splits), [points, s.splits])
  const tree = useMemo(() => asTree(root), [root])
  const [stage, setStage] = useState(0)
  const [hovered, setHovered] = useState<number | null>(null)
  const [seen, setSeen] = useState(stages)
  if (seen !== stages) {
    setSeen(stages)
    setStage(0)
    setHovered(null)
  }
  const clock = useRef(0)
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  const narrow = useNarrow()
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
  const shares = (n: Node) => {
    const n1 = n.idx.filter((i) => points[i].c === 1).length
    return { n1, n0: n.idx.length - n1, n: n.idx.length }
  }
  const correct = leaves.reduce((a, l) => {
    const { n0, n1 } = shares(l)
    return a + Math.max(n0, n1)
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
  const nodeOf = (id: number) => tree.nodes[id].node
  const visible = (id: number) => nodeOf(id).born <= stage
  // Nodes not yet split are drawn as leaves; nodes not yet made are hidden, keeping the final tree's layout.
  const cut = tree.nodes.map((_, id) => id).filter((id) => visible(id) && nodeOf(id).splitAt > stage)
  const focus = hovered !== null && visible(hovered) ? hovered : null
  const x = useAxis({ label: 'x₁', range: [BOX[0], BOX[1]] })
  const y = useAxis({ label: 'x₂', range: [BOX[2], BOX[3]], equal: x })
  const plane = (
    <Plot
      x={x}
      y={y}
      scale={narrow ? 0.56 : 1}
      onPlotClick={([a, b]) => setExtra((e) => [...e, { x: a, y: b, c: s.add === 'a' ? 0 : 1 }])}
    >
      <Shapes
        shapes={leaves.map((l) => {
          const { n0, n1, n } = shares(l)
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
            tone: n1 > n0 ? 1 : 0,
            opacity: 0.08 + 0.4 * ((2 * Math.max(n0, n1)) / Math.max(1, n) - 1),
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
      {focus !== null && <Segments segments={outline(nodeOf(focus).box)} tone="destructive" live />}
    </Plot>
  )
  return (
    <div ref={box}>
      <Figure
        title="Grow a decision tree"
        purpose="CART splits the plane one axis-aligned cut at a time, always where impurity falls most."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={narrow ? 0.62 : 1.9}
        controlsCollapsed
        readouts={
          <StatusText>
            {`${leaves.length} leaves after ${stage} splits: ${((100 * correct) / points.length).toFixed(0)}% of points in a leaf of their class.`}
          </StatusText>
        }
        caption="Each node shows its class shares; hover one to outline its region. Click the plane to add a point of the chosen class and watch the tree regrow around it."
      >
        <Panes
          narrow={narrow}
          plane={plane}
          tree={(height) => (
            <TreeView
              tree={tree}
              height={height}
              hidden={(id) => !visible(id)}
              cut={cut}
              highlight={focus === null ? [] : pathToRoot(tree, focus)}
              onNodeHover={setHovered}
              nodeSummary={(id) => {
                const { n0, n1, n } = shares(nodeOf(id))
                return { shares: [n0 / Math.max(1, n), n1 / Math.max(1, n)], n }
              }}
              ariaLabel="The decision tree grown so far"
            />
          )}
        />
      </Figure>
    </div>
  )
}
