import type { Graph, GridRender } from 'aifn-compute/foundation/contracts'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { aStarSteps, dijkstraSteps } from 'aifn-compute/graph/shortest-paths'
import { Button, choice, Figure, StatusText, useFigureState } from 'aifn-render'
import { GridView } from 'aifn-render/gym'
import { useMemo, useRef, useState } from 'react'
import { rng } from '@examples/data'
import { useFrames, useOnScreen } from './live'

/** Rooms across and down; the grid has walls between them, so it is 2 × ROOMS + 1 cells square. */
const ROOMS = 12
const W = 2 * ROOMS + 1
const START = W + 1
const GOAL = W * (W - 2) + W - 2
const NAMES = ['explored', 'frontier', 'path']
/** Node expansions per frame. */
const PER_FRAME = 3

/** A perfect maze (one route between any two rooms) carved by a randomised depth-first walk; true cells are open. */
function carve(seed: number): boolean[] {
  const r = rng(seed)
  const open = new Array<boolean>(W * W).fill(false)
  const room = (i: number, j: number) => (2 * j + 1) * W + (2 * i + 1)
  const seen = new Set<number>([0])
  open[room(0, 0)] = true
  for (const stack: [number, number][] = [[0, 0]]; stack.length;) {
    const [i, j] = stack[stack.length - 1]
    const next = (
      [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const
    )
      .map(([di, dj]) => [i + di, j + dj] as [number, number])
      .filter(([a, b]) => a >= 0 && b >= 0 && a < ROOMS && b < ROOMS && !seen.has(b * ROOMS + a))
    if (!next.length) {
      stack.pop()
      continue
    }
    const [a, b] = next[Math.floor(r.uniform() * next.length)]
    seen.add(b * ROOMS + a)
    open[room(a, b)] = true
    open[(room(a, b) + room(i, j)) / 2] = true
    stack.push([a, b])
  }
  return open
}

/** The maze as an undirected graph on every cell: an edge of weight 1 between each pair of neighbouring open cells. */
function graphOf(open: boolean[]): Graph {
  const edges: { from: number; to: number; weight: number }[] = []
  for (let c = 0; c < W * W; c++) {
    if (!open[c]) continue
    if (c % W < W - 1 && open[c + 1]) edges.push({ from: c, to: c + 1, weight: 1 })
    if (c + W < W * W && open[c + W]) edges.push({ from: c, to: c + W, weight: 1 })
  }
  return { kind: 'graph', nodes: W * W, edges, directed: false }
}

const manhattan = (c: number) => Math.abs((c % W) - (GOAL % W)) + Math.abs(Math.floor(c / W) - Math.floor(GOAL / W))
const METHODS = {
  Dijkstra: (g: Graph) => dijkstraSteps(g, { source: START, target: GOAL }),
  'A*': (g: Graph) => aStarSteps(g, { source: START, target: GOAL, heuristic: manhattan }),
  greedy: (g: Graph) => aStarSteps(g, { source: START, target: GOAL, heuristic: (c) => 5 * manhattan(c) }),
}

/**
 * A maze searched from one corner to the other, a few expansions a frame: explored cells, the frontier, then the path
 * found. Dijkstra floods evenly; A* leans towards the goal; greedy (A* with the heuristic inflated) rushes at it and
 * can settle for a longer path. Once solved it holds a moment, then searches a new maze.
 */
export function MazeSearch() {
  const s = useFigureState({ method: choice(['A*', 'Dijkstra', 'greedy'], 'A*', { label: 'search' }) })
  const [seed, setSeed] = useState(1)
  const open = useMemo(() => carve(seed), [seed])
  const render = useMemo<GridRender<number>>(
    () => ({
      kind: 'grid',
      width: W,
      height: W,
      cells: open.map((o, c) => (c === START ? 'start' : c === GOAL ? 'goal' : o ? 'open' : 'wall')),
      cell: (c) => c,
      actionVectors: [],
    }),
    [open],
  )
  const method = s.method as keyof typeof METHODS
  const run = useMemo(() => live(METHODS[method](graphOf(open)), undefined), [open, method])
  const [view, setView] = useState({ tones: [] as number[], expanded: 0, length: 0, solved: false })
  const [seen, setSeen] = useState(run)
  if (seen !== run) {
    setSeen(run)
    setView({ tones: [], expanded: 0, length: 0, solved: false })
  }
  const hold = useRef(0)
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  useFrames(shown, (_t, dt) => {
    if (view.solved) {
      hold.current += dt
      if (hold.current > 2.5) {
        hold.current = 0
        setSeed((x) => x + 1)
      }
      return
    }
    let state = null
    let done = false
    for (let k = 0; k < PER_FRAME; k++) {
      const next = run.next()
      if (next.done) {
        done = true
        break
      }
      state = next.value.state
      if (next.value.stopped) {
        done = true
        break
      }
    }
    if (!state) return
    const settled = toFlat(state.settled)
    const distance = toFlat(state.distance)
    const tones: number[] = Array.from({ length: W * W }, (_, c) =>
      settled[c] ? 0 : Number.isFinite(distance[c]) ? 1 : -1,
    )
    let length = 0
    if (done || settled[GOAL]) {
      const pred = toFlat(state.predecessor)
      for (let c = GOAL; c >= 0; c = pred[c]) {
        tones[c] = 2
        length++
      }
    }
    setView({ tones, expanded: state.expanded, length, solved: done || settled[GOAL] === 1 })
  })
  return (
    <div ref={box}>
      <Figure
        title="Solve a maze"
        purpose="Shortest-path search on the maze's graph, a few expansions a frame."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={1}
        controlsCollapsed
        readouts={
          <div className="flex flex-wrap items-center gap-2">
            <StatusText>
              {view.solved
                ? `Solved: ${view.expanded} cells explored, a path of ${view.length - 1} steps.`
                : `${view.expanded} cells explored…`}
            </StatusText>
            <Button size="sm" variant="outline" className="ml-auto" onClick={() => setSeed((x) => x + 1)}>
              New maze
            </Button>
          </div>
        }
        caption="Dijkstra floods evenly, A* leans towards the goal, greedy rushes at it and may take a longer path."
      >
        <GridView render={render} tones={view.tones.length ? { values: view.tones, names: NAMES } : null} />
      </Figure>
    </div>
  )
}
