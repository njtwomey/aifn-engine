import type { GridRender } from 'aifn-compute/foundation/contracts'
import { Figure, Player, usePlayhead } from 'aifn-render'
import { GridView } from 'aifn-render/gym'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Search on a grid',
  question: 'How do I show a search (visited cells, frontier, path) on a grid in colour only?',
  explain:
    '`tones` colours each cell by a category index into `names` (−1 for none) in place of the cell kinds; walls stay. Without `policy` or `path` the grid has no arrows. A `Player` steps the breadth-first search one expansion at a time.',
}

const ROWS = ['.....#....', '.###.#.##.', '.#...#..#G', '.#.####.#.', 'S........#'] // top row first
const W = ROWS[0].length
const H = ROWS.length
const KIND: Record<string, string> = { '#': 'wall', S: 'start', G: 'goal', '.': 'open' }
// region
const GRID: GridRender<number> = {
  kind: 'grid',
  width: W,
  height: H,
  cells: Array.from({ length: W * H }, (_, c) => KIND[ROWS[H - 1 - Math.floor(c / W)][c % W]]),
  cell: (s) => s,
  actionVectors: [],
}
// endregion
const neighbours = (c: number) =>
  [
    [0, 1],
    [1, 0],
    [0, -1],
    [-1, 0],
  ]
    .map(([dx, dy]) => [(c % W) + dx, Math.floor(c / W) + dy])
    .filter(([x, y]) => x >= 0 && y >= 0 && x < W && y < H && GRID.cells[y * W + x] !== 'wall')
    .map(([x, y]) => y * W + x)

// Breadth-first search from the start: after each expansion, a tone per cell (0 visited, 1 frontier, 2 path).
const start = GRID.cells.indexOf('start')
const goal = GRID.cells.indexOf('goal')
const parent = new Map([[start, -1]])
const frames: number[][] = []
for (const queue = [start]; queue.length > 0 && !parent.has(goal);) {
  const c = queue.shift()!
  for (const n of neighbours(c))
    if (!parent.has(n)) {
      parent.set(n, c)
      queue.push(n)
    }
  frames.push(Array.from({ length: W * H }, (_, s) => (queue.includes(s) ? 1 : parent.has(s) ? 0 : -1)))
}
const last = [...frames[frames.length - 1]]
for (let c = goal; c >= 0; c = parent.get(c)!) last[c] = 2
frames.push(last)
const NAMES = ['visited', 'frontier', 'path']

// region
export default function SearchColours() {
  const [k, setK] = usePlayhead(frames.length)
  return (
    <Figure
      title="Breadth-first search"
      purpose="The frontier grows one ring at a time until it reaches the goal; then the path back is traced."
      controls={<Player value={k} onChange={setK} count={frames.length} label="expansion" />}
    >
      <GridView render={GRID} tones={{ values: frames[k], names: NAMES }} />
    </Figure>
  )
}
// endregion
