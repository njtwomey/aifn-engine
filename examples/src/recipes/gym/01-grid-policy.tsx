import type { GridRender } from 'aifn-compute/foundation/contracts'
import { Figure } from 'aifn-render'
import { GridView } from 'aifn-render/gym'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Maze with a policy',
  question: 'How do I draw a maze or gridworld with values, a policy and a path?',
  explain:
    '`GridView` draws any compute `GridRender` (cell kinds, cell of a state, action vectors): `value` colours the cells, `policy` (an action per cell) draws arrows, `path` (a list of states) draws counted move arrows. Here the value is minus the distance to the goal and the policy steps downhill.',
}

const ROWS = ['....#...G', '.##.#.##.', '.#..#..#.', '.#.###.#.', 'S........'] // top row first
const W = ROWS[0].length
const H = ROWS.length
const KIND: Record<string, string> = { '#': 'wall', S: 'start', G: 'goal', '.': 'open' }
const MOVES: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
] // up, right, down, left
// region
const MAZE: GridRender<number> = {
  kind: 'grid',
  width: W,
  height: H,
  cells: Array.from({ length: W * H }, (_, c) => KIND[ROWS[H - 1 - Math.floor(c / W)][c % W]]),
  cell: (s) => s,
  actionVectors: MOVES,
}
// endregion
/** The cell one move away, or −1 off the grid or into a wall. */
const next = (c: number, [dx, dy]: [number, number]) => {
  const [x, y] = [(c % W) + dx, Math.floor(c / W) + dy]
  return x < 0 || y < 0 || x >= W || y >= H || MAZE.cells[y * W + x] === 'wall' ? -1 : y * W + x
}

// Breadth-first distances from the goal; each open cell's action moves one step closer.
const goal = MAZE.cells.indexOf('goal')
const dist = new Array<number>(W * H).fill(Infinity)
dist[goal] = 0
for (const queue = [goal]; queue.length > 0;) {
  const c = queue.shift()!
  for (const m of MOVES)
    if (next(c, m) >= 0 && dist[next(c, m)] === Infinity) (queue.push(next(c, m)), (dist[next(c, m)] = dist[c] + 1))
}
const policy = dist.map((d, c) => (d > 0 && d < Infinity ? MOVES.findIndex((m) => dist[next(c, m)] === d - 1) : -1))
const path = [MAZE.cells.indexOf('start')]
while (path.at(-1) !== goal) path.push(next(path.at(-1)!, MOVES[policy[path.at(-1)!]]))
const values = dist.map((d) => (d < Infinity ? -d : NaN))

// region
export default function GridPolicy() {
  return (
    <Figure title="Shortest routes in a maze" purpose="Values, the greedy policy and the route from the start.">
      <GridView
        render={MAZE}
        value={{ values, label: '−distance', scale: 'sequential' }}
        policy={policy}
        path={path}
        inkLatest={false}
        agent={false}
      />
    </Figure>
  )
}
// endregion
