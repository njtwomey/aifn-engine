/**
 * Finite MDPs on grids, and finite MDPs as environments: `mdpEnvironment` turns any `TabularMdp` into an `Environment`
 * with a tabular model, an oracle (V* by value iteration), legal-action masking and, on a grid, a grid render. The
 * builders (`gridworld`, `cliffWalking`, `maze`, `frozenLake`) return `TabularMdp`s; their registered environment
 * forms are `gridworldEnvironment`, `cliffWalkingEnvironment`, `mazeEnvironment` and `frozenLakeEnvironment`.
 */

import type { Environment, EnvironmentInfo, GridRender, Outcome, TabularModel } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, discreteDomain, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { fromData } from 'aifn-compute/foundation/tensor'
import {
  cellState,
  GRID_ACTION_NAMES,
  GRID_ACTIONS,
  hasIllegalActions,
  isActive,
  legalActions,
  optimalValues,
  sampleOutcome,
  stateCell,
  type CellKind,
  type TabularMdp,
} from '../mdp'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── A finite MDP as an environment ───────────────────────────────────────────────────────────────────────────────────

/** A tabular MDP as an `Environment` whose state and observation are the state index and whose action is an index. */
export type MdpEnvironment = Environment<number, number, number> & {
  readonly model: TabularModel<number>
  readonly render?: GridRender<number>
}

/**
 * A finite MDP as an `Environment` (docs/aifn-gym.md §6): episodes start at `mdp.start`; a step samples one outcome
 * with one uniform draw; arriving at a terminal state ends the episode (`terminated`) and pays the outcome's reward
 * plus γ times the state's terminal value, so returns agree with the MDP's values. The tables are the `model` (for
 * planners); the `oracle` gives V* by value iteration (computed once, on first use) and E[r | s, a]; `legal` lists
 * the actions with outcomes when some state has an illegal action; `ending` calls a goal a success and a hole, trap or
 * time-out a failure; a grid MDP gains a grid `render`. `horizon` (default 4 × states) caps an episode.
 */
export function mdpEnvironment(
  mdp: TabularMdp,
  { horizon = 4 * mdp.states }: { horizon?: number } = {},
): MdpEnvironment {
  const id = (s: number) => s
  const model: TabularModel<number> = {
    kind: 'tabular',
    states: mdp.states,
    actions: mdp.actions,
    outcomes: mdp.outcomes,
    terminal: mdp.terminal,
    terminalValue: mdp.terminalValue,
    gamma: mdp.gamma,
    encode: id,
    decode: id,
  }
  let vStar: Float64Array | undefined
  const g = mdp.grid
  return {
    name: mdp.name,
    observation: discreteDomain(mdp.states),
    action: discreteDomain(mdp.actions, mdp.actionNames),
    gamma: mdp.gamma,
    horizon,
    reset: () => ({ state: mdp.start, observation: mdp.start }),
    step(s, a, stream) {
      if (!isActive(mdp, s)) throw new DomainError(mdp.name, `${mdp.name}: step from terminal state ${s}`)
      if (mdp.outcomes[s * mdp.actions + a].length === 0)
        throw new DomainError(mdp.name, `${mdp.name}: action ${a} is not legal in state ${s}`)
      const o = sampleOutcome(stream, mdp, s, a)
      const terminated = !isActive(mdp, o.next)
      const reward = o.reward + (terminated ? mdp.gamma * mdp.terminalValue[o.next] : 0)
      return { state: o.next, observation: o.next, reward, terminated, truncated: false }
    },
    ...(hasIllegalActions(mdp) && { legal: (s: number) => legalActions(mdp, s) }),
    ending(s, how, steps) {
      if (how === 'truncated')
        return { success: false, reason: `timed out after ${steps} step${steps === 1 ? '' : 's'}` }
      const kind = g?.kinds[s]
      if (kind === 'goal') return { success: true, reason: 'reached the goal' }
      if (kind === 'hole') return { success: false, reason: 'fell into a hole' }
      if (kind === 'trap' || kind === 'cliff') return { success: false, reason: `fell into a ${kind}` }
      const v = mdp.terminalValue[s]
      return v === 0 ? null : { success: v > 0, reason: `ended at an exit worth ${v}` }
    },
    model,
    oracle: {
      expectedReward: (s, a) =>
        mdp.outcomes[s * mdp.actions + a].reduce(
          (acc, o) => acc + o.p * (o.reward + (isActive(mdp, o.next) ? 0 : mdp.gamma * mdp.terminalValue[o.next])),
          0,
        ),
      optimalValues: () => fromData((vStar ??= optimalValues(mdp)).slice()),
    },
    ...(g && {
      render: {
        kind: 'grid' as const,
        width: g.width,
        height: g.height,
        cells: g.kinds,
        cell: id,
        actionVectors: g.actionVectors,
      },
    }),
  }
}

// ── Grid builders ────────────────────────────────────────────────────────────────────────────────────────────────────

interface GridSpec {
  name: string
  width: number
  height: number
  kinds: CellKind[]
  start: number
  gamma: number
  /** Probabilities of the intended direction and of each perpendicular one. */
  slip: number
  /** The result of landing on a cell after a move from `from`. */
  land: (cell: number) => { next: number; reward: number }
  terminalValue?: (s: number) => number
  /** Kinds that end an episode. */
  terminalKinds: CellKind[]
  /** A move off the grid or into a wall leaves the agent in place; `clamp` does so without calling `land`. */
}

function buildGrid(spec: GridSpec): TabularMdp {
  const { width, height, kinds, slip } = spec
  const S = width * height
  const terminal = Uint8Array.from(kinds, (k) => (k === 'wall' || spec.terminalKinds.includes(k) ? 1 : 0))
  const terminalValue = Float64Array.from({ length: S }, (_, s) =>
    terminal[s] && spec.terminalValue ? spec.terminalValue(s) : 0,
  )
  const move = (s: number, a: number) => {
    const [x, y] = stateCell(width, s)
    const nx = x + GRID_ACTIONS[a][0]
    const ny = y + GRID_ACTIONS[a][1]
    if (nx < 0 || ny < 0 || nx >= width || ny >= height) return s
    const t = cellState(width, nx, ny)
    return kinds[t] === 'wall' ? s : t
  }
  const outcomes: Outcome[][] = []
  for (let s = 0; s < S; s++)
    for (let a = 0; a < 4; a++) {
      if (terminal[s]) {
        outcomes.push([])
        continue
      }
      const dirs: [number, number][] =
        slip > 0
          ? [
              [a, 1 - slip],
              [(a + 1) % 4, slip / 2],
              [(a + 3) % 4, slip / 2],
            ]
          : [[a, 1]]
      // Merge outcomes that land in the same place with the same reward.
      const merged = new Map<string, Outcome>()
      for (const [dir, p] of dirs) {
        if (p <= 0) continue
        const { next, reward } = spec.land(move(s, dir))
        const key = `${next}:${reward}`
        const o = merged.get(key)
        if (o) o.p += p
        else merged.set(key, { p, next, reward })
      }
      outcomes.push([...merged.values()])
    }
  return {
    name: spec.name,
    states: S,
    actions: 4,
    outcomes,
    start: spec.start,
    terminal,
    terminalValue,
    gamma: spec.gamma,
    actionNames: GRID_ACTION_NAMES,
    grid: { width, height, kinds, actionVectors: GRID_ACTIONS },
  }
}

/** Options for `gridworld`. */
export interface GridworldOptions {
  width?: number
  height?: number
  /** Wall cells as [x, y]. */
  walls?: readonly (readonly [number, number])[]
  /** Terminal cells with their exit values (the reward for reaching them). */
  terminals?: readonly { x: number; y: number; value: number }[]
  /** Probability that a move goes to one of the two perpendicular directions instead (split equally). Default 0.2. */
  noise?: number
  /** Reward for every move. Default 0. */
  stepReward?: number
  gamma?: number
  start?: readonly [number, number]
}

/**
 * The stochastic gridworld of Russell and Norvig (2021, "Artificial Intelligence: A Modern Approach", §17.1) as the
 * site uses it: a 4 × 3 grid with a wall at (1, 1), exits worth +1 at (3, 2) and −1 at (3, 1), moves that slip
 * sideways with probability `noise`, and a step reward. Terminal cells hold their exit value.
 */
export function gridworld(options: GridworldOptions = {}): TabularMdp {
  const { width = 4, height = 3, noise = 0.2, stepReward = 0, gamma = 0.9 } = options
  const walls = options.walls ?? [[1, 1]]
  const terminals = options.terminals ?? [
    { x: 3, y: 2, value: 1 },
    { x: 3, y: 1, value: -1 },
  ]
  const kinds: CellKind[] = Array(width * height).fill('open')
  for (const [x, y] of walls) kinds[cellState(width, x, y)] = 'wall'
  const values = new Map<number, number>()
  for (const t of terminals) {
    const s = cellState(width, t.x, t.y)
    kinds[s] = t.value > 0 ? 'goal' : t.value < 0 ? 'trap' : 'terminal'
    values.set(s, t.value)
  }
  const [sx, sy] = options.start ?? [0, 0]
  return buildGrid({
    name: 'gridworld',
    width,
    height,
    kinds,
    start: cellState(width, sx, sy),
    gamma,
    slip: noise,
    land: (cell) => ({ next: cell, reward: stepReward }),
    terminalValue: (s) => values.get(s) ?? 0,
    terminalKinds: ['goal', 'trap', 'terminal'],
  })
}

/**
 * Cliff walking (Sutton and Barto, 2018, "Reinforcement Learning: An Introduction", Example 6.6): a 12 × 4 grid, start
 * at the bottom-left, goal at the bottom-right, and the cells between them a cliff. Every step pays −1; stepping into
 * the cliff pays −100 and returns the agent to the start. Moves are deterministic.
 */
export function cliffWalking({
  width = 12,
  height = 4,
  gamma = 1,
}: { width?: number; height?: number; gamma?: number } = {}): TabularMdp {
  const kinds: CellKind[] = Array(width * height).fill('open')
  const start = 0
  const goal = width - 1
  for (let x = 1; x < width - 1; x++) kinds[x] = 'cliff'
  kinds[start] = 'start'
  kinds[goal] = 'goal'
  return buildGrid({
    name: 'cliff walking',
    width,
    height,
    kinds,
    start,
    gamma,
    slip: 0,
    land: (cell) => (kinds[cell] === 'cliff' ? { next: start, reward: -100 } : { next: cell, reward: -1 }),
    terminalKinds: ['goal'],
  })
}

/** Options for `maze`. */
export interface MazeOptions {
  /** Probability of slipping to a perpendicular direction (split equally). Default 0. */
  slip?: number
  /** Rewards for a step, reaching the goal and falling into a trap. Defaults −1, 10 and −20. */
  rewards?: { step?: number; goal?: number; trap?: number }
  gamma?: number
}

/**
 * A maze from rows of text, top row first: `#` wall, `.` open, `S` start, `G` goal, `T` trap. Reaching the goal pays
 * the goal reward and ends the episode; a trap pays the trap reward and sends the agent back to the start; every other
 * move pays the step reward.
 */
export function maze(rows: readonly string[], options: MazeOptions = {}): TabularMdp {
  const { slip = 0, gamma = 0.95 } = options
  const rewards = { step: -1, goal: 10, trap: -20, ...options.rewards }
  const height = rows.length
  const width = rows[0].length
  const kinds: CellKind[] = Array(width * height).fill('open')
  let start = 0
  rows.forEach((row, i) => {
    const y = height - 1 - i
    ;[...row].forEach((ch, x) => {
      const s = cellState(width, x, y)
      if (ch === '#') kinds[s] = 'wall'
      else if (ch === 'T') kinds[s] = 'trap'
      else if (ch === 'G') kinds[s] = 'goal'
      else if (ch === 'S') {
        kinds[s] = 'start'
        start = s
      }
    })
  })
  return buildGrid({
    name: 'maze',
    width,
    height,
    kinds,
    start,
    gamma,
    slip,
    land: (cell) =>
      kinds[cell] === 'goal'
        ? { next: cell, reward: rewards.goal }
        : kinds[cell] === 'trap'
          ? { next: start, reward: rewards.trap }
          : { next: cell, reward: rewards.step },
    terminalKinds: ['goal'],
  })
}

/**
 * Built-in maze layouts, top row first (`#` wall, `S` start, `G` goal, `T` trap). `room` is an open 8 × 8 room;
 * `corridors` a winding maze; `cliff` a row of traps between start and goal (cliff walking as a maze); `routes` a short
 * bridge between traps against a long detour, where Q-learning and SARSA choose different routes.
 */
export const MAZES: Record<
  'small' | 'classic' | 'traps' | 'room' | 'corridors' | 'cliff' | 'routes',
  readonly string[]
> = {
  small: ['.....', '.###.', '.#G#.', '.#.#.', 'S....'],
  classic: ['.....#...G', '.###.#.##.', '.#...#....', '.#.####.#.', '.#......#.', 'S..####...'],
  traps: ['S..T....', '.#.#.##.', '.#...T..', '.####.#.', '......#G'],
  room: ['.......G', '........', '........', '........', '........', '........', '........', 'S.......'],
  corridors: [
    '..........G',
    '##########.',
    '.....#.....',
    '.###.#.###.',
    '.#.....#...',
    '.#######.##',
    '.#...#...#.',
    '.#.#.#.###.',
    '...#.#.#...',
    '####.#.#.#.',
    'S....#...#.',
  ],
  cliff: ['..........', '..........', '..........', 'STTTTTTTTG'],
  routes: ['.........', '.#######.', '.#######.', '.#TTTTT#.', 'S.......G', '##TTTTT##'],
}

/** The episode cap of an environment form: the longest episode before the rollout truncates it. */
export interface HorizonOption {
  /** Default 4 × the number of cells. */
  horizon?: number
}

const withHorizon = (mdp: TabularMdp, horizon: number | undefined) =>
  mdpEnvironment(mdp, horizon === undefined ? {} : { horizon })

/** `gridworld` as an environment (`mdpEnvironment`). */
export const gridworldEnvironment = ({ horizon, ...options }: GridworldOptions & HorizonOption = {}): MdpEnvironment =>
  withHorizon(gridworld(options), horizon)

/** `cliffWalking` as an environment (`mdpEnvironment`). */
export const cliffWalkingEnvironment = ({
  horizon,
  ...options
}: Parameters<typeof cliffWalking>[0] & HorizonOption = {}): MdpEnvironment =>
  withHorizon(cliffWalking(options), horizon)

/** Options for `mazeEnvironment`. */
export interface MazeEnvironmentOptions extends MazeOptions {
  /** A built-in layout or rows of text. Default `small`. */
  layout?: keyof typeof MAZES | readonly string[]
  /** The longest episode before truncation. Default 4 × the number of cells. */
  horizon?: number
}

/**
 * A maze as an `Environment` (docs/aifn-gym.md): the observation is the agent's cell index and the action one
 * of four moves (up, right, down, left); reaching the goal ends the episode (`terminated`), and the rollout truncates it
 * at `horizon`. Its `model` is the maze's transition table (for value iteration) and its `render` the grid.
 */
export function mazeEnvironment({
  layout = 'small',
  horizon,
  ...options
}: MazeEnvironmentOptions = {}): MdpEnvironment {
  return withHorizon(maze(typeof layout === 'string' ? MAZES[layout] : layout, options), horizon)
}

/** The FrozenLake maps of Gymnasium. */
export const FROZEN_LAKE_MAPS: Record<'4x4' | '8x8', readonly string[]> = {
  '4x4': ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  '8x8': ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF', 'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG'],
}

/**
 * FrozenLake (Gymnasium's `FrozenLake-v1`): cross a frozen lake from S to G without falling into a hole H. On slippery
 * ice the agent moves in the intended direction or either perpendicular one with probability 1/3 each. Reaching G pays
 * 1; holes end the episode with nothing. Map rows are given top row first; actions follow this module's order (up,
 * right, down, left), not Gymnasium's.
 */
export function frozenLake({
  map = '4x4',
  slippery = true,
  gamma = 0.99,
}: { map?: '4x4' | '8x8' | readonly string[]; slippery?: boolean; gamma?: number } = {}): TabularMdp {
  const rows = typeof map === 'string' ? FROZEN_LAKE_MAPS[map] : map
  const height = rows.length
  const width = rows[0].length
  const kinds: CellKind[] = Array(width * height).fill('open')
  let start = 0
  rows.forEach((row, i) =>
    [...row].forEach((ch, x) => {
      const s = cellState(width, x, height - 1 - i)
      if (ch === 'H') kinds[s] = 'hole'
      else if (ch === 'G') kinds[s] = 'goal'
      else if (ch === 'S') {
        kinds[s] = 'start'
        start = s
      }
    }),
  )
  return buildGrid({
    name: 'FrozenLake',
    width,
    height,
    kinds,
    start,
    gamma,
    slip: slippery ? 2 / 3 : 0,
    land: (cell) => ({ next: cell, reward: kinds[cell] === 'goal' ? 1 : 0 }),
    terminalKinds: ['goal', 'hole'],
  })
}

/** `frozenLake` as an environment (`mdpEnvironment`). */
export const frozenLakeEnvironment = ({
  horizon,
  ...options
}: Parameters<typeof frozenLake>[0] & HorizonOption = {}): MdpEnvironment => withHorizon(frozenLake(options), horizon)

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const environment = definer<EnvironmentInfo>('environment', 'gym/environments')
const tabular = { observation: 'discrete', action: 'discrete', capabilities: ['model', 'oracle', 'render'] } as const
const horizon = (fallback: number) => int(1, 10000, { default: fallback })

environment(
  {
    key: 'gridworldEnvironment',
    name: 'Gridworld',
    summary: 'A grid of cells with walls and terminal rewards, where moves slip sideways with probability `noise`.',
    family: 'mdp',
    params: space({
      width: int(2, 20, { default: 4 }),
      height: int(2, 20, { default: 3 }),
      noise: real(0, 1, { default: 0.2 }),
      stepReward: real(-1, 1, { default: 0 }),
      gamma: real(0, 1, { default: 0.9, label: 'γ' }),
      horizon: horizon(48),
    }),
    ...tabular,
    notes: ['markov-decision-process', 'value-iteration', 'policy-iteration'],
  },
  gridworldEnvironment,
)

environment(
  {
    key: 'cliffWalkingEnvironment',
    name: 'Cliff walking',
    summary: 'A start and a goal along the edge of a cliff that sends the agent back to the start at a large cost.',
    family: 'mdp',
    params: space({
      width: int(3, 30, { default: 12 }),
      height: int(2, 20, { default: 4 }),
      gamma: real(0, 1, { default: 1, label: 'γ' }),
      horizon: horizon(192),
    }),
    ...tabular,
    notes: ['q-learning', 'sarsa'],
  },
  cliffWalkingEnvironment,
)

environment(
  {
    key: 'mazeEnvironment',
    name: 'Maze',
    summary:
      'A maze (start, goal, walls, traps) with optional slip: cell-index observations, four moves, a tabular model.',
    family: 'mdp',
    params: space({
      layout: oneOf(['small', 'classic', 'traps', 'room', 'corridors', 'cliff', 'routes']),
      slip: real(0, 1, { default: 0 }),
      gamma: real(0, 1, { default: 0.95, label: 'γ' }),
      horizon: horizon(100),
    }),
    ...tabular,
    notes: ['solving-a-maze', 'q-learning'],
  },
  mazeEnvironment,
)

environment(
  {
    key: 'frozenLakeEnvironment',
    name: 'FrozenLake',
    summary:
      'Gymnasium’s FrozenLake: cross the ice to the goal without falling through a hole, on slippery ice or not.',
    family: 'mdp',
    params: space({
      map: oneOf(['4x4', '8x8']),
      slippery: bool({ default: true }),
      gamma: real(0, 1, { default: 0.99, label: 'γ' }),
      horizon: horizon(100),
    }),
    ...tabular,
    notes: ['markov-decision-process', 'value-iteration'],
  },
  frozenLakeEnvironment,
)
