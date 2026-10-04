/**
 * Environment renderers for `GymTrainer`, one per `render.kind`: each draws the chosen episode at a step as one `Plot`.
 * `grid` draws a gridworld or maze (cells by kind, the path so far, the agent); `bandit` (an environment without a
 * render whose episodes are single pulls) draws the pulls per arm up to the chosen episode. Add a kind to
 * `GYM_RENDERERS` (`registry.ts`). `pendulum` and `cartpole` draw classic control with
 * `PendulumView` and `CartPoleView`.
 */
import { useMemo, type ComponentType } from 'react'
import type {
  CartPoleRender,
  Environment,
  EpisodeEnd,
  GridRender,
  PendulumRender,
  Trajectory,
  Training,
} from 'aifn-compute/foundation/contracts'
import { Bars, Plot, Points, useAxis } from '@render/viz'
import { GridView, type GridValueField } from './GridView'
import { CartPoleView } from './CartPoleView'
import { PendulumView, type PendulumViewProps } from './PendulumView'

/** What a renderer receives: the environment, the chosen episode's trajectory and step, and the training run. */
export type GymRenderProps = {
  env: Environment<unknown, unknown, unknown>
  trajectory: Trajectory<unknown, unknown, unknown>
  /** Index into `trajectory.states` (0 is the reset). */
  step: number
  training: Training<unknown>
  /** The chosen episode (1-based). */
  episode: number
  /** The agent's state after the chosen episode (re-run from the nearest checkpoint; call it inside a memo). */
  agentAfter: () => unknown
  /** Page-specific options of a renderer (the pendulum's draggable `start`). */
  options?: Readonly<Record<string, unknown>>
  /** At the episode's last step, how it ended: the scene is drawn in the destructive or the success tone. */
  end?: EpisodeEnd | null
}

/** The tone of a scene at an ending: destructive for a failure, success for a success. */
const toneOf = (end: EpisodeEnd | null | undefined): 'destructive' | 'success' | undefined =>
  end ? (end.success ? 'success' : 'destructive') : undefined

export type GymRenderer = ComponentType<GymRenderProps>

/** What the agent knows, drawn under the path: a value field and a greedy policy per cell index. */
export type GridOverlay = { value?: GridValueField | null; policy?: ArrayLike<number> | null }

/**
 * Options of `GridRenderer` (`GymTrainerProps.rendererOptions`): `overlay` maps the agent's state after the chosen
 * episode to a value field and policy (a tabular agent's max Q and greedy actions); `onPlotClick` takes clicks on the
 * grid (toggling walls).
 */
export type GridRendererOptions = {
  overlay?: (agent: unknown) => GridOverlay
  onPlotClick?: (point: [number, number]) => void
}

/**
 * A gridworld or maze: the shared `GridView` with the episode's path up to the step and the agent there, over what the
 * agent had learnt after that episode when the page gives an `overlay`.
 */
export function GridRenderer({ env, trajectory, step, end, agentAfter, options }: GymRenderProps) {
  const { overlay, onPlotClick } = (options ?? {}) as GridRendererOptions
  const drawn = useMemo(() => (overlay ? overlay(agentAfter()) : null), [overlay, agentAfter])
  return (
    <GridView
      render={env.render as GridRender<unknown>}
      title={end ? `${env.name}: ${end.reason}` : env.name}
      value={drawn?.value}
      policy={drawn?.policy}
      path={trajectory.states}
      step={step}
      end={end}
      onPlotClick={onPlotClick}
    />
  )
}

export function BanditRenderer({ env, trajectory, step, training, episode }: GymRenderProps) {
  const arms = env.action.kind === 'discrete' ? env.action.n : 0
  const names = useMemo(
    () => (env.action.kind === 'discrete' && env.action.names) || Array.from({ length: arms }, (_, a) => `${a + 1}`),
    [env, arms],
  )
  const x = useMemo(() => Array.from({ length: arms }, (_, a) => a), [arms])
  const pulls = useMemo(() => {
    const c = new Array<number>(arms).fill(0)
    for (let i = 0; i < Math.min(episode, training.firstAction.length); i++) c[training.firstAction[i]] += 1
    return c
  }, [arms, episode, training])
  const arm = step > 0 ? (trajectory.actions[0] as number) : -1
  const xa = useAxis({ label: 'arm', categories: names })
  const ya = useAxis({ label: 'pulls so far', hold: 'union' })
  return (
    <Plot x={xa} y={ya} title={`pulls after episode ${episode}`}>
      <Bars name="pulls" x={x} y={pulls} slot={0} width={0.6} />
      {arm >= 0 && <Points name="this pull" x={[arm]} y={[pulls[arm]]} emphasis />}
    </Plot>
  )
}

/** A pendulum at the step, with the bob's path over the previous 20 steps. */
export function PendulumRenderer({ env, trajectory, step, options, end }: GymRenderProps) {
  const at = Math.min(step, trajectory.states.length - 1)
  const trail = useMemo(() => trajectory.states.slice(Math.max(0, at - 20), at + 1), [trajectory, at])
  return (
    <PendulumView
      render={env.render as PendulumRender<unknown>}
      state={trajectory.states[at]}
      trail={trail}
      title={env.name}
      start={options?.start as PendulumViewProps<unknown>['start']}
      tone={toneOf(end)}
    />
  )
}

/** A cart-pole at the step. */
export function CartPoleRenderer({ env, trajectory, step, end }: GymRenderProps) {
  const at = Math.min(step, trajectory.states.length - 1)
  return (
    <CartPoleView
      render={env.render as CartPoleRender<unknown>}
      state={trajectory.states[at]}
      title={end ? `${env.name}: ${end.reason}` : env.name}
      tone={toneOf(end)}
    />
  )
}
