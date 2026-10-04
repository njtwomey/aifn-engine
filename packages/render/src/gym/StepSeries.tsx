/**
 * The chosen episode against step, under `GymTrainer`'s Player: the environment's state series (`render.series`) and
 * the actions (a strip of names for a discrete action, a line per element of a box), with a draggable cursor at the
 * played step. Step k's action is the one taken from state k.
 */
import { useMemo } from 'react'
import type { Environment, Trajectory } from 'aifn-compute/foundation/contracts'
import { Curve, Handle, Plot, Plots, Raster, useAxis, useChartHeight } from '@render/viz'
import { actionSeries, stateSeries } from './series'

export type StepSeriesProps = {
  env: Environment<unknown, unknown, unknown>
  trajectory: Trajectory<unknown, unknown, unknown>
  step: number
  onStep: (step: number) => void
  /** The share of the figure's height. */
  scale: number
}

export function StepSeries({ env, trajectory, step, onStep, scale }: StepSeriesProps) {
  const states = stateSeries(env)
  const actions = useMemo(() => actionSeries(env), [env])
  const data = useMemo(() => {
    const xs = trajectory.states.map((_, i) => i)
    const xa = trajectory.actions.map((_, i) => i)
    return {
      xs,
      xa,
      states: states.map((s) => trajectory.states.map(s.value)),
      actions: actions.map((a) =>
        a.kind === 'strip'
          ? [trajectory.actions.map((v) => v as number)]
          : trajectory.actions.map((v) => (v as ArrayLike<number>)[a.index]),
      ),
    }
  }, [trajectory, states, actions])
  const sa = useAxis({ label: 'step' })
  // One y axis per panel: at most four panels (two state series, two action series) in practice.
  const y0 = useAxis({ label: states[0]?.name ?? '' })
  const y1 = useAxis({ label: states[1]?.name ?? '' })
  const y2 = useAxis({ label: states[2]?.name ?? '' })
  // A strip is too short for a rotated axis name: its one category label names it, horizontally.
  const strip = actions[0]?.kind === 'strip'
  const ya = [useAxis(strip ? { categories: [actions[0]?.name ?? 'action'] } : { label: actions[0]?.name ?? 'action' })]
  const ys = [y0, y1, y2]
  // The cursor is labelled on the top panel only: the panels sit edge to edge, so a label above a lower panel would
  // overlap the panel above it.
  const cursorAt = (top: boolean) => (
    <Handle kind="x" at={step} label={top ? 'step' : undefined} onDrag={(v) => onStep(Math.max(0, Math.round(v)))} />
  )
  const panels = [
    ...states.slice(0, 3).map((s, i) => (
      <Plot key={s.name} x={sa} y={ys[i]} legend={false}>
        <Curve name={s.name} x={data.xs} y={data.states[i]} slot={i} />
        {cursorAt(i === 0)}
      </Plot>
    )),
    ...actions.slice(0, 1).map((a, i) =>
      a.kind === 'strip' ? (
        <Plot key={a.name} x={sa} y={ya[0]} legend={false}>
          <Raster x={data.xa} y={[0]} z={data.actions[i] as number[][]} scale="categorical" categoryNames={a.names} />
          {cursorAt(states.length === 0)}
        </Plot>
      ) : (
        <Plot key={a.name} x={sa} y={ya[0]} legend={false}>
          <Curve name={a.name} x={data.xa} y={data.actions[i] as number[]} slot={3} />
          {cursorAt(states.length === 0)}
        </Plot>
      ),
    ),
  ]
  // Row heights that give each panel the plot height its weight asks for: the lowest row also carries the step axis's
  // labels (about 40 px more than the others' margins), which would otherwise come out of its plot area and clip its
  // rotated axis name.
  const total = useChartHeight() * scale
  const weights = panels.map((_, i) => (i < states.length || !strip ? 1 : 0.6))
  const unit = Math.max(1, (total - 40) / weights.reduce((a, b) => a + b, 0))
  const heights = weights.map((w, i) => w + (i === weights.length - 1 ? 40 / unit : 0))
  return (
    <Plots rows={panels.length} cols={1} tight scale={scale} heights={heights}>
      {panels}
    </Plots>
  )
}
