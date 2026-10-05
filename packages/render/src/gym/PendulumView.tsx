import { useMemo } from 'react'
import type { PendulumRender } from 'aifn-compute/foundation/contracts'
import { Curve, Handle, Plot, Points, useAxis } from '../viz'

/** Props of `PendulumView`: one step's environment state, drawn by the environment's `pendulum` render spec. */
export type PendulumViewProps<S> = {
  render: PendulumRender<S>
  state: S
  /** Recent states, drawn as the bob's faint path (oldest first). */
  trail?: readonly S[]
  /** A draggable start angle: the marker sits at `angle` and dragging sets it (radians from upright). */
  start?: { angle: number; onDrag: (angle: number) => void }
  title?: string
  /** An ending's tone: the rod and bob in the destructive red or the success green. */
  tone?: 'destructive' | 'success'
}

/** The bob of a rod at angle θ from upright (anticlockwise), as Gymnasium draws it. */
const bob = (theta: number, l: number): [number, number] => [-l * Math.sin(theta), l * Math.cos(theta)]

/** A pendulum drawn as a rod from its pivot at the state's angle, with equal units on both axes. */
export function PendulumView<S>({ render, state, trail, start, title, tone }: PendulumViewProps<S>) {
  const l = render.length
  const reach = 1.3 * l
  const x = useAxis({ label: 'x (m)', range: [-reach, reach] })
  const y = useAxis({ label: 'y (m)', range: [-reach, reach], equal: x })
  const [bx, by] = bob(render.angle(state), l)
  const path = useMemo(() => {
    const pts = (trail ?? []).map((s) => bob(render.angle(s), l))
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) }
  }, [trail, render, l])
  const s0 = start && bob(start.angle, l)
  return (
    <Plot x={x} y={y} title={title}>
      {trail && <Curve name="bob, recent path" x={path.x} y={path.y} slot={1} thin silent />}
      <Curve name="rod" x={[0, bx]} y={[0, by]} emphasis={!tone} tone={tone} width={4} silent />
      <Points name="pivot" x={[0]} y={[0]} muted />
      <Points name="bob" x={[bx]} y={[by]} emphasis={!tone} tone={tone} size={14} />
      {start && s0 && <Points name="start" x={[s0[0]]} y={[s0[1]]} slot={0} />}
      {start && s0 && (
        <Handle kind="point" at={s0} label="θ₀" onDrag={([px, py]) => start.onDrag(Math.atan2(-px, py))} />
      )}
    </Plot>
  )
}
