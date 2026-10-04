import type { Algorithm } from 'aifn-compute/foundation/contracts'
import { trace } from 'aifn-compute/foundation/trace'
import { Contours, Curve, Figure, Player, Plot, Points, useAxis, usePlayhead } from 'aifn-render'
import { useMemo } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Step through a trace',
  question: 'How do I step through the iterations of an algorithm?',
  explain:
    'Record every state once with `trace(algorithm, start, steps, { keep: "all" })`, then let a `Player` choose which step to draw: the path so far, and the current point `live`. The axes are fixed, so playback never rescales.',
}

// Gradient descent on f(x, y) = x² + 10y², as an aifn Algorithm
type State = { t: number; x: number; y: number }
const descent: Algorithm<[number, number], State> = {
  name: 'descent',
  init: ([x, y]) => ({ t: 0, x, y }),
  step: (s) => ({ t: s.t + 1, x: s.x - 0.09 * 2 * s.x, y: s.y - 0.09 * 20 * s.y }),
}
const xs = grid(-3, 3, 60)
const ys = grid(-1.5, 1.5, 60)
const f = ys.map((b) => xs.map((a) => a * a + 10 * b * b))

export default function StepThroughTrace() {
  const run = useMemo(() => trace(descent, [-2.8, 1.2], 40, { keep: 'all' }).steps, [])
  const [step, setStep] = usePlayhead(run.length)
  const path = run.slice(0, step + 1)
  const x = useAxis({ label: 'x', range: [-3, 3] })
  const y = useAxis({ label: 'y', range: [-1.5, 1.5], equal: x })
  return (
    <Figure
      title="Gradient descent, step by step"
      purpose="A too-large step zig-zags across the narrow valley."
      controls={<Player value={step} onChange={setStep} count={run.length} />}
    >
      <Plot x={x} y={y}>
        <Contours x={xs} y={ys} z={f} levels={[0.5, 2, 5, 10, 20]} labels={false} />
        <Curve name="path" x={path.map((s) => s.x)} y={path.map((s) => s.y)} showPoints live />
        <Points name="now" x={[run[step].x]} y={[run[step].y]} emphasis live />
      </Plot>
    </Figure>
  )
}
