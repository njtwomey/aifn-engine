import { Curve, Figure, Player, Plot, useAxis, usePlayhead } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Player',
  question: 'How do I animate a chart through a sequence of frames?',
  explain:
    '`Player` walks a position from 0 to `count − 1`: first, previous, play, next, last, a steppable slider and a speed (60 per second by default). Fix the axes so the frames do not rescale. One player per figure.',
}

const xs = grid(0, 10, 300)
const FRAMES = 240

export default function PlayerRecipe() {
  const [frame, setFrame] = usePlayhead(FRAMES)
  const t = frame / 24
  const x = useAxis({ label: 'x', range: [0, 10] })
  const y = useAxis({ label: 'u(x, t)', range: [-2.2, 2.2] })
  return (
    <Figure
      title="Two travelling waves"
      purpose="Waves moving in opposite directions add up to a beat."
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES}
          label="frame"
          format={(f) => `t = ${(f / 24).toFixed(2)}`}
        />
      }
    >
      <Plot x={x} y={y}>
        <Curve name="sum" x={xs} y={xs.map((v) => Math.sin(2 * v - 3 * t) + Math.sin(2.3 * v + 3 * t))} live />
      </Plot>
    </Figure>
  )
}
