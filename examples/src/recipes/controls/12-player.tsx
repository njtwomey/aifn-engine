import { Figure, Player, Plot, Points, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Player',
  question: 'How do I use a player as a control?',
  explain:
    '`count` is the number of positions; `format` shows a position as what it stands for (here a time); `loop` restarts at the end. A player opens at 0: to open later, give `startReason`, a sentence saying why.',
}

const ts = grid(0, 2 * Math.PI, 121)

export default function PlayerControl() {
  const [i, setI] = useState(60)
  const x = useAxis({ label: 'x', range: [-1.2, 1.2] })
  const y = useAxis({ label: 'y', range: [-1.2, 1.2], equal: x })
  return (
    <Figure
      title="A player that opens half way"
      purpose="A point on a Lissajous curve, opened at its crossing."
      controls={
        <Player
          value={i}
          onChange={setI}
          count={ts.length}
          loop
          format={(k) => `t = ${ts[k].toFixed(2)}`}
          startReason="the point is at the curve's central crossing"
        />
      }
    >
      <Plot x={x} y={y}>
        <Points name="curve" x={ts.map((t) => Math.sin(2 * t))} y={ts.map((t) => Math.sin(3 * t))} muted size={4} />
        <Points name="now" x={[Math.sin(2 * ts[i])]} y={[Math.sin(3 * ts[i])]} emphasis live />
      </Plot>
    </Figure>
  )
}
