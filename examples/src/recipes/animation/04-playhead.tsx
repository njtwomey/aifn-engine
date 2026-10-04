import { Curve, Figure, int, Player, Plot, Points, useAxis, useFigureState, usePlayhead } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Keep the moment',
  question: 'How does a player keep its place when the run changes length?',
  explain:
    '`usePlayhead(count)` holds the position as a fraction of the run, so a change that adds or removes frames shows the same moment of the new run. Players open at frame 0; pass `startReason` to open later.',
}

export default function Playhead() {
  const s = useFigureState({ frames: int(50, { ge: 5, le: 500, label: 'frames per lap' }) })
  const [frame, setFrame] = usePlayhead(s.frames)
  const angle = (2 * Math.PI * frame) / (s.frames - 1)
  const t = grid(0, 2 * Math.PI, 100)
  const x = useAxis({ label: 'x', range: [-1.2, 1.2] })
  const y = useAxis({ label: 'y', range: [-1.2, 1.2], equal: x })
  return (
    <Figure
      title="A lap at any resolution"
      purpose="Change the frame count: the point stays where it was on the lap."
      state={s}
      controls={<Player value={frame} onChange={setFrame} count={s.frames} loop />}
    >
      <Plot x={x} y={y}>
        <Curve name="lap" x={t.map(Math.cos)} y={t.map(Math.sin)} muted />
        <Points name="now" x={[Math.cos(angle)]} y={[Math.sin(angle)]} emphasis live />
      </Plot>
    </Figure>
  )
}
