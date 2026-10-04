import { Button, Curve, Figure, Plot, StatusText, useAxis } from 'aifn-render'
import { useState } from 'react'
import { Play } from 'lucide-react'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Run button',
  question: 'How do I add a Run button with a status line?',
  explain:
    "A `Button` starts the work and a `StatusText` beside it says what happened: `attention` before the first run, `muted` progress after. (The lab's `TrainControls` and `useTrainedRun` are not part of aifn-render.)",
}

export default function RunButtons() {
  const [runs, setRuns] = useState<number[][]>([])
  const run = () => {
    const r = rng(runs.length + 1)
    let v = 0
    setRuns([...runs, Array.from({ length: 100 }, () => (v += r.normal()))])
  }
  const x = useAxis({ label: 'step' })
  const y = useAxis({ label: 'position' })
  return (
    <Figure
      title="Random walks on demand"
      purpose="Each press adds one walk."
      controls={
        <div className="flex items-center gap-3">
          <Button size="sm" onClick={run}>
            <Play /> Run
          </Button>
          <StatusText tone={runs.length ? 'muted' : 'attention'}>
            {runs.length ? `${runs.length} walk${runs.length > 1 ? 's' : ''} so far` : 'Press Run to draw a walk.'}
          </StatusText>
        </div>
      }
    >
      <Plot x={x} y={y}>
        {runs.map((w, i) => (
          <Curve key={i} name={`walk ${i + 1}`} x={w.map((_, t) => t)} y={w} slot={i % 4} />
        ))}
      </Plot>
    </Figure>
  )
}
