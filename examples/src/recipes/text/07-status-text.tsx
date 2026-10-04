import { Button, StatusText } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Status line',
  question: 'How do I show progress, a prompt or an error beside a control?',
  explain:
    '`StatusText` is one line: `muted` for ordinary progress, `attention` for a state that asks the reader to act, and `error` in the destructive colour with an icon, so a failure never reads like progress.',
}

export default function StatusTextRecipe() {
  const [runs, setRuns] = useState(0)
  return (
    <div className="flex flex-col gap-2 rounded-lg border p-4">
      <div className="flex items-center gap-3">
        <Button size="sm" variant="outline" onClick={() => setRuns((r) => r + 1)}>
          Run
        </Button>
        {runs === 0 && <StatusText tone="attention">Not run yet: press Run.</StatusText>}
        {runs > 0 && runs % 3 !== 0 && <StatusText>Run {runs} finished in 12 ms.</StatusText>}
        {runs > 0 && runs % 3 === 0 && <StatusText tone="error">Run {runs} failed: the matrix is singular.</StatusText>}
      </div>
    </div>
  )
}
