import { Play, RotateCcw, StepForward } from 'lucide-react'
import { Button } from '../../ui/button'
import { ButtonGroup } from '../../ui/button-group'

/** Step, Run and Reset for iterative algorithms run live (k-means, EM, ...). Step and Run disable once `done`. */
export function StepControls({
  onStep,
  onRun,
  onReset,
  done,
}: {
  onStep: () => void
  onRun: () => void
  onReset: () => void
  done: boolean
}) {
  return (
    <ButtonGroup aria-label="Algorithm">
      <Button variant="outline" size="sm" onClick={onStep} disabled={done}>
        <StepForward /> Step
      </Button>
      <Button variant="outline" size="sm" onClick={onRun} disabled={done}>
        <Play /> Run
      </Button>
      <Button variant="outline" size="sm" onClick={onReset}>
        <RotateCcw /> Reset
      </Button>
    </ButtonGroup>
  )
}
