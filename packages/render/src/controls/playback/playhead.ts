import { useCallback, useState } from 'react'

/**
 * A player's position that keeps its place in the run when the run's length changes (a new step size, a longer run):
 * held as a fraction of the way through, so a parameter change shows the same moment of the new run. `start` is that
 * fraction at first (0 by default; a later start needs the Player's `startReason`). Returns the position for `count`
 * and its setter, for `Player`.
 */
export function usePlayhead(count: number, start = 0): [number, (position: number) => void] {
  const [fraction, setFraction] = useState(start)
  const last = Math.max(0, count - 1)
  const position = Math.min(last, Math.max(0, Math.round(fraction * last)))
  const set = useCallback((p: number) => setFraction(last > 0 ? Math.min(1, Math.max(0, p / last)) : 1), [last])
  return [position, set]
}
