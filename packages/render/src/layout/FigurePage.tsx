import { useMemo, type ReactNode } from 'react'
import { createFigureIds, FigureIdsContext } from './figure-ids'
import { FigureScope } from './figure-size'

/** One page of figures: sizes are remembered under `scope`, and figure ids are unique within it. */
export function FigurePage({ scope, children }: { scope: string; children: ReactNode }) {
  const ids = useMemo(() => createFigureIds(), [])
  return (
    <FigureScope.Provider value={scope}>
      <FigureIdsContext.Provider value={ids}>{children}</FigureIdsContext.Provider>
    </FigureScope.Provider>
  )
}
