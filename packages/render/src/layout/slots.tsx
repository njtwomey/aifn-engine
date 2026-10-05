/**
 * Frame slots: how a view (a panel, which never renders a `Figure`; design S §4.1) contributes its own controls,
 * readouts and a line about the object to the figure it sits in. Inside a Figure, `PanelSlot` portals its children
 * into the figure's slot; on its own (a dashboard cell, a page) it draws them in place around the panel.
 */
import { useContext, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Readouts } from '../viz/Readout'
import { Controls } from './Controls'

import { FrameSlotsContext, type SlotName } from './slots-context'

export type { FrameSlots, SlotName } from './slots-context'

/**
 * Children drawn in the enclosing Figure's `slot` (`about` under the purpose, `controls` after the figure's own,
 * `readouts` after the figure's own), or in place when there is no Figure.
 */
export function PanelSlot({ slot, children }: { slot: SlotName; children: ReactNode }) {
  const slots = useContext(FrameSlotsContext)
  // On the server (the render check) there is no DOM to portal into: draw in place, so the code still runs.
  if (!slots || typeof document === 'undefined') {
    if (slot === 'controls') return <Controls className="mb-3">{children}</Controls>
    if (slot === 'readouts') return <Readouts className="mt-2">{children}</Readouts>
    return <div className="mb-2 text-xs text-muted-foreground">{children}</div>
  }
  const target = slots[slot]
  // Before the figure has mounted its slot (and on the server) the children wait rather than flash in place.
  return target ? createPortal(children, target) : null
}
