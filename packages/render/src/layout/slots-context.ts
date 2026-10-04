import { createContext } from 'react'

export type SlotName = 'about' | 'controls' | 'readouts'
export type FrameSlots = Partial<Record<SlotName, HTMLElement | null>>

/** The slots of the enclosing Figure (null outside one); `PanelSlot` portals into them. */
export const FrameSlotsContext = createContext<FrameSlots | null>(null)
