import { createContext } from 'react'

/** Size presets for a figure's chart area. `full` takes the width of the page. */
export const FIGURE_SIZES = {
  S: { width: 480, height: 260 },
  M: { width: 720, height: 360 },
  L: { width: 960, height: 480 },
  XL: { width: 1280, height: 600 },
  full: { width: '100%', height: 520 },
} as const

export type FigureSize = keyof typeof FIGURE_SIZES

/**
 * Where figure sizes are remembered: the shell sets it to the specimen's key, so a figure's size is kept per specimen.
 */
export const FigureScope = createContext('lab')
