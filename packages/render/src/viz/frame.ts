/**
 * What a chart learns from the frame around it. A `Figure` (in aifn-render/layout) owns the size of its chart area and tells
 * the charts inside how tall to be; charts register their data (for the frame's export button) and their hovered values
 * (for the frame's hover readout). Outside a frame, charts use their own `height` and register nothing.
 */
import {
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from 'react'

/** One hovered value, for a readout: a series name, its value at the hovered position, and its colour. */
export type HoverRow = { label: string; value: string; color?: string }
/** What a chart reports while hovered: where (e.g. "x = 1.5") and the value of each series there. */
export type HoverInfo = { at: string; rows: HoverRow[] }

export type FrameContextValue = {
  /** The height every chart in the frame takes, in pixels. */
  height?: number
  setData?: (key: string, data: unknown) => void
  setHover?: (key: string, hover: HoverInfo | null) => void
}

export const FrameContext = createContext<FrameContextValue>({})

/** The default chart height outside a frame. */
export const DEFAULT_HEIGHT = 280

/**
 * The height a chart should take: the frame's when inside a Figure (the layout owns the size), otherwise the chart's own
 * `height` prop, otherwise the default.
 */
export function useChartHeight(own?: number): number {
  const { height } = useContext(FrameContext)
  return height ?? own ?? DEFAULT_HEIGHT
}

/** Registers `data` (or a function computing it at export time) with the enclosing frame while mounted. Memoise it. */
export function useFrameData(data: unknown): void {
  const { setData } = useContext(FrameContext)
  const key = useId()
  useEffect(() => {
    if (!setData) return
    setData(key, data)
    return () => setData(key, undefined)
  }, [setData, key, data])
}

/** A setter for this chart's entry in the enclosing frame's hover readout (a no-op outside a frame). */
export function useFrameHover(): (hover: HoverInfo | null) => void {
  const { setHover } = useContext(FrameContext)
  const key = useId()
  useEffect(() => () => setHover?.(key, null), [setHover, key])
  return useMemo(() => (hover: HoverInfo | null) => setHover?.(key, hover), [setHover, key])
}

/** The content-box size of an element, updated by a ResizeObserver; 0 × 0 until it is measured. */
export function useElementSize<T extends HTMLElement>(): [RefObject<T | null>, { width: number; height: number }] {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.round(entry.contentRect.width)
      const height = Math.round(entry.contentRect.height)
      setSize((s) => (s.width === width && s.height === height ? s : { width, height }))
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return [ref, size]
}
