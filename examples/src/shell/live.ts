/** Hooks for the landing page's figures that move on their own while they are on screen. */
import { useEffect, useRef, useState, type RefObject } from 'react'

const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

/** Whether at least half of the element is on screen (not clipped away by a carousel or scrolled off). */
export function useOnScreen(ref: RefObject<HTMLElement | null>) {
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(([e]) => setShown(e.isIntersecting && e.intersectionRatio > 0.5), {
      threshold: [0, 0.5, 1],
    })
    io.observe(el)
    return () => io.disconnect()
  }, [ref])
  return shown
}

/**
 * Calls `frame(seconds, dt)` about `fps` times a second while `running` (and the reader has not asked for reduced
 * motion): `seconds` is the time run so far, kept across pauses, and `dt` the time since the last call.
 */
export function useFrames(running: boolean, frame: (seconds: number, dt: number) => void, fps = 30) {
  const latest = useRef(frame)
  useEffect(() => {
    latest.current = frame
  })
  const clock = useRef(0)
  useEffect(() => {
    if (!running || reducedMotion()) return
    let id = 0
    let last = performance.now()
    const tick = (now: number) => {
      id = requestAnimationFrame(tick)
      const dt = (now - last) / 1000
      if (dt < 1 / fps - 0.002) return
      last = now
      clock.current += Math.min(dt, 0.1)
      latest.current(clock.current, Math.min(dt, 0.1))
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [running, fps])
}

/**
 * Whether the reader has touched the figure recently: `touch()` marks it, and it is cleared `idle` seconds later, so
 * automatic motion pauses while the reader drives and resumes after they leave it alone.
 */
export function useTouched(idle = 5) {
  const [touched, setTouched] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  const touch = () => {
    setTouched(true)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setTouched(false), idle * 1000)
  }
  return { touched, touch }
}
