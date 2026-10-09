import { useEffect, useRef, useState } from 'react'

/**
 * A playhead that plays itself while its element is on screen: from 0 each time the element comes into view, moving
 * on after `wait(step)` milliseconds at each position, up to `count - 1`. Once the reader moves it (`set`), it is
 * theirs and never plays itself again. Returns the element's ref, the position and the reader's setter.
 */
export function useAutoPlay(count: number, wait: (step: number) => number) {
  const [step, setStep] = useState(0)
  const [auto, setAuto] = useState(true)
  const [visible, setVisible] = useState(false)
  const autoRef = useRef(true)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      ([e]) => {
        const shown = e.isIntersecting && e.intersectionRatio > 0.5
        setVisible(shown)
        if (shown && autoRef.current) setStep(0)
      },
      { threshold: [0, 0.5, 1] },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  const delay = wait(step)
  useEffect(() => {
    if (!visible || !auto || step >= count - 1) return
    const id = setTimeout(() => setStep((s) => s + 1), delay)
    return () => clearTimeout(id)
  }, [visible, auto, step, count, delay])
  const set = (s: number) => {
    autoRef.current = false
    setAuto(false)
    setStep(s)
  }
  return { ref, step: Math.min(step, count - 1), set }
}

/** The time a self-playing figure takes to reach its last position: the sum of its waits. */
export const playTime = (count: number, wait: (step: number) => number) =>
  Array.from({ length: count - 1 }, (_, s) => wait(s)).reduce((a, b) => a + b, 0)
