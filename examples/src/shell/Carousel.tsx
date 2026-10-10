import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react'
import { useEffect, useRef, useState, type ComponentType } from 'react'
import { Button, cn } from 'aifn-render'

export type Slide = {
  /** The tab's name, e.g. 'Optimisers'. */
  label: string
  /** One line beside the tabs on what the slide shows. */
  blurb: string
  Figure: ComponentType
  /** How long the slide shows before the next, in milliseconds (default `INTERVAL`). */
  duration?: number
}

/** How long a slide shows before the next, in milliseconds. */
const INTERVAL = 7000

const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

/**
 * Live figures one at a time, advancing on their own every few seconds. Every slide stays mounted, so a figure keeps
 * its state when it comes round again. It pauses while the pointer is over it or focus is inside, stops for good once
 * the reader drags or clicks in a figure, and never advances on its own when the reader asks for reduced motion.
 */
export function Carousel({ slides }: { slides: readonly Slide[] }) {
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(() => !reducedMotion())
  const [held, setHeld] = useState(false)
  const [cycle, setCycle] = useState(0)
  const n = slides.length
  const show = (i: number) => {
    setIndex(((i % n) + n) % n)
    setCycle((c) => c + 1)
  }
  useEffect(() => {
    if (!playing || held) return
    const id = setTimeout(() => setIndex((i) => (i + 1) % n), slides[index].duration ?? INTERVAL)
    return () => clearTimeout(id)
  }, [playing, held, index, cycle, n, slides])
  const running = playing && !held
  // On a narrow screen the tab strip scrolls: keep the slide on show in view (scrolling only the strip, not the page).
  const strip = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = strip.current
    const tab = el?.children[index] as HTMLElement | undefined
    if (!el || !tab || el.scrollWidth <= el.clientWidth) return
    el.scrollTo({ left: tab.offsetLeft - (el.clientWidth - tab.offsetWidth) / 2, behavior: 'smooth' })
  }, [index])
  // The frame takes the height of the slide on show (slides differ), following it as the figure resizes.
  const panels = useRef<(HTMLDivElement | null)[]>([])
  const [height, setHeight] = useState<number>()
  useEffect(() => {
    const el = panels.current[index]
    if (!el) return
    const ro = new ResizeObserver(() => setHeight(el.offsetHeight))
    ro.observe(el)
    return () => ro.disconnect()
  }, [index])
  return (
    <div
      className="space-y-4"
      onPointerEnter={() => setHeld(true)}
      onPointerLeave={() => setHeld(false)}
      onFocus={() => setHeld(true)}
      onBlur={() => setHeld(false)}
      aria-roledescription="carousel"
    >
      <div className="flex items-center gap-2">
        {/* One row that scrolls sideways on a narrow screen, wrapping once there is room. */}
        <div
          ref={strip}
          role="tablist"
          className="flex min-w-0 [scrollbar-width:none] gap-1 overflow-x-auto rounded-lg border bg-card p-1 sm:flex-wrap"
        >
          {slides.map((s, i) => (
            <button
              key={s.label}
              role="tab"
              aria-selected={i === index}
              onClick={() => show(i)}
              className={cn(
                'relative shrink-0 overflow-hidden rounded-md px-3 py-1.5 text-sm whitespace-nowrap transition-colors',
                i === index ? 'bg-accent font-medium text-foreground' : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {s.label}
              {i === index && running && (
                <span
                  key={`${index}-${cycle}`}
                  className="absolute inset-x-0 bottom-0 h-0.5 origin-left animate-[carousel-progress_linear_forwards] bg-foreground/60"
                  style={{ animationDuration: `${s.duration ?? INTERVAL}ms` }}
                />
              )}
            </button>
          ))}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="icon-sm" aria-label="Previous" onClick={() => show(index - 1)}>
            <ChevronLeft />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={playing ? 'Pause the carousel' : 'Play the carousel'}
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? <Pause /> : <Play />}
          </Button>
          <Button variant="ghost" size="icon-sm" aria-label="Next" onClick={() => show(index + 1)}>
            <ChevronRight />
          </Button>
        </div>
      </div>
      <p className="text-sm text-muted-foreground">{slides[index].blurb}</p>
      <div
        className="overflow-hidden rounded-xl border bg-card shadow-lg shadow-black/5 transition-[height] duration-500 motion-reduce:transition-none dark:shadow-black/40"
        style={{ height: height === undefined ? undefined : height + 2 }}
      >
        <div
          className="flex items-start transition-transform duration-700 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none"
          style={{ transform: `translateX(-${index * 100}%)` }}
          onPointerDown={() => setPlaying(false)}
        >
          {slides.map((s, i) => (
            <div
              key={s.label}
              ref={(el) => {
                panels.current[i] = el
              }}
              className="w-full shrink-0 p-2"
              aria-hidden={i !== index}
              inert={i !== index}
              role="tabpanel"
            >
              <s.Figure />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
