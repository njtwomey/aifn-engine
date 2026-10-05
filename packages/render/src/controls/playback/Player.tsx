import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Button } from '../../ui/button'
import { ButtonGroup } from '../../ui/button-group'
import { cn } from '../../lib/utils'
import { Select } from '../selection/Select'
import { Slider } from '../numeric/Slider'

const SPEEDS = ['1', '2', '5', '10', '20', '30', '60', '120', '240'] as const
type Speed = (typeof SPEEDS)[number]

export type PlayerProps = {
  /** The current position, 0 … count − 1 (e.g. a kept step of a trace). */
  value: number
  onChange: (position: number) => void
  /** Number of positions. */
  count: number
  label?: ReactNode
  /** Formats a position in the slider's field, e.g. as the step number it stands for. */
  format?: (position: number) => string
  /**
   * Positions per second when playing (default 60, the lab-wide rate). Figures leave it alone; only a view whose frames
   * are simulation time steps (e.g. a gym episode) passes its own rate.
   */
  defaultSpeed?: number
  /** Start again from the beginning at the end rather than stopping. */
  loop?: boolean
  /**
   * Why this player opens at a later position (e.g. "the orbit has settled on its 2-cycle"). Players open at position 0
   * by default: without a reason, a non-zero starting value is reset to 0 when the player mounts.
   */
  startReason?: string
  className?: string
}

/**
 * Playback for anything that walks through a sequence (a trace's steps, frames, sweeps): first, previous, play/pause,
 * next and last buttons, a steppable slider with a typed position, and a speed. While the player has focus, Space
 * toggles play and the left and right arrows step (on the buttons; the slider handles its own arrows).
 */
export function Player({
  value,
  onChange,
  count,
  label = 'step',
  format,
  defaultSpeed = 60,
  loop = false,
  startReason,
  className,
}: PlayerProps) {
  const last = Math.max(0, count - 1)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState<Speed>(() => nearestSpeed(defaultSpeed))
  // Every walk-through opens at its first position unless the figure says why not (see `startReason`). Done here, once,
  // so no figure has to remember it.
  const opened = useRef(false)
  useEffect(() => {
    if (opened.current) return
    opened.current = true
    if (value !== 0 && !startReason) onChange(0)
  }, [value, startReason, onChange])
  const position = useRef(value)
  const change = useRef(onChange)
  useEffect(() => {
    position.current = value
    change.current = onChange
  })

  useEffect(() => {
    if (!playing) return
    let frame = 0
    let previous = performance.now()
    let carry = 0
    const tick = (now: number) => {
      carry += ((now - previous) / 1000) * Number(speed)
      previous = now
      const steps = Math.floor(carry)
      if (steps > 0) {
        carry -= steps
        let next = position.current + steps
        if (next > last) {
          if (!loop) {
            change.current(last)
            setPlaying(false)
            return
          }
          next %= last + 1
        }
        change.current(next)
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, speed, last, loop])

  const toggle = () => {
    if (!playing && value >= last && !loop) onChange(0)
    setPlaying((p) => !p)
  }
  const go = (p: number) => {
    setPlaying(false)
    onChange(Math.min(Math.max(p, 0), last))
  }

  return (
    <div
      className={cn('col-span-full flex flex-wrap items-end gap-x-4 gap-y-3', className)}
      onKeyDown={(e) => {
        const target = e.target as HTMLElement
        if (e.key === ' ' && target.tagName !== 'INPUT') {
          e.preventDefault()
          toggle()
        } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && target.tagName === 'BUTTON') {
          e.preventDefault()
          go(value + (e.key === 'ArrowLeft' ? -1 : 1))
        }
      }}
    >
      <ButtonGroup aria-label="Playback">
        <IconButton label="First" onClick={() => go(0)} disabled={value <= 0}>
          <SkipBack />
        </IconButton>
        <IconButton label="Previous" onClick={() => go(value - 1)} disabled={value <= 0}>
          <ChevronLeft />
        </IconButton>
        <IconButton label={playing ? 'Pause' : 'Play'} onClick={toggle} disabled={count < 2}>
          {playing ? <Pause /> : <Play />}
        </IconButton>
        <IconButton label="Next" onClick={() => go(value + 1)} disabled={value >= last}>
          <ChevronRight />
        </IconButton>
        <IconButton label="Last" onClick={() => go(last)} disabled={value >= last}>
          <SkipForward />
        </IconButton>
      </ButtonGroup>
      <Slider
        className="max-w-none min-w-48 flex-1"
        label={label}
        value={value}
        min={0}
        max={Math.max(1, last)}
        step={1}
        steppable={false}
        format={format}
        // Nothing to scrub (an untrained trainer, a one-frame run): the track is inert, like the buttons.
        disabled={count < 2}
        onChange={go}
      />
      <Select
        className="w-28"
        label="speed (per s)"
        value={speed}
        onChange={setSpeed}
        options={SPEEDS.map((s) => ({ value: s, label: `${s} / s` }))}
      />
    </div>
  )
}

/** The listed speed nearest `perSecond` on a log scale. */
function nearestSpeed(perSecond: number): Speed {
  let best: Speed = SPEEDS[0]
  for (const s of SPEEDS)
    if (Math.abs(Math.log(Number(s) / perSecond)) < Math.abs(Math.log(Number(best) / perSecond))) best = s
  return best
}

function IconButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  children: ReactNode
}) {
  return (
    <Button variant="outline" size="icon-sm" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      {children}
    </Button>
  )
}
