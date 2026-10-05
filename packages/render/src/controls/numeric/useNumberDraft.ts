import { useState, type FocusEvent, type KeyboardEvent } from 'react'
import { formatField } from '../../state/step'

/** A draft's value, or why it cannot be committed. */
export type DraftCheck = { value: number; error?: undefined } | { error: string }

/**
 * The editing behaviour of a typed number: while focused the field holds a draft; Escape reverts; ↑ and ↓ step.
 * Accepts "−" and exponent notation.
 *
 * Without `check` (a slider's field) Enter or blur commits any number and the owner clamps it. With `check` (a
 * validated number field) only a valid draft commits: an invalid one shows `error`, Enter is refused, and blur reverts
 * to the last valid value. `stepBy` gives the value ↑/↓ moves to (default ± `step`, × 10 with Shift); null is no move.
 */
export function useNumberDraft({
  value,
  onCommit,
  step,
  format = formatField,
  check,
  stepBy,
  initialDraft = null,
}: {
  value: number
  onCommit: (v: number) => void
  step: number
  format?: (v: number) => string
  check?: (text: string) => DraftCheck
  stepBy?: (base: number, dir: 1 | -1, big: boolean) => number | null
  /** Text to start with as an uncommitted draft (the UI kit shows the invalid state this way). */
  initialDraft?: string | null
}) {
  const [draft, setDraft] = useState<string | null>(initialDraft)
  const parse = (text: string): DraftCheck => {
    if (check) return check(text)
    const n = Number(text.trim().replace(/^−/, '-'))
    return text.trim() !== '' && Number.isFinite(n) ? { value: n } : { error: 'not a number' }
  }
  const checked = draft === null ? null : parse(draft)
  const error = check && checked?.error !== undefined ? checked.error : null
  const commit = (text: string | null) => {
    setDraft(null)
    if (text === null) return
    const c = parse(text)
    if (c.error === undefined) onCommit(c.value)
  }
  return {
    error,
    props: {
      value: draft ?? format(value),
      inputMode: 'decimal' as const,
      'aria-invalid': error ? true : undefined,
      onFocus: (e: FocusEvent<HTMLInputElement>) => {
        setDraft(String(value))
        e.currentTarget.select()
      },
      onChange: (e: { target: { value: string } }) => setDraft(e.target.value),
      onBlur: () => commit(draft),
      onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
          // A validated field refuses an invalid draft: the message stays and the field keeps focus.
          if (!error) e.currentTarget.blur()
        } else if (e.key === 'Escape') {
          setDraft(null)
          const el = e.currentTarget
          // Blur once the reset has rendered, so the blur's commit sees no draft.
          requestAnimationFrame(() => el.blur())
        } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault()
          const base = checked && checked.error === undefined ? checked.value : value
          const dir = e.key === 'ArrowUp' ? 1 : -1
          const next = stepBy ? stepBy(base, dir, e.shiftKey) : base + dir * step * (e.shiftKey ? 10 : 1)
          if (next !== null) onCommit(next)
          setDraft(null)
        }
      },
    },
  }
}
