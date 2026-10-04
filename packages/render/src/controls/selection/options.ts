import type { ReactNode } from 'react'

/** One choice: a value, what to show, and optional words that search should also match. */
export type Option<T extends string> = { value: T; label?: ReactNode; keywords?: string; disabled?: boolean }

/** Options may be plain strings or objects; plain strings are their own labels. */
export type Options<T extends string> = readonly (T | Option<T>)[]

export type NormalOption<T extends string> = { value: T; label: ReactNode; text: string; disabled: boolean }

/** Options as objects, each with a plain-text label for searching and for the closed field. */
export function normalise<T extends string>(options: Options<T>): NormalOption<T>[] {
  return options.map((o) => {
    const option: Option<T> = typeof o === 'string' ? { value: o } : o
    const label = option.label ?? option.value
    const text = typeof label === 'string' || typeof label === 'number' ? String(label) : option.value
    return {
      value: option.value,
      label,
      text: [text, option.keywords].filter(Boolean).join(' '),
      disabled: !!option.disabled,
    }
  })
}
