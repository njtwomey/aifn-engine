/** The `tex` tag and live slots of `Equation` (layout/Equation.tsx): TeX with highlighted live values. */
import { formatNumber } from '@render/viz/format'

/** A live value in an equation: highlighted, formatted to `digits` significant digits (default the lab's format). */
export type LiveValue = { value: number | string; digits?: number; strong?: boolean }
export type Slot = number | string | LiveValue

/** A TeX template with live values in its slots (from the `tex` tag). */
export type EquationTemplate = { strings: readonly string[]; slots: readonly Slot[] }

/** The `tex` tag: TeX with `${…}` slots for live values. The TeX is raw (backslashes are kept). */
export function tex(strings: TemplateStringsArray, ...slots: Slot[]): EquationTemplate {
  return { strings: strings.raw, slots }
}

/** A live value with options, for a slot: `${live(p, { digits: 3, strong: true })}`. */
export const live = (value: number | string, options: Omit<LiveValue, 'value'> = {}): LiveValue => ({
  value,
  ...options,
})

const number = (v: number, digits?: number) => {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? '\\text{NaN}' : v > 0 ? '\\infty' : '-\\infty'
  const text = digits === undefined ? formatNumber(v) : Number(v.toPrecision(digits)).toString()
  // The lab's format writes −, ×10^ and Unicode; KaTeX wants ASCII minus and \times.
  return text
    .replace(/−/g, '-')
    .replace(/×\s*10\^?([-−]?\d+)/g, '\\times 10^{$1}')
    .replace(/e([+-]?\d+)$/, '\\times 10^{$1}')
}

/** The template as one TeX string, each slot wrapped in a highlight class. */
export function toTex(t: EquationTemplate | string): string {
  if (typeof t === 'string') return t
  let out = t.strings[0]
  t.slots.forEach((slot, i) => {
    const s: LiveValue = typeof slot === 'object' ? slot : { value: slot }
    const body = typeof s.value === 'number' ? number(s.value, s.digits) : s.value
    out += `\\htmlClass{eq-slot${s.strong ? ' eq-slot-strong' : ''}}{${body}}` + t.strings[i + 1]
  })
  return out
}
