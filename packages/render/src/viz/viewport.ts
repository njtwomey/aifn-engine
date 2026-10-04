/**
 * Ranges, zoom and pan for the axis models. Zoom scales a range about a point, pan shifts it by a fraction of its span;
 * on a log axis both happen in log space, so a zoomed log axis keeps whole decades proportional.
 */

export type Range = [number, number]

const toLog = (log: boolean) => (v: number) => (log ? Math.log10(v) : v)
const fromLog = (log: boolean) => (v: number) => (log ? 10 ** v : v)

/** A usable range: finite, increasing, and positive on a log axis. */
export function validRange(r: Range | undefined, log = false): r is Range {
  return !!r && Number.isFinite(r[0]) && Number.isFinite(r[1]) && r[0] < r[1] && (!log || r[0] > 0)
}

/** `r` scaled about `about` (default its centre) by `factor`: above 1 zooms out, below 1 zooms in. */
export function zoomRange(r: Range, factor: number, log = false, about?: number): Range {
  if (log && !(r[0] > 0)) return r
  const [f, g] = [toLog(log), fromLog(log)]
  const [a, b] = [f(r[0]), f(r[1])]
  const c = about !== undefined && Number.isFinite(f(about)) ? f(about) : (a + b) / 2
  return [g(c + (a - c) * factor), g(c + (b - c) * factor)]
}

/** `r` shifted by `fraction` of its span: positive moves the view right (x) or up (y). */
export function panRange(r: Range, fraction: number, log = false): Range {
  if (log && !(r[0] > 0)) return r
  const [f, g] = [toLog(log), fromLog(log)]
  const d = (f(r[1]) - f(r[0])) * fraction
  return [g(f(r[0]) + d), g(f(r[1]) + d)]
}
