/**
 * Whether the reader is holding the pointer down anywhere on the page (dragging a handle, a slider thumb, a probe),
 * and a signal when they let go. The scheduler (`useComputed`) reads it to tell a drag from a single change: during a
 * drag, slow derived work is thinned out or deferred, and it always runs on release.
 */

let held = false
let installed = false
const releases = new Set<() => void>()

function install() {
  if (installed || typeof window === 'undefined') return
  installed = true
  // Capture phase, so a widget that stops propagation (a chart's drag, a slider) still reports.
  window.addEventListener('pointerdown', () => (held = true), true)
  const up = () => {
    if (!held) return
    held = false
    for (const f of [...releases]) f()
  }
  window.addEventListener('pointerup', up, true)
  window.addEventListener('pointercancel', up, true)
  window.addEventListener('blur', up)
}

// Installed when the module loads (the scheduler imports it with the first figure), not on the first query: a lazy
// install missed the first press, so the first drag after load ran `release`-mode work on every move.
install()

/** True while a pointer is pressed anywhere in the window. */
export function isPointerHeld(): boolean {
  install()
  return held
}

/** Call `f` once, the next time the pointer is released; returns a function that cancels it. */
export function onceReleased(f: () => void): () => void {
  install()
  const once = () => {
    releases.delete(once)
    f()
  }
  releases.add(once)
  return () => releases.delete(once)
}
