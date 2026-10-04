/**
 * Deterministic layouts for undirected graphs given as data: a circle, and a force-directed layout (Fruchterman and
 * Reingold, 1991, "Graph drawing by force-directed placement", Software: Practice and Experience 21(11)) started from
 * the circle, with a fixed cooling schedule, so the same graph always gets the same picture. Positions are in grid
 * units, centred on zero, for `DiagramNode` `x`/`y`.
 */

export type Point = { x: number; y: number }

/** n nodes evenly on a circle of the given radius (grid units), node 0 at the top, going clockwise. */
export function circleLayout(n: number, radius = Math.max(1.5, (n * 1.4) / (2 * Math.PI))): Point[] {
  return Array.from({ length: n }, (_, i) => {
    const a = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2
    return { x: radius * Math.cos(a), y: radius * Math.sin(a) }
  })
}

/** Options of `forceLayout`. */
export type ForceOptions = {
  /** Iterations of the force simulation (default 300). */
  iterations?: number
  /** The ideal edge length in grid units (default 2). */
  length?: number
}

/**
 * A force-directed layout: every pair of nodes repels with force k²/d and every edge attracts with force d²/k (k the
 * ideal length); each iteration moves nodes by at most a temperature that cools linearly to zero. Edge direction is
 * ignored; self-loops and unknown ends are skipped.
 */
export function forceLayout(
  n: number,
  edges: readonly { from: number; to: number }[],
  options: ForceOptions = {},
): Point[] {
  const { iterations = 300, length: k = 2 } = options
  const pos = circleLayout(n, Math.max(k, (n * k) / (2 * Math.PI)))
  const links = edges.filter((e) => e.from !== e.to && e.from >= 0 && e.to >= 0 && e.from < n && e.to < n)
  const start = k * Math.sqrt(n) * 0.5
  for (let it = 0; it < iterations; it++) {
    const temperature = start * (1 - it / iterations)
    const dx = new Float64Array(n)
    const dy = new Float64Array(n)
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        let x = pos[i].x - pos[j].x
        let y = pos[i].y - pos[j].y
        let d = Math.hypot(x, y)
        if (d < 1e-6) {
          // Coincident nodes: separate them along a fixed direction so the result stays deterministic.
          x = 1e-3 * (i - j)
          y = 1e-3
          d = Math.hypot(x, y)
        }
        const f = (k * k) / d
        dx[i] += (x / d) * f
        dy[i] += (y / d) * f
        dx[j] -= (x / d) * f
        dy[j] -= (y / d) * f
      }
    for (const e of links) {
      const x = pos[e.from].x - pos[e.to].x
      const y = pos[e.from].y - pos[e.to].y
      const d = Math.max(Math.hypot(x, y), 1e-6)
      const f = (d * d) / k
      dx[e.from] -= (x / d) * f
      dy[e.from] -= (y / d) * f
      dx[e.to] += (x / d) * f
      dy[e.to] += (y / d) * f
    }
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(dx[i], dy[i])
      if (d === 0) continue
      const step = Math.min(d, temperature)
      pos[i] = { x: pos[i].x + (dx[i] / d) * step, y: pos[i].y + (dy[i] / d) * step }
    }
  }
  const mx = pos.reduce((a, p) => a + p.x, 0) / Math.max(n, 1)
  const my = pos.reduce((a, p) => a + p.y, 0) / Math.max(n, 1)
  return pos.map((p) => ({ x: p.x - mx, y: p.y - my }))
}
