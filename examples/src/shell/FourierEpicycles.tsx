import { fft } from 'aifn-compute/foundation/fourier'
import { tensor, toComplexFlat } from 'aifn-compute/foundation/tensor'
import {
  choice,
  Curve,
  Figure,
  int,
  Plot,
  Points,
  Segments,
  seriesColor,
  useAxis,
  useFigureState,
  useTheme,
} from 'aifn-render'
import { useRef, useState } from 'react'
import { grid } from '@examples/data'
import { useFrames, useOnScreen } from './live'

const N = 512
const ts = grid(0, 2 * Math.PI, N + 1).slice(0, N)

/** The k-th vertex of the Hilbert curve on an n × n grid (n a power of two), by the usual quadrant-rotation walk. */
function hilbertVertex(n: number, k: number): number[] {
  let [x, y] = [0, 0]
  for (let s = 1, d = k; s < n; s *= 2, d = Math.floor(d / 4)) {
    const rx = 1 & Math.floor(d / 2)
    const ry = 1 & (d ^ rx)
    if (ry === 0) {
      if (rx === 1) [x, y] = [s - 1 - x, s - 1 - y]
      ;[x, y] = [y, x]
    }
    x += s * rx
    y += s * ry
  }
  return [x, y]
}

/**
 * The order-3 Hilbert curve (8 × 8) traced out and back along itself, so the loop closes without a jump, sampled at
 * N points evenly by arc length.
 */
function hilbertLoop(): number[][] {
  const n = 8
  const out = Array.from({ length: n * n }, (_, k) => hilbertVertex(n, k).map((v) => ((v - (n - 1) / 2) * 6) / (n - 1)))
  const path = [...out, ...out.slice(0, -1).reverse()]
  // Every edge has the same length, so even spacing in arc length is even spacing in edge index.
  const edges = path.length - 1
  return Array.from({ length: N }, (_, i) => {
    const u = (i * edges) / N
    const e = Math.floor(u)
    const f = u - e
    return [path[e][0] + f * (path[e + 1][0] - path[e][0]), path[e][1] + f * (path[e + 1][1] - path[e][1])]
  })
}

/** The closed curves to draw, each sampled at N points around one loop. */
const SHAPES: Record<string, number[][]> = {
  hilbert: hilbertLoop(),
  heart: ts.map((t) => [
    (16 * Math.sin(t) ** 3) / 5,
    (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 5,
  ]),
  star: ts.map((t) => {
    const r = 2.2 + 0.9 * Math.cos(5 * t)
    return [r * Math.sin(t), r * Math.cos(t)]
  }),
  square: ts.map((t) => {
    // Round a square's perimeter at constant speed: side s of four, position u along it.
    const q = (4 * t) / (2 * Math.PI)
    const s = Math.floor(q)
    const u = 2 * (q - s) - 1
    const side = [
      [u, -1],
      [1, u],
      [-u, 1],
      [-1, -u],
    ][s]
    return [2.6 * side[0], 2.6 * side[1]]
  }),
}

type Term = { freq: number; re: number; im: number; amp: number }

/** The curve's Fourier series as rotating circles, largest first: the DFT of x + iy over one loop, divided by N. */
function series(points: number[][]): Term[] {
  const X = toComplexFlat(fft(tensor(points.map(([a, b]) => ({ re: a, im: b })))))
  return X.map(({ re, im }, k) => ({
    freq: k < N / 2 ? k : k - N,
    re: re / N,
    im: im / N,
    amp: Math.hypot(re, im) / N,
  }))
    .filter((t) => t.amp > 1e-9)
    .sort((a, b) => b.amp - a.amp)
}
const SERIES = Object.fromEntries(Object.entries(SHAPES).map(([k, v]) => [k, series(v)]))

/** The partial sums of the first `terms` circles at angle θ: the centres of the chain, ending at the pen. */
function chain(terms: Term[], theta: number) {
  let [x, y] = [0, 0]
  const centres = [[x, y]]
  for (const t of terms) {
    const [c, s] = [Math.cos(t.freq * theta), Math.sin(t.freq * theta)]
    x += t.re * c - t.im * s
    y += t.re * s + t.im * c
    centres.push([x, y])
  }
  return centres
}

const circle = grid(0, 2 * Math.PI, 40)
/** The pen's trail is drawn as this many pieces, each older one fainter, so it fades out behind the pen. */
const PIECES = 10

/** A `#rrggbb` colour at opacity `a`. */
const withAlpha = (hex: string, a: number) =>
  `rgba(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)}, ${a.toFixed(3)})`

/** How much of a loop the pen's trail covers before it has faded out. */
const TRAIL = 0.45

/** Seconds the pen takes to go once round. */
const PERIOD = 10

/**
 * A closed curve drawn by a chain of rotating circles, its Fourier series: each circle turns at a whole-number speed,
 * and their sum traces the curve. Fewer terms round it off; a square's corners ring (Gibbs). It turns on its own while
 * on screen.
 */
export function FourierEpicycles() {
  const s = useFigureState({
    shape: choice(['hilbert', 'heart', 'star', 'square'], 'hilbert', { label: 'shape' }),
    terms: int(120, { min: 1, max: 300, label: 'circles' }),
  })
  const [theta, setTheta] = useState(0)
  const box = useRef<HTMLDivElement>(null)
  const shown = useOnScreen(box)
  useFrames(shown, (_t, dt) => setTheta((th) => (th + (2 * Math.PI * dt) / PERIOD) % (2 * Math.PI)), 40)
  const terms = SERIES[s.shape].slice(0, s.terms)
  const centres = chain(terms, theta)
  const pen = centres[centres.length - 1]
  // The pen's path over the last part of a loop (TRAIL of it), behind it.
  const trail = grid(theta - 2 * Math.PI * TRAIL, theta, Math.round(700 * TRAIL)).map((th) => chain(terms, th).at(-1)!)
  // The trail in pieces counted back from the pen (piece 0 the newest), sharing an end point so they join up.
  const blue = seriesColor(useTheme().resolved, 0)
  const size = Math.ceil(trail.length / PIECES)
  const pieces = Array.from({ length: PIECES }, (_, k) => {
    const to = trail.length - k * size
    const span = trail.slice(Math.max(0, to - size - 1), to)
    return { span, color: withAlpha(blue, (1 - k / PIECES) ** 2.2) }
  }).filter((piece) => piece.span.length > 1)
  const rings = { x: [] as number[], y: [] as number[] }
  terms.forEach((t, i) => {
    const [cx, cy] = centres[i]
    for (const a of circle) {
      rings.x.push(cx + t.amp * Math.cos(a))
      rings.y.push(cy + t.amp * Math.sin(a))
    }
    rings.x.push(NaN)
    rings.y.push(NaN)
  })
  const shape = SHAPES[s.shape]
  const x = useAxis({ label: 'x', range: [-4.2, 4.2] })
  const y = useAxis({ label: 'y', range: [-4.2, 4.2], equal: x })
  return (
    <div ref={box}>
      <Figure
        title="Draw with circles"
        purpose="A Fourier series as a chain of rotating circles; the last one's tip traces the shape."
        state={s}
        hoverReadout={false}
        defaultSize="L"
        aspect={0.9}
        controlsCollapsed
        caption="Fewer circles round the shape off; the square's corners ring however many you add."
      >
        <Plot x={x} y={y}>
          <Curve
            name="shape"
            x={[...shape.map((p) => p[0]), shape[0][0]]}
            y={[...shape.map((p) => p[1]), shape[0][1]]}
            muted
            dashed
            silent
          />
          <Curve name="circles" x={rings.x} y={rings.y} muted thin silent live />
          <Segments
            segments={centres.slice(1).map((c, i) => ({ from: [centres[i][0], centres[i][1]], to: [c[0], c[1]] }))}
            emphasis
            live
          />
          {pieces.map((piece, k) => (
            <Curve
              key={k}
              x={piece.span.map((p) => p[0])}
              y={piece.span.map((p) => p[1])}
              color={piece.color}
              silent
              live
            />
          ))}
          <Points name="tip" x={[pen[0]]} y={[pen[1]]} emphasis live />
        </Plot>
      </Figure>
    </div>
  )
}
