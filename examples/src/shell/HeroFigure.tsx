import { normal, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { cholesky, choleskySolve, solveTriangular } from 'aifn-compute/numerics/linalg'
import { Area, Curve, Figure, Handle, Plot, slider, useAxis, useFigureState, type Vec2 } from 'aifn-render'
import { useState } from 'react'
import { grid } from '@examples/data'

const xs = grid(-5, 5, 160)
const S = tensor(xs.map((v) => [v]))
/** Fixed standard normals, one row per posterior draw, so the draws bend with the data instead of jumping. */
const Z = toRows(normal(stream('home'), 0, 1, { shape: [3, xs.length] }))
const START: Vec2[] = [
  [-3.6, -0.8],
  [-2, 1.1],
  [-0.6, 0.4],
  [1.2, -1.2],
  [3, 0.9],
]

/**
 * The exact GP posterior on the grid (Rasmussen and Williams, Algorithm 2.1) and three draws from it: the mean is
 * K*ᵀ(K + σ²I)⁻¹y and the covariance K** − VᵀV with V = L⁻¹K*.
 */
function posterior(pts: Vec2[], lengthscale: number, noiseSd: number) {
  const k = rbf({ lengthscale })
  const X = tensor(pts.map((p) => [p[0]]))
  const K = toRows(gram(k, X)).map((row, i) => row.map((v, j) => v + (i === j ? noiseSd ** 2 : 0)))
  const { L } = cholesky(tensor(K))
  const Ks = gram(k, X, S)
  const alpha = toFlat(choleskySolve(L, tensor(pts.map((p) => p[1]))))
  const KsRows = toRows(Ks)
  const V = toRows(solveTriangular(L, Ks))
  const mean = xs.map((_, j) => KsRows.reduce((a, row, i) => a + row[j] * alpha[i], 0))
  const cov = toRows(gram(k, S)).map((row, a) => row.map((v, b) => v - V.reduce((s, r) => s + r[a] * r[b], 0)))
  const sd = cov.map((row, j) => Math.sqrt(Math.max(row[j], 0)))
  const Lp = toRows(cholesky(tensor(cov)).L)
  const draws = Z.map((z) => mean.map((m, j) => Lp[j].reduce((a, l, i) => a + l * z[i], m)))
  return { mean, sd, draws }
}

/** The front page's live figure: a Gaussian process refitted on every move of a data point. */
export function HeroFigure() {
  const [pts, setPts] = useState(START)
  const s = useFigureState({
    lengthscale: slider(0.3, 3, 1.42, { label: 'lengthscale' }),
    noise: slider(0.01, 0.5, 0.312, { label: 'noise sd' }),
  })
  const { mean, sd, draws } = posterior(pts, s.lengthscale, s.noise)
  const x = useAxis({ label: 'x', range: [-5, 5] })
  const y = useAxis({ label: 'f(x)', range: [-3, 3] })
  return (
    <Figure
      title="Drag the data"
      purpose="A Gaussian process, refitted in your browser on every move."
      state={s}
      hoverReadout={false}
      caption="Drag any ink point; the mean, the ±2 sd band and three posterior draws follow."
    >
      <Plot x={x} y={y}>
        <Area
          name="± 2 sd"
          x={xs}
          y={mean.map((v, i) => v + 2 * sd[i])}
          base={mean.map((v, i) => v - 2 * sd[i])}
          slot={0}
          opacity={0.18}
          live
        />
        {draws.map((d, k) => (
          <Curve key={k} name={`draw ${k + 1}`} x={xs} y={d} slot={k + 1} live />
        ))}
        <Curve name="mean" x={xs} y={mean} slot={0} live />
        {pts.map((p, i) => (
          <Handle key={i} kind="point" at={p} onDrag={(q) => setPts((ps) => ps.map((r, j) => (j === i ? q : r)))} />
        ))}
      </Plot>
    </Figure>
  )
}
