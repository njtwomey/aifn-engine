/**
 * The general (non-symmetric) real eigenproblem, needed for linear systems $\xvec' = \Amat\xvec$: stability of fixed
 * points, modes
 * of a linear flow, poles of a state-space model and the roots of a polynomial (its companion matrix). Balancing
 * (Parlett & Reinsch, 1969), reduction to upper Hessenberg form by stabilised elimination (EISPACK `elmhes`), and the
 * shifted QR algorithm with Francis double shifts (EISPACK `hqr`), in the form of Press et al. (2007), "Numerical
 * Recipes", 3rd ed., §11.6–11.7.
 * Eigenvectors come from inverse iteration in complex arithmetic (Golub & Van Loan, 2013, §7.6.1).
 */

import { dense, fromData, isTraced, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'
import { NotDifferentiableError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'

const { toMatrixF64 } = dense

/**
 * Balance a (1-based, $n \times n$) matrix in place with powers of 2 (Parlett & Reinsch, 1969).
 *
 * @param a The matrix as an array of rows indexed from 1 (`a[1..n][1..n]`; row 0 and column 0 are unused). Modified in
 *   place: each row is scaled by a power of 2 and its column by the inverse power (a similarity transform, so the
 *   eigenvalues are unchanged) until the row and column sums of absolute values are comparable.
 * @param n The number of rows (and columns) of the matrix.
 */
function balance(a: number[][], n: number): void {
  const RADIX = 2
  const sqrdx = RADIX * RADIX
  let done = false
  while (!done) {
    done = true
    for (let i = 1; i <= n; i++) {
      let r = 0
      let c = 0
      for (let j = 1; j <= n; j++)
        if (j !== i) {
          c += Math.abs(a[j][i])
          r += Math.abs(a[i][j])
        }
      if (c !== 0 && r !== 0) {
        let g = r / RADIX
        let f = 1
        const s = c + r
        while (c < g) {
          f *= RADIX
          c *= sqrdx
        }
        g = r * RADIX
        while (c > g) {
          f /= RADIX
          c /= sqrdx
        }
        if ((c + r) / f < 0.95 * s) {
          done = false
          g = 1 / f
          for (let j = 1; j <= n; j++) a[i][j] *= g
          for (let j = 1; j <= n; j++) a[j][i] *= f
        }
      }
    }
  }
}

/**
 * Eigenvalues of an upper Hessenberg matrix (1-based, destroyed) by the shifted QR algorithm with Francis double
 * shifts and exceptional shifts at iterations 10 and 20 (EISPACK `hqr`). Returns false if an eigenvalue needed more
 * than 60 iterations.
 *
 * @param a The upper Hessenberg matrix as an array of rows indexed from 1 (`a[1..n][1..n]`; row 0 and column 0 are
 *   unused). Overwritten by the iteration: its contents are meaningless on return.
 * @param n The number of rows (and columns) of the matrix.
 * @param wr Where the real parts of the eigenvalues are written, at indices 1 to $n$ (index 0 is unused), in the order
 *   the iteration finds them (unsorted). After a failure the entries not yet found are left as they were.
 * @param wi Where the imaginary parts of the eigenvalues are written, at indices 1 to $n$, matching `wr`. A complex
 *   pair occupies two adjacent entries with opposite signs.
 * @returns True when every eigenvalue was found; false when one needed more than 60 iterations.
 */
function hqr(a: number[][], n: number, wr: number[], wi: number[]): boolean {
  let anorm = 0
  for (let i = 1; i <= n; i++) for (let j = Math.max(i - 1, 1); j <= n; j++) anorm += Math.abs(a[i][j])
  let nn = n
  let t = 0
  let p = 0
  let q = 0
  let r = 0
  let s = 0
  let w = 0
  let x = 0
  let y = 0
  let z = 0
  while (nn >= 1) {
    let its = 0
    let l: number
    do {
      for (l = nn; l >= 2; l--) {
        s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l])
        if (s === 0) s = anorm
        if (Math.abs(a[l][l - 1]) + s === s) {
          a[l][l - 1] = 0
          break
        }
      }
      x = a[nn][nn]
      if (l === nn) {
        // One root found.
        wr[nn] = x + t
        wi[nn] = 0
        nn--
      } else {
        y = a[nn - 1][nn - 1]
        w = a[nn][nn - 1] * a[nn - 1][nn]
        if (l === nn - 1) {
          // Two roots found: the eigenvalues of the trailing 2×2 block.
          p = 0.5 * (y - x)
          q = p * p + w
          z = Math.sqrt(Math.abs(q))
          x += t
          if (q >= 0) {
            z = p + (p >= 0 ? Math.abs(z) : -Math.abs(z))
            wr[nn - 1] = wr[nn] = x + z
            if (z) wr[nn] = x - w / z
            wi[nn - 1] = wi[nn] = 0
          } else {
            wr[nn - 1] = wr[nn] = x + p
            wi[nn - 1] = -z
            wi[nn] = z
          }
          nn -= 2
        } else {
          if (its === 60) return false
          if (its === 10 || its === 20) {
            // Exceptional shift.
            t += x
            for (let i = 1; i <= nn; i++) a[i][i] -= x
            s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2])
            y = x = 0.75 * s
            w = -0.4375 * s * s
          }
          ++its
          let m: number
          for (m = nn - 2; m >= l; m--) {
            z = a[m][m]
            r = x - z
            s = y - z
            p = (r * s - w) / a[m + 1][m] + a[m][m + 1]
            q = a[m + 1][m + 1] - z - r - s
            r = a[m + 2][m + 1]
            s = Math.abs(p) + Math.abs(q) + Math.abs(r)
            p /= s
            q /= s
            r /= s
            if (m === l) break
            const u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r))
            const v = Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z) + Math.abs(a[m + 1][m + 1]))
            if (u + v === v) break
          }
          for (let i = m + 2; i <= nn; i++) {
            a[i][i - 2] = 0
            if (i !== m + 2) a[i][i - 3] = 0
          }
          // Double QR step on rows l..nn and columns m..nn.
          for (let k = m; k <= nn - 1; k++) {
            if (k !== m) {
              p = a[k][k - 1]
              q = a[k + 1][k - 1]
              r = 0
              if (k !== nn - 1) r = a[k + 2][k - 1]
              if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0) {
                p /= x
                q /= x
                r /= x
              }
            }
            const root = Math.sqrt(p * p + q * q + r * r)
            if ((s = p >= 0 ? root : -root) !== 0) {
              if (k === m) {
                if (l !== m) a[k][k - 1] = -a[k][k - 1]
              } else a[k][k - 1] = -s * x
              p += s
              x = p / s
              y = q / s
              z = r / s
              q /= p
              r /= p
              for (let j = k; j <= nn; j++) {
                p = a[k][j] + q * a[k + 1][j]
                if (k !== nn - 1) {
                  p += r * a[k + 2][j]
                  a[k + 2][j] -= p * z
                }
                a[k + 1][j] -= p * y
                a[k][j] -= p * x
              }
              const mmin = nn < k + 3 ? nn : k + 3
              for (let i = l; i <= mmin; i++) {
                p = x * a[i][k] + y * a[i][k + 1]
                if (k !== nn - 1) {
                  p += z * a[i][k + 2]
                  a[i][k + 2] -= p * r
                }
                a[i][k + 1] -= p * q
                a[i][k] -= p
              }
            }
          }
        }
      }
    } while (l < nn - 1)
  }
  return true
}

/**
 * Reduce a (1-based, $n \times n$) matrix in place to upper Hessenberg form by elimination with pivoting (EISPACK
 * `elmhes`).
 *
 * @param a The matrix as an array of rows indexed from 1 (`a[1..n][1..n]`; row 0 and column 0 are unused). Modified in
 *   place: on return it is upper Hessenberg, with the same eigenvalues and exact zeros below the subdiagonal.
 * @param n The number of rows (and columns) of the matrix.
 */
function elmhes(a: number[][], n: number): void {
  for (let m = 2; m < n; m++) {
    let x = 0
    let i = m
    for (let j = m; j <= n; j++)
      if (Math.abs(a[j][m - 1]) > Math.abs(x)) {
        x = a[j][m - 1]
        i = j
      }
    if (i !== m) {
      for (let j = m - 1; j <= n; j++) [a[i][j], a[m][j]] = [a[m][j], a[i][j]]
      for (let j = 1; j <= n; j++) [a[j][i], a[j][m]] = [a[j][m], a[j][i]]
    }
    if (x !== 0)
      for (i = m + 1; i <= n; i++) {
        let y = a[i][m - 1]
        if (y !== 0) {
          y /= x
          a[i][m - 1] = y
          for (let j = m; j <= n; j++) a[i][j] -= y * a[m][j]
          for (let j = 1; j <= n; j++) a[j][m] += y * a[j][i]
        }
      }
  }
  // Clear the multipliers stored below the subdiagonal.
  for (let i = 3; i <= n; i++) for (let j = 1; j < i - 1; j++) a[i][j] = 0
}

/** The eigenvalues and eigenvectors of a real square matrix, as complex128 tensors. */
export type Eigen = {
  /**
   * The eigenvalues, complex128 $[n]$, sorted by real part (descending), then imaginary part (descending): complex ones
   * come in adjacent conjugate pairs, the positive imaginary part first. `realPart`/`imagPart` give float64 views.
   */
  values: Tensor
  /**
   * The eigenvectors as columns, complex128 $[n, n]$: column $k$ belongs to eigenvalue $k$. Each has unit 2-norm and
   * its largest component real and positive (imaginary parts exactly 0 for a real eigenvalue). For a defective
   * eigenvalue the columns of a repeated eigenvalue coincide. Zeros when `vectors: false`.
   */
  vectors: Tensor
  /** False when the QR iteration did not converge (the eigenvalues are then unreliable). */
  converged: boolean
}

/**
 * Solve $(\Amat - \lambda\Imat)\vvec = \bvec$ in complex arithmetic by Gaussian elimination with partial pivoting; a
 * zero pivot is replaced by a tiny one, which is what inverse iteration wants (the solution then points along the
 * eigenvector).
 *
 * @param a The real matrix $\Amat$ as a row-major array of $n^2$ values; read, not modified.
 * @param n The number of rows (and columns) of $\Amat$.
 * @param lr The real part of the shift $\lambda$.
 * @param li The imaginary part of the shift $\lambda$.
 * @param br The real parts of the right-hand side $\bvec$, $n$ values; read, not modified.
 * @param bi The imaginary parts of the right-hand side $\bvec$, $n$ values; read, not modified.
 * @param tiny The pivot threshold: a pivot whose modulus is below it has its real part replaced by this value, so the
 *   elimination never divides by zero.
 * @returns `xr` and `xi`, the real and imaginary parts of the solution $\vvec$, as new arrays of $n$ values each.
 */
function shiftedSolve(a: Float64Array, n: number, lr: number, li: number, br: number[], bi: number[], tiny: number) {
  const mr = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => a[i * n + j] - (i === j ? lr : 0)))
  const mi = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? -li : 0)))
  const xr = [...br]
  const xi = [...bi]
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.hypot(mr[r][c], mi[r][c]) > Math.hypot(mr[p][c], mi[p][c])) p = r
    ;[mr[c], mr[p]] = [mr[p], mr[c]]
    ;[mi[c], mi[p]] = [mi[p], mi[c]]
    ;[xr[c], xr[p]] = [xr[p], xr[c]]
    ;[xi[c], xi[p]] = [xi[p], xi[c]]
    if (Math.hypot(mr[c][c], mi[c][c]) < tiny) mr[c][c] = tiny
    const dr = mr[c][c]
    const di = mi[c][c]
    const d2 = dr * dr + di * di
    for (let r = c + 1; r < n; r++) {
      // f = m[r][c] / m[c][c]
      const fr = (mr[r][c] * dr + mi[r][c] * di) / d2
      const fi = (mi[r][c] * dr - mr[r][c] * di) / d2
      if (fr === 0 && fi === 0) continue
      for (let j = c; j < n; j++) {
        mr[r][j] -= fr * mr[c][j] - fi * mi[c][j]
        mi[r][j] -= fr * mi[c][j] + fi * mr[c][j]
      }
      xr[r] -= fr * xr[c] - fi * xi[c]
      xi[r] -= fr * xi[c] + fi * xr[c]
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let sr = xr[r]
    let si = xi[r]
    for (let j = r + 1; j < n; j++) {
      sr -= mr[r][j] * xr[j] - mi[r][j] * xi[j]
      si -= mr[r][j] * xi[j] + mi[r][j] * xr[j]
    }
    const dr = mr[r][r]
    const di = mi[r][r]
    const d2 = dr * dr + di * di
    xr[r] = (sr * dr + si * di) / d2
    xi[r] = (si * dr - sr * di) / d2
  }
  return { xr, xi }
}

/**
 * Normalise a complex vector to unit 2-norm with its largest component real and positive.
 *
 * @param xr The real parts of the vector's components. Modified in place; left as it is when the vector is zero or not
 *   finite.
 * @param xi The imaginary parts of the vector's components, the same length as `xr`. Modified in place: imaginary
 *   parts smaller than $10^{-15}$ in magnitude after the normalisation are set to exactly 0.
 */
function normalise(xr: number[], xi: number[]): void {
  let big = 0
  let k = 0
  let s = 0
  for (let i = 0; i < xr.length; i++) {
    const m = Math.hypot(xr[i], xi[i])
    s += m * m
    if (m > big) [big, k] = [m, i]
  }
  const nrm = Math.sqrt(s)
  if (!(nrm > 0)) return
  // Multiply by conj(x_k)/|x_k| / ‖x‖ so that component k becomes real and positive.
  const cr = xr[k] / big / nrm
  const ci = -xi[k] / big / nrm
  for (let i = 0; i < xr.length; i++) {
    const r = xr[i] * cr - xi[i] * ci
    const im = xr[i] * ci + xi[i] * cr
    xr[i] = r
    xi[i] = Math.abs(im) < 1e-15 ? 0 : im
  }
}

/**
 * The eigenvalues (and, unless `vectors: false`, eigenvectors) of a real $n \times n$ matrix $\Amat$:
 * $\Amat\vvec = \lambda\vvec$. Eigenvalues come from balancing, Hessenberg reduction and the Francis double-shift QR
 * algorithm; each eigenvector from three steps of inverse iteration with the shift $\lambda$ perturbed by
 * $10^{-10} \lVert \Amat \rVert$ so the shifted system is not exactly singular.
 * Sorted by real part, then imaginary part, both descending (so the most unstable mode comes first). Not
 * differentiable: eigenvalues and eigenvectors of a general matrix may be complex, and aifn has no rule for them;
 * traced input throws `NotDifferentiableError` (use `eigh` for a symmetric matrix).
 *
 * @param a The real square matrix $\Amat$ ($n \times n$), as a tensor or nested arrays; it need not be symmetric. Every
 *   entry must be finite. Read, not modified.
 * @param options What to compute beyond the eigenvalues.
 * @param options.vectors Whether to compute the eigenvectors (default true). With false the inverse iteration is
 *   skipped and `vectors` in the result is an $n \times n$ matrix of zeros.
 * @returns The eigenvalues `values` (complex128, $n$ of them, sorted), the eigenvectors `vectors` as the columns of a
 *   complex128 $n \times n$ matrix in the same order, and `converged`, false when the QR iteration gave up.
 *
 * @example Complex eigenvalues of a rotation
 * // A quarter turn has eigenvalues ±i, so the values come back as complex numbers.
 * const { values, converged } = eig(tensor([[0, -1], [1, 0]]))
 * print('values =', values)
 * print('converged =', converged)
 *
 * @example A non-symmetric matrix with real eigenvalues
 * const { values } = eig(tensor([[2, 1], [0, 3]]), { vectors: false })
 * print('values =', values)
 */
export function eig(a: MatrixLike, { vectors = true }: { vectors?: boolean } = {}): Eigen {
  if (isTraced(a as unknown)) {
    throw new NotDifferentiableError(
      'eig',
      'eig: the general eigenproblem (possibly complex eigenvalues) has no derivative rule; use eigh for a symmetric matrix',
    )
  }
  const { data: A, m, n } = toMatrixF64(a, 'eig')
  if (m !== n) throw new ShapeError('eig', `eig: expected a square matrix, got ${m}×${n}`)
  for (let i = 0; i < A.length; i++)
    if (!Number.isFinite(A[i])) throw new NumericalError('eig', 'eig: the matrix must be finite', 'not-finite')
  const h = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i > 0 && j > 0 ? A[(i - 1) * n + (j - 1)] : 0)),
  )
  const wr = new Array<number>(n + 1).fill(0)
  const wi = new Array<number>(n + 1).fill(0)
  let converged = true
  if (n > 0) {
    balance(h, n)
    elmhes(h, n)
    converged = hqr(h, n, wr, wi)
  }
  const values = Array.from({ length: n }, (_, k) => [wr[k + 1], wi[k + 1]])
  values.sort((u, v) => v[0] - u[0] || v[1] - u[1])
  const vr = new Float64Array(n * n)
  const vi = new Float64Array(n * n)
  if (vectors && n > 0) {
    let scale = 0
    for (let i = 0; i < A.length; i++) scale = Math.max(scale, Math.abs(A[i]))
    scale = Math.max(scale, 1e-300)
    const tiny = 1e-14 * scale
    for (let k = 0; k < n; k++) {
      const [lr, li] = values[k]
      // A fixed, generic start vector: it has a component along every eigenvector almost surely.
      let xr = Array.from({ length: n }, (_, i) => 1 + 0.1 * Math.sin(1.3 * i + 0.7))
      let xi = new Array<number>(n).fill(0)
      const shift = lr + 1e-10 * scale
      for (let it = 0; it < 3; it++) {
        ;({ xr, xi } = shiftedSolve(A, n, shift, li, xr, xi, tiny))
        normalise(xr, xi)
      }
      if (li === 0) xi.fill(0)
      for (let i = 0; i < n; i++) {
        vr[i * n + k] = xr[i]
        vi[i * n + k] = xi[i]
      }
    }
  }
  const lambda = new Float64Array(2 * n)
  values.forEach(([re, im], k) => {
    lambda[2 * k] = re
    lambda[2 * k + 1] = im
  })
  const v = new Float64Array(2 * n * n)
  for (let i = 0; i < n * n; i++) {
    v[2 * i] = vr[i]
    v[2 * i + 1] = vi[i]
  }
  return {
    values: fromData(lambda, [n], 'complex128'),
    vectors: fromData(v, [n, n], 'complex128'),
    converged,
  }
}
