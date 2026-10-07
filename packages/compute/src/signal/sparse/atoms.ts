/**
 * Dictionaries and sparse vectors: reading a dictionary $\Dmat$ ($m \times k$, one atom $\dvec_j$ per column), scaling
 * its atoms to unit norm, its mutual coherence, and hard thresholding to the $s$ largest entries.
 *
 * Every function of `aifn-compute/signal/sparse` takes the dictionary with its atoms as columns, so a signal
 * $\yvec \in \reals^m$ is represented as $\yvec \approx \Dmat\xvec$ with $\xvec \in \reals^k$, and $k > m$ (an
 * overcomplete dictionary) is the usual case. Atoms need not have unit norm; the greedy pursuits score them by
 * normalised correlation either way.
 */

import type { Index, MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'

type F64 = dense.F64

/** A dictionary read into a row-major array: `m` rows (the signal length) and `k` columns (the atoms). */
export type Dictionary = { data: F64; m: Size; k: Size }

/**
 * Read a dictionary into a row-major array.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param where The caller's name, for error messages.
 * @returns The entries of $\Dmat$, row-major, with its `m` rows and `k` columns.
 */
export function readDictionary(D: MatrixLike, where: string): Dictionary {
  const { data, m, n } = dense.toMatrixF64(D, where)
  if (m === 0 || n === 0) throw new DomainError(where, `${where}: the dictionary is empty`)
  return { data: Float64Array.from(data), m, k: n }
}

/**
 * Read a signal and check its length against the dictionary's.
 *
 * @param y The signal $\yvec$, $m$ values.
 * @param m The dictionary's number of rows, which the signal's length must equal.
 * @param where The caller's name, for error messages.
 * @returns A copy of the signal's values.
 */
export function readSignal(y: VectorLike, m: Size, where: string): F64 {
  const v = Float64Array.from(dense.toF64(y, where))
  if (v.length !== m)
    throw new DomainError(where, `${where}: the signal has ${v.length} values but the dictionary's atoms have ${m}`)
  return v
}

/**
 * The Euclidean norm of every atom.
 *
 * @param d The dictionary, as `readDictionary` returns it.
 * @returns $\lVert \dvec_j \rVert$ for $j = 1, \dots, k$.
 */
export function atomNorms(d: Dictionary): F64 {
  const out = new Float64Array(d.k)
  for (let i = 0; i < d.m; i++) for (let j = 0; j < d.k; j++) out[j] += d.data[i * d.k + j] ** 2
  return out.map(Math.sqrt)
}

/**
 * Scale every atom (column) of a dictionary to unit Euclidean norm, $\dvec_j / \lVert \dvec_j \rVert$. A zero atom is
 * left as it is.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns. It is not modified.
 * @returns A new $m \times k$ dictionary whose non-zero atoms have unit norm.
 *
 * @example Normalise the atoms of a dictionary
 * const D = normaliseAtoms([[3, 1], [4, 1]])
 * print('D =', D)
 */
export function normaliseAtoms(D: MatrixLike): Tensor {
  const d = readDictionary(D, 'normaliseAtoms')
  const norms = atomNorms(d)
  const out = d.data.map((v, at) => {
    const n = norms[at % d.k]
    return n > 0 ? v / n : v
  })
  return dense.mat(out, d.m, d.k)
}

/**
 * The mutual coherence of a dictionary, $\mu(\Dmat) = \max_{i \ne j} |\dvec_i^\top \dvec_j| / (\lVert \dvec_i
 * \rVert \lVert \dvec_j \rVert)$: the largest absolute cosine between two atoms (Donoho and Elad, 2003). It is $0$ for
 * orthogonal atoms and at least $\sqrt{(k - m)/(m(k - 1))}$ for $k$ atoms in $\reals^m$ (the Welch bound). Any
 * $\xvec$ with fewer than $\frac{1}{2}(1 + 1/\mu)$ non-zeros is the unique sparsest representation of $\Dmat\xvec$,
 * and both orthogonal matching pursuit and basis pursuit recover it (Tropp, 2004).
 *
 * @param D The dictionary $\Dmat$, $m \times k$ with $k \ge 2$, atoms as columns. A zero atom is skipped.
 * @returns $\mu(\Dmat)$, between $0$ and $1$.
 *
 * @example The union of two orthonormal bases
 * // The spikes (identity) and the Haar basis of R^2: every spike meets every Haar atom at 45 degrees.
 * const s = Math.SQRT1_2
 * const D = [[1, 0, s, s], [0, 1, s, -s]]
 * print('coherence =', mutualCoherence(D))
 * print('unique below', 0.5 * (1 + 1 / mutualCoherence(D)), 'non-zeros')
 */
export function mutualCoherence(D: MatrixLike): number {
  const d = readDictionary(D, 'mutualCoherence')
  if (d.k < 2) throw new DomainError('mutualCoherence', 'mutualCoherence: the dictionary needs at least two atoms')
  const norms = atomNorms(d)
  let mu = 0
  for (let a = 0; a < d.k; a++)
    for (let b = a + 1; b < d.k; b++) {
      if (norms[a] === 0 || norms[b] === 0) continue
      let s = 0
      for (let i = 0; i < d.m; i++) s += d.data[i * d.k + a] * d.data[i * d.k + b]
      mu = Math.max(mu, Math.abs(s) / (norms[a] * norms[b]))
    }
  return mu
}

/**
 * The indices of the $s$ entries of largest magnitude, in descending order of magnitude (the earlier index first among
 * equals).
 *
 * @param x The values.
 * @param s How many indices to keep; at most the length of `x`.
 * @returns The $s$ indices.
 */
export function largest(x: ArrayLike<number>, s: Size): Index[] {
  return Array.from({ length: x.length }, (_, i) => i)
    .sort((a, b) => Math.abs(x[b]) - Math.abs(x[a]) || a - b)
    .slice(0, s)
}

/**
 * Hard thresholding $H_s(\xvec)$: keep the $s$ entries of largest magnitude and set the rest to zero. It is the
 * Euclidean projection onto the vectors with at most $s$ non-zeros, the step of iterative hard thresholding.
 *
 * @param x The vector $\xvec$.
 * @param s The number of entries to keep, a non-negative integer; $s$ at least the length of $\xvec$ keeps them all.
 * @returns A new vector with the length of $\xvec$ and at most $s$ non-zeros.
 *
 * @example Keep the two largest entries
 * print('H_2(x) =', hardThreshold([0.5, -3, 1, 2], 2))
 */
export function hardThreshold(x: VectorLike, s: Size): Tensor {
  if (!(Number.isInteger(s) && s >= 0))
    throw new DomainError('hardThreshold', `hardThreshold: s must be a non-negative integer, got ${s}`)
  const v = dense.toF64(x, 'hardThreshold')
  const out = new Float64Array(v.length)
  for (const i of largest(v, s)) out[i] = v[i]
  return dense.vec(out)
}
