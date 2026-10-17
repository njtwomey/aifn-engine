/**
 * Cluster-then-label: a clustering of all inputs, labelled and unlabelled, turned into a classifier by the few labels
 * (Zhu and Goldberg, 2009, ch. 3; Castelli and Cover, 1995, for the mixture form, in which the unlabelled data fix the
 * components and the labels only decide which component is which class).
 *
 * Each cluster takes the class whose labelled points give it the most responsibility,
 * $c(k) = \argmax_c \sum_{i : y_i = c} r_{ik}$, with $r_{ik}$ the responsibility of cluster $k$ for labelled point $i$
 * (1 or 0 for a hard clustering). A cluster no labelled point touches stays unassigned. A point's class probabilities
 * are the responsibilities summed over the clusters of each class, $P(c \mid \xvec) = \sum_{k : c(k) = c} r_k(\xvec)$,
 * renormalised over the assigned clusters. The method rests entirely on the cluster assumption: the labels never move
 * a cluster, they only name it. Zhu and Goldberg train a supervised learner inside each cluster; this assigns one class
 * per cluster, the form used with a mixture model (any responsibilities, such as those of `gaussianMixture` in
 * `aifn-methods/unsupervised/clustering`).
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense } from 'aifn-compute/foundation/tensor'

/**
 * The class of each cluster from the responsibilities of the labelled points; see the file comment. Throws
 * `ShapeError` when the responsibilities and labels differ in rows and `DomainError` for a label that is neither $-1$
 * nor a class.
 *
 * @param responsibilities $r_{ik}$ for all $n$ points, $n \times K$ (rows of a mixture's responsibilities, or one-hot
 *   rows of a hard clustering).
 * @param labels One entry per point: its class $0, \dots, C - 1$, or $-1$ when unlabelled.
 * @param classes The number of classes $C$.
 * @returns The class of each of the $K$ clusters, $-1$ for a cluster no labelled point touches.
 *
 * @example The second cluster is named by the one labelled point that leans on it
 * const r = [[0.99, 0.01], [0.9, 0.1], [0.2, 0.8], [0.98, 0.02]]
 * print(clusterClasses(r, [0, -1, -1, 1], 2))
 */
export function clusterClasses(responsibilities: MatrixLike, labels: ArrayLike<number>, classes: number): Int32Array {
  const { data: R, m: n, n: K } = dense.toMatrixF64(responsibilities, 'clusterClasses')
  if (n !== labels.length)
    throw new ShapeError('clusterClasses', 'clusterClasses: responsibilities and labels differ in rows')
  const mass = Array.from({ length: K }, () => new Array<number>(classes).fill(0))
  for (let i = 0; i < n; i++) {
    const c = labels[i]
    if (c < 0) continue
    if (!(Number.isInteger(c) && c < classes))
      throw new DomainError('clusterClasses', 'clusterClasses: labels must be -1 or 0, ..., C - 1')
    for (let k = 0; k < K; k++) mass[k][c] += R[i * K + k]
  }
  return Int32Array.from(mass, (row) => {
    let best = -1
    for (let c = 0; c < classes; c++) if (row[c] > 0 && (best < 0 || row[c] > row[best])) best = c
    return best
  })
}

/**
 * Class probabilities from cluster responsibilities and the class of each cluster; see the file comment.
 *
 * @param responsibilities $r_k(\xvec)$ of the $m$ query points, $m \times K$.
 * @param clusterClass The class of each cluster (`clusterClasses`), $-1$ for an unassigned one.
 * @param classes The number of classes $C$.
 * @returns Row-major $m \times C$ probabilities; a row whose mass lies only on unassigned clusters is uniform.
 *
 * @example The probabilities of the points that named the clusters
 * const r = [[0.99, 0.01], [0.9, 0.1], [0.2, 0.8], [0.98, 0.02]]
 * print(clusterThenLabelProbabilities(r, clusterClasses(r, [0, -1, -1, 1], 2), 2))
 */
export function clusterThenLabelProbabilities(
  responsibilities: MatrixLike,
  clusterClass: ArrayLike<number>,
  classes: number,
): Float64Array {
  const { data: R, m, n: K } = dense.toMatrixF64(responsibilities, 'clusterThenLabelProbabilities')
  const out = new Float64Array(m * classes)
  for (let i = 0; i < m; i++) {
    let total = 0
    for (let k = 0; k < K; k++) {
      const c = clusterClass[k]
      if (c < 0) continue
      out[i * classes + c] += R[i * K + k]
      total += R[i * K + k]
    }
    for (let c = 0; c < classes; c++) out[i * classes + c] = total > 0 ? out[i * classes + c] / total : 1 / classes
  }
  return out
}
