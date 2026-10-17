/**
 * An exponential moving average of parameter trees: the teacher update of mean teacher (Tarvainen and Valpola, 2017,
 * eq. 1) and of self-distillation without labels (DINO; Caron et al., 2021, §3.1), where the teacher's weights follow
 * the student's as $\thetavec'_t = \alpha\,\thetavec'_{t-1} + (1 - \alpha)\,\thetavec_t$.
 */

import { treeZip, type Params } from 'aifn-compute/foundation/pytree'
import { add, mul } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * One step of an exponential moving average of parameters: every leaf of the result is
 * $\alpha\,\thetavec' + (1 - \alpha)\,\thetavec$, with $\thetavec'$ the teacher's leaf and $\thetavec$ the student's
 * at the same path. A decay of 0 copies the student, a decay of 1 keeps the teacher. Throws `DomainError` unless the
 * decay is in $[0, 1]$, and `ShapeError` when the trees have different numbers of leaves. Differentiable in the
 * leaves (the result is usually used as a constant).
 *
 * @param teacher The averaged parameters $\thetavec'$; not modified.
 * @param student The current parameters $\thetavec$, a tree of the same structure; not modified.
 * @param decay The weight $\alpha$ kept on the teacher, in $[0, 1]$ (0.99 to 0.999 is usual).
 * @returns A new tree of the teacher's structure holding the average.
 *
 * @example A teacher that moves a tenth of the way to the student
 * const teacher = { w: tensor([0, 0]), b: 1 }
 * const student = { w: tensor([1, 2]), b: 0 }
 * print('decay 0.9:', parameterEma(teacher, student, 0.9))
 * print('decay 0 copies the student:', parameterEma(teacher, student, 0))
 */
export function parameterEma<P extends Params>(teacher: P, student: P, decay: number): P {
  if (!(decay >= 0 && decay <= 1)) throw new DomainError('parameterEma', 'parameterEma: decay must be in [0, 1]')
  return treeZip([teacher, student], ([a, b]) => add(mul(decay, a), mul(1 - decay, b))) as P
}
