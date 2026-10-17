/** aifn-compute/nn/training: `parameterEma`, the teacher update of mean teacher and DINO. */
import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { parameterEma, trainingFunctions } from 'aifn-compute/nn/training'

describe('parameterEma', () => {
  const teacher = { w: tensor([0, 4]), layers: [{ b: 1 }] }
  const student = { w: tensor([2, 0]), layers: [{ b: 3 }] }

  it('averages every leaf with weight decay on the teacher', () => {
    const out = parameterEma(teacher, student, 0.75)
    expect(Array.from(toFlat(out.w))).toEqual([0.5, 3])
    expect(out.layers[0].b).toBe(1.5)
  })

  it('decay 0 copies the student and decay 1 keeps the teacher', () => {
    expect(Array.from(toFlat(parameterEma(teacher, student, 0).w))).toEqual([2, 0])
    expect(Array.from(toFlat(parameterEma(teacher, student, 1).w))).toEqual([0, 4])
  })

  it('refuses a decay outside [0, 1] and is registered', () => {
    expect(() => parameterEma(teacher, student, 1.5)).toThrow(/decay/)
    expect(trainingFunctions.parameterEma.info.role).toBe('transform')
  })
})
