import { describe, expect, it } from 'vitest'
import { fieldArrows } from './vectors'

const length = (a: { from: [number, number]; to: [number, number] }) =>
  Math.hypot(a.to[0] - a.from[0], a.to[1] - a.from[1])

describe('fieldArrows', () => {
  // The rotation field (−y, x): its magnitude is the distance from the origin.
  const rotation = (x: number, y: number) => [-y, x] as const
  const square = { x: [-2, 2], y: [-2, 2], n: 5 } as const

  it('draws every arrow the same length by default, centred on its grid point, and reports the magnitude', () => {
    const arrows = fieldArrows(rotation, square)
    // 25 grid points less the origin, where the field vanishes.
    expect(arrows).toHaveLength(24)
    for (const a of arrows) {
      expect(length(a)).toBeCloseTo(0.8, 12)
      const centre = [(a.from[0] + a.to[0]) / 2, (a.from[1] + a.to[1]) / 2]
      expect(a.magnitude).toBeCloseTo(Math.hypot(centre[0], centre[1]), 12)
    }
  })

  it('draws lengths in proportion to the magnitude with length: magnitude, the longest filling `scale` of a cell', () => {
    const arrows = fieldArrows(rotation, { ...square, length: 'magnitude' })
    const longest = Math.max(...arrows.map((a) => a.magnitude))
    for (const a of arrows) expect(length(a)).toBeCloseTo((0.8 * a.magnitude) / longest, 12)
  })

  it('keeps arrows equal in grid units when the axes have different spans', () => {
    const arrows = fieldArrows(() => [1, 1], { x: [0, 10], y: [0, 1], n: 3 })
    // Spacing 5 by 0.5. Measured in grid spacings, every arrow is 0.8 long, and it points along (1/5, 1/0.5): the
    // direction the field has on a plot whose grid fills the box.
    for (const a of arrows) {
      const gx = (a.to[0] - a.from[0]) / 5
      const gy = (a.to[1] - a.from[1]) / 0.5
      expect(Math.hypot(gx, gy)).toBeCloseTo(0.8, 12)
      expect(gy / gx).toBeCloseTo(10, 10)
    }
  })
})
