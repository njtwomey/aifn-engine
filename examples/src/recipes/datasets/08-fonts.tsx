import { choice, Figure, Plot, Shapes, useAxis, useFigureState } from 'aifn-render'
import { toRows } from 'aifn-compute/foundation/tensor'
import { FONT_CLASSES, fonts, glyphContours } from 'aifn-methods/data/real/fonts'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Fonts',
  question: 'What does the fonts dataset look like?',
  explain:
    '`fonts({ chars })` gives 66 real fonts as rows of outline samples in dense correspondence: sample k of a letter is the same point of its outline in every font, so fonts can be averaged and interpolated. Here one letter in every font, coloured by design class (sans, humanist, serif, didone, slab, mono).',
}

const letters = 'ABCDEFGHIJLMNOPRSTUVXYZ'.split('')

export default function Fonts() {
  const s = useFigureState({ letter: choice(letters, 'R', { label: 'letter' }) })
  const data = fonts({ chars: s.letter })
  const glyph = data.meta.glyphs[0]
  const shapes = toRows(data.x).map((row, i) => ({
    contours: glyphContours(row, glyph).map((c) =>
      Array.from(c.x, (v, j) => [(i % 11) * 900 + v, -Math.floor(i / 11) * 1000 + c.y[j]] as const),
    ),
    tone: FONT_CLASSES.indexOf(data.meta.fonts[i].cls),
  }))
  const x = useAxis({})
  const y = useAxis({ equal: x })
  return (
    <Figure
      title="One letter in 66 fonts"
      purpose="Outlines in dense correspondence, by design class."
      state={s}
      defaultSize="L"
    >
      <Plot x={x} y={y}>
        <Shapes shapes={shapes} />
      </Plot>
    </Figure>
  )
}
