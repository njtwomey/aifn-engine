import { Figure, Pixels, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Image',
  question: 'How do I show an image or a matrix as pixels?',
  explain:
    '`Pixels` draws pixel (r, c) at (c, r), from `rgb` triples in [0, 1] or one `values` entry per pixel on a colour scale. `inverse` on y puts row 0 at the top.',
}

const W = 32
const H = 24
const rgb = Array.from({ length: W * H }, (_, i) => {
  const r = Math.floor(i / W)
  const c = i % W
  return [c / W, r / H, (r + c) % 8 < 4 ? 0.9 : 0.2]
}).flat()

export default function PixelsRecipe() {
  const x = useAxis({ label: 'column' })
  const y = useAxis({ label: 'row', inverse: true })
  return (
    <Figure title="An image" purpose="A 32 × 24 RGB image with row 0 at the top.">
      <Plot x={x} y={y}>
        <Pixels width={W} height={H} rgb={rgb} />
      </Plot>
    </Figure>
  )
}
