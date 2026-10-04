import { Bars, Figure, Plot, Segmented, Select, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Choice',
  question: 'When do I use a dropdown, and when a segmented control?',
  explain:
    '`Select` hides its options in a menu, for long or wordy lists; `Segmented` shows two to five short options side by side. Options are values or `{ value, label }`; pickers work on strings, so map numbers through `String` (a schema `choice` takes numbers directly).',
}

const shops = { north: [12, 7, 9, 3], south: [8, 10, 4, 6], online: [15, 3, 2, 9] }

export default function ChoiceRecipe() {
  const [shop, setShop] = useState<keyof typeof shops>('north')
  const [scale, setScale] = useState('1')
  const x = useAxis({ categories: ['apples', 'pears', 'plums', 'figs'] })
  const y = useAxis({ label: 'kg', range: [0, 45] })
  return (
    <Figure
      title="A dropdown and a segmented control"
      purpose="Pick a shop from a menu and a multiplier from a row of buttons."
      controls={
        <>
          <Select
            label="shop"
            value={shop}
            onChange={setShop}
            options={[
              { value: 'north', label: 'North Street' },
              { value: 'south', label: 'South Quay' },
              { value: 'online', label: 'Online' },
            ]}
          />
          <Segmented label="multiplier" value={scale} onChange={setScale} options={['1', '2', '3']} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Bars name={shop} x={[0, 1, 2, 3]} y={shops[shop].map((v) => v * Number(scale))} />
      </Plot>
    </Figure>
  )
}
