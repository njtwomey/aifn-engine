import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import { titanic } from 'aifn-methods/data/real'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Titanic',
  question: 'What does the Titanic table look like?',
  explain:
    '`titanic()` gives the 2201 people aboard as a table of nominal columns (`class`, `sex`, `age`) and the target `survived`. The survival rate by class and sex shows the pattern a subgroup search should find: nearly all women in first and second class and among the crew survived, under half in third class, and far fewer men in every class.',
}

const { table } = titanic()
const classes = ['1st', '2nd', '3rd', 'crew']
const column = (name: string) => table[name] as readonly (string | number)[]
const rate = (cls: string, sex: string) => {
  const aboard = column('class').flatMap((c, i) => (c === cls && column('sex')[i] === sex ? [i] : []))
  return aboard.filter((i) => column('survived')[i] === 1).length / aboard.length
}

export default function Titanic() {
  const x = useAxis({ categories: classes })
  const y = useAxis({ label: 'share who survived', range: [0, 1] })
  return (
    <Figure title="Titanic" purpose="The share who survived, by class and sex.">
      <Plot x={x} y={y}>
        {['female', 'male'].map((sex, k) => (
          <Bars
            key={sex}
            name={sex}
            x={classes.map((_, c) => c + (k - 0.5) * 0.4)}
            y={classes.map((c) => rate(c, sex))}
            width={0.4}
          />
        ))}
      </Plot>
    </Figure>
  )
}
