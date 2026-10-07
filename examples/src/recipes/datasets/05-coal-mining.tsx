import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { coalMining } from 'aifn-methods/data/real'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Coal-mining disasters',
  question: 'What does the coal-mining disasters series look like?',
  explain:
    '`coalMining()` gives the number of British coal-mine explosions that killed ten or more, per year from 1851 to 1962, with the years as `t`. The rate falls from about three a year to about one around 1890: the classic test of changepoint detection.',
}

const data = coalMining()

export default function CoalMining() {
  const x = useAxis({ label: 'year' })
  const y = useAxis({ label: 'disasters', range: [0, undefined] })
  return (
    <Figure title="Coal-mining disasters" purpose="Disasters per year in British coal mines, 1851–1962.">
      <Plot x={x} y={y}>
        <Bars name="disasters" x={toFlat(data.t!)} y={toFlat(data.x)} width={0.8} />
      </Plot>
    </Figure>
  )
}
