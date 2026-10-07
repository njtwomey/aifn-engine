import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { mobyHyphenation } from 'aifn-methods/data/real/hyphenation'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Hyphenation',
  question: 'What does the Moby hyphenation dataset look like?',
  explain:
    '`mobyHyphenation(s)` gives the 8000 most common English words of 4 to 15 letters with their dictionary hyphenation points, split by word into training and test words, and the dictionary as `truth`. Longer words have more points; under the chart, the most common words with two or more points.',
}

const data = mobyHyphenation(stream(1))
const lengths = Array.from({ length: 12 }, (_, k) => k + 4)
const meanPoints = lengths.map((n) => {
  const words = data.all.words.filter((w) => w.word.length === n)
  return words.reduce((a, w) => a + w.hyphens.length, 0) / words.length
})
const hyphenated = (w: { word: string; hyphens: readonly number[] }) =>
  w.word
    .split('')
    .map((ch, i) => (w.hyphens.includes(i) ? `${ch}-` : ch))
    .join('')
const sample = data.all.words.filter((w) => w.hyphens.length > 1).slice(0, 12)

export default function Hyphenation() {
  const x = useAxis({ label: 'word length (letters)' })
  const y = useAxis({ label: 'hyphenation points per word', range: [0, undefined] })
  return (
    <Figure title="Moby hyphenation" purpose="Hyphenation points per word, by word length.">
      <Plot x={x} y={y}>
        <Bars name="mean points" x={lengths} y={meanPoints} width={0.8} />
      </Plot>
      <p className="flex flex-wrap gap-x-4 gap-y-1 px-1 font-mono text-sm">
        {sample.map((w) => (
          <span key={w.word}>{hyphenated(w)}</span>
        ))}
      </p>
    </Figure>
  )
}
