import { Figure, Plot, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Point labels',
  question: 'How do I label points in a scatter plot without overlaps?',
  explain:
    '`labels` gives each point a label (null for none); labels that would collide are dropped, keeping those with the highest `labelPriority` first.',
}

const cities = ['Dublin', 'Cork', 'Galway', 'Limerick', 'Waterford', 'Kilkenny', 'Sligo']
const lon = [-6.26, -8.47, -9.05, -8.63, -7.11, -7.25, -8.48]
const lat = [53.35, 51.9, 53.27, 52.66, 52.26, 52.65, 54.27]
const people = [1200, 210, 80, 95, 53, 27, 20]

export default function PointLabels() {
  const x = useAxis({ label: 'longitude' })
  const y = useAxis({ label: 'latitude', equal: x })
  return (
    <Figure title="Labelled points" purpose="Irish cities, labelled by size first.">
      <Plot x={x} y={y}>
        <Points name="cities" x={lon} y={lat} labels={cities} labelPriority={people} />
      </Plot>
    </Figure>
  )
}
