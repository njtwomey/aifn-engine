import { Figure, Plot, Points, Segments, useAxis } from 'aifn-render'
import { forceLayout } from 'aifn-render/diagram'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { karateClub } from 'aifn-methods/data/real'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Karate club',
  question: 'What does Zachary’s karate club look like?',
  explain:
    '`karateClub()` gives 34 members as nodes of `graph` (78 friendships) and, as `y`, the club each joined when the club split. Laid out by `forceLayout`, the two factions sit on either side of a few bridging friendships: the standard test of community detection and of graph neural networks.',
}

const data = karateClub()
const at = forceLayout(data.graph.nodes, data.graph.edges, { iterations: 500 })
const club = Array.from(toFlat(data.y!))

export default function KarateClub() {
  const x = useAxis({})
  const y = useAxis({ equal: x })
  return (
    <Figure title="Zachary’s karate club" purpose="Friendships among 34 members, coloured by the club each joined.">
      <Plot x={x} y={y}>
        <Segments
          segments={data.graph.edges.map((e) => ({
            from: [at[e.from].x, at[e.from].y] as const,
            to: [at[e.to].x, at[e.to].y] as const,
          }))}
          muted
        />
        <Points x={at.map((p) => p.x)} y={at.map((p) => p.y)} group={club} groupNames={data.meta.labelNames} size={9} />
      </Plot>
    </Figure>
  )
}
