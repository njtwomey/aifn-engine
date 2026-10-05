import type { ProbeModel } from '../../../state/probe'
import type { Handle } from '../../handles'
import { defineLayer, type CommonProps, type Orient } from '../layer'
import { LABEL_ROW } from './marks'

export type ProbeProps = CommonProps & {
  probe: ProbeModel
  /**
   * Heights to mark at the probe's x (e.g. the curve's value there, or several curves'), as dots on the plot. For a
   * point probe the point itself is the mark.
   */
  at?: number | readonly number[]
  /** `y`: the probe's x runs along this plot's y axis (a rotated density sharing the map's axis). */
  orient?: Orient
  /** Draw the probe without a handle (it is moved on another plot). Default true. */
  draggable?: boolean
}

/**
 * A probe on a Plot (DESIGN.md §6): a guide line (or a point with its guides) that drags, writing the probe's shared
 * coordinates, and dots marking values at the probe. Live by default: it moves by patch and never widens the axes.
 */
export const Probe = defineLayer<ProbeProps>({
  kind: 'Probe',
  live: true,
  slotted: () => false,
  legend: () => [],
  // A vertical guide's label sits above the plot area.
  margins: ({ probe, orient, draggable }) => {
    if (draggable === false || (probe.x !== undefined && probe.y !== undefined)) return {}
    const vertical =
      probe.x !== undefined
        ? orient !== 'y' && !!probe.label
        : probe.y !== undefined && orient === 'y' && !!probe.yLabel
    return vertical ? { labelRow: LABEL_ROW } : {}
  },
  build: (p, ctx) => {
    const { probe } = p
    const label = probe.label
    const flip = p.orient === 'y'
    const handles: Handle[] = []
    const series: Record<string, unknown>[] = []
    const point = probe.x !== undefined && probe.y !== undefined
    if (point) {
      handles.push({
        kind: 'point',
        at: flip ? [probe.y!, probe.x!] : [probe.x!, probe.y!],
        onDrag: ([a, b]) => probe.set(flip ? { x: b, y: a } : { x: a, y: b }),
        label,
      })
    } else if (probe.x !== undefined) {
      const x = probe.x
      handles.push(
        flip ? { kind: 'y', at: x, onDrag: probe.setX, label } : { kind: 'x', at: x, onDrag: probe.setX, label },
      )
      const heights = p.at === undefined ? [] : typeof p.at === 'number' ? [p.at] : [...p.at]
      if (heights.length)
        series.push({
          id: ctx.id,
          name: `__probe${ctx.id}`,
          type: 'scatter',
          data: heights.filter(Number.isFinite).map((h) => (flip ? [h, x] : [x, h])),
          symbol: 'circle',
          symbolSize: 8,
          silent: true,
          itemStyle: { color: ctx.color, borderColor: 'transparent' },
          z: 6,
        })
    } else if (probe.y !== undefined) {
      handles.push(
        flip
          ? { kind: 'x', at: probe.y, onDrag: probe.setY, label: probe.yLabel }
          : { kind: 'y', at: probe.y, onDrag: probe.setY, label: probe.yLabel },
      )
    }
    return { series, handles: p.draggable === false ? [] : handles }
  },
})
