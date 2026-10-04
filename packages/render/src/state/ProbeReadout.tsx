import type { ReactNode } from 'react'
import { formatNumber } from '@render/viz/format'
import { Readout, Readouts } from '@render/viz/Readout'
import type { ProbeModel } from './probe'

/**
 * The values at a probe, as one readout line: where the probe is, then each labelled value (numbers formatted, any
 * other node shown as given).
 */
export function ProbeReadout({
  probe,
  values,
  className,
}: {
  probe: ProbeModel
  values: Readonly<Record<string, number | ReactNode>>
  className?: string
}) {
  const at = [
    probe.x !== undefined ? `${probe.label ?? 'x'} = ${formatNumber(probe.x)}` : null,
    probe.y !== undefined ? `${probe.yLabel ?? 'y'} = ${formatNumber(probe.y)}` : null,
  ]
    .filter(Boolean)
    .join(', ')
  return (
    <Readouts className={className}>
      <span className="font-mono text-muted-foreground tabular-nums">at {at}</span>
      {Object.entries(values).map(([label, v]) => (
        <Readout key={label} label={label} value={typeof v === 'number' ? formatNumber(v) : v} />
      ))}
    </Readouts>
  )
}
