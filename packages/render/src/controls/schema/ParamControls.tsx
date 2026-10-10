import { memo, type ReactNode } from 'react'
import {
  isActive,
  optionValue,
  type AnyValues,
  type LeafDef,
  type ParamDef,
  type ParamDefs,
  type ParamValue,
  type VariantsDef,
} from '../../state/schema'
import type { FigureState } from '../../state/useFigureState'
import { ControlGroup } from '../../layout/Controls'
import { Choice } from '../selection/Choice'
import { NumberField } from '../numeric/NumberField'
import { Slider } from '../numeric/Slider'
import { Switch } from '../base/Switch'
import { RevealToggle } from '../visual/RevealToggle'

export type ParamControlsProps = {
  defs: ParamDefs
  /** The values at this level: leaves, rows as records, variants as their typed case (`{ key, values }`). */
  values: AnyValues
  /** Set a field by path relative to `defs` (`name`, `row.name`, `family` to a case, `family.param`). */
  set: (path: string, value: ParamValue) => void
}

type Chosen = { key: string; values: AnyValues }

const isChosen = (v: unknown): v is Chosen =>
  typeof v === 'object' && v !== null && 'key' in v && 'values' in v && 'spec' in v

/** The values a `when` reads at one level: a variants field reads as its chosen case. */
const flat = (values: AnyValues): AnyValues =>
  Object.fromEntries(Object.entries(values).map(([k, v]) => [k, isChosen(v) ? v.key : v]))

/**
 * One control per field, in declaration order, hiding those whose `when` fails and those placed `onChart`. A
 * variants field draws its case picker then the case's controls; a row nests its fields. Renders a fragment, so the
 * controls join the surrounding `Controls` grid (or a Figure's controls slot). Pass a `useParams` result:
 * `<ParamControls {...p} />`.
 */
export function ParamControls({ defs, values, set }: ParamControlsProps) {
  const seen = flat(values)
  return (
    <>
      {Object.entries(defs).map(([name, def]) =>
        def.onChart || !isActive(def, seen) ? null : (
          <FieldControl key={name} name={name} def={def} value={values[name]} set={set} />
        ),
      )}
    </>
  )
}

/**
 * One field's control, re-rendered only when its definition or value changes (values keep their identity while
 * unchanged). `set` is left out of the comparison: every `set` a figure passes ends in its state's stable setter.
 */
const FieldControl = memo(FieldControlImpl, (a, b) => a.name === b.name && a.def === b.def && a.value === b.value)

function FieldControlImpl({
  name,
  def,
  value,
  set,
}: {
  name: string
  def: ParamDef
  value: unknown
  set: ParamControlsProps['set']
}): ReactNode {
  if (def.kind === 'variants') return <VariantFields name={name} def={def} value={value as Chosen} set={set} />
  if (def.kind === 'row')
    return <ParamControls defs={def.fields} values={value as AnyValues} set={(p, v) => set(`${name}.${p}`, v)} />
  return <LeafControl name={name} def={def} value={value as ParamValue} onChange={(v) => set(name, v)} />
}

function VariantFields({
  name,
  def,
  value,
  set,
}: {
  name: string
  def: VariantsDef
  value: Chosen
  set: ParamControlsProps['set']
}) {
  const options = Object.entries(def.specs).map(([key, c]) => ({ value: key, label: c.label }))
  return (
    <>
      <Choice
        label={def.choiceLabel ?? 'function'}
        value={value.key}
        onChange={(k) => set(name, k)}
        options={options}
      />
      <ParamControls
        defs={{ ...def.specs[value.key].params, ...def.shared }}
        values={value.values}
        set={(p, v) => set(`${name}.${p}`, v)}
      />
    </>
  )
}

/** The control for one leaf field: a slider, a number field, a choice, a switch or a revealing toggle. */
export function LeafControl({
  name,
  def,
  value,
  onChange,
}: {
  name: string
  def: LeafDef
  value: ParamValue
  onChange: (v: ParamValue) => void
}): ReactNode {
  const label = def.label ?? name
  switch (def.kind) {
    case 'slider':
      return (
        <Slider
          label={label}
          value={value as number}
          onChange={onChange}
          min={def.min}
          max={def.max}
          step={def.step}
          steppable={def.steppable}
          format={def.format}
        />
      )
    case 'number':
      return (
        <NumberField
          label={label}
          value={value as number}
          onChange={onChange}
          type={def.type}
          gt={def.gt}
          ge={def.ge}
          lt={def.lt}
          le={def.le}
          min={def.min}
          max={def.max}
          scale={def.scale}
          step={def.step}
          suggestions={def.suggestions}
          spacing={def.spacing}
          increment={def.increment}
          points={def.points}
          points_per_decade={def.points_per_decade}
          logTransform={def.logTransform}
          headerValue={def.headerValue}
          format={def.format}
        />
      )
    case 'choice': {
      // Choices may be numbers; the pickers work on text, mapped back to the option's own value.
      const values = def.options.map(optionValue)
      const options = def.options.map((o) =>
        typeof o === 'object' ? { ...o, value: String(o.value) } : { value: String(o) },
      )
      return (
        <Choice
          label={label}
          value={String(value)}
          onChange={(text) => onChange(values.find((v) => String(v) === text) ?? def.initial)}
          options={options}
          searchable={def.searchable}
        />
      )
    }
    case 'switch':
      return def.style === 'reveal' ? (
        <RevealToggle label={label} pressed={value as boolean} onChange={onChange} />
      ) : (
        <Switch label={label} checked={value as boolean} onChange={onChange} className="self-end pb-1" />
      )
  }
}

/**
 * The control rows of a figure state (DESIGN.md §4): each `row` and each `variants` field is one labelled row, in
 * declaration order; consecutive plain fields share an unlabelled row; fields placed `onChart` draw no control. Sits
 * in a `Controls` grid (a Figure draws it for `state`). `collapsed` starts every row collapsed that does not say
 * otherwise itself.
 */
export const FigureControls = memo(
  FigureControlsImpl,
  // A figure re-renders for its readouts and charts too; its rows depend only on the values and the schema (a schema
  // declared once, outside the component, keeps its identity, so the rows are skipped while neither changes).
  (a, b) => a.state.values === b.state.values && a.state.schema === b.state.schema && a.collapsed === b.collapsed,
)

function FigureControlsImpl({ state, collapsed = false }: { state: FigureState<ParamDefs>; collapsed?: boolean }) {
  const defs = state.schema
  const values = state.values as AnyValues
  const seen = flat(values)
  const rows: {
    key: string
    label?: ReactNode
    description?: ReactNode
    collapsible?: boolean
    defaultCollapsed?: boolean
    plain: boolean
    defs: Record<string, ParamDef>
  }[] = []
  for (const [name, def] of Object.entries(defs)) {
    if (def.onChart || !isActive(def, seen)) continue
    const last = rows[rows.length - 1]
    if (def.kind === 'row' || def.kind === 'variants')
      rows.push({
        key: name,
        label: def.label,
        description: def.description,
        collapsible: def.collapsible,
        defaultCollapsed: def.defaultCollapsed,
        plain: false,
        defs: { [name]: def },
      })
    else if (last?.plain) last.defs[name] = def
    else rows.push({ key: name, plain: true, defs: { [name]: def } })
  }
  const set = (path: string, value: ParamValue) => state.set(path, value)
  return (
    <div className="flex w-full flex-col gap-2.5">
      {rows.map((r) => (
        <ControlGroup
          key={r.key}
          title={r.label ?? 'Configuration'}
          description={r.description}
          collapsible={r.collapsible ?? true}
          defaultCollapsed={r.defaultCollapsed ?? collapsed}
        >
          <ParamControls defs={r.defs} values={values} set={set} />
        </ControlGroup>
      ))}
    </div>
  )
}
