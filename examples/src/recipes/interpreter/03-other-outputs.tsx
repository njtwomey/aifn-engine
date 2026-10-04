import { corePrelude } from 'aifn-compute/interpreter'
import {
  CodeEditor,
  Figure,
  Plot,
  ProgramStatus,
  Raster,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useProgram,
} from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Values, tables and images',
  question: "How do I show a program's value as a number, a table or an image?",
  explain:
    'A program returns any plain value; the page picks a view by its shape. Here a number becomes a `Readout`, rows of numbers a `Table`, and a 2-D array a `Raster`. The program runs as you type.',
}

const START = `const z = array.linspace(-2, 2, 60).map((b) => array.linspace(-3, 3, 80).map((a) => math.sin(a * b)))
const rows = [1, 2, 3, 4].map((k) => [k, k * k, math.round(math.sqrt(k) * 1000) / 1000])
return { total: stats.mean(z.flat().map(math.abs)), rows, z }
`

export default function OtherOutputs() {
  const [code, setCode] = useState(START)
  const run = useProgram(code)
  const v = (run.result?.ok ? run.result.value : {}) as { total?: number; rows?: number[][]; z?: number[][] }
  const [x, y] = [useAxis({ label: 'column' }), useAxis({ label: 'row' })]
  return (
    <Figure
      title="Three kinds of output"
      purpose="A number, a table and an image from one program."
      defaultSize="L"
      equation={
        <div className="flex flex-col gap-2 text-left">
          <CodeEditor value={code} onChange={setCode} prelude={corePrelude} errors={run.errors} label="Program" />
          <ProgramStatus run={run} />
        </div>
      }
      readouts={typeof v.total === 'number' && <Readout label="total (mean of |z|)" value={v.total.toFixed(4)} />}
    >
      {/* region */}
      <div className="grid gap-4 md:grid-cols-[1fr_2fr]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>k</TableHead>
              <TableHead>k²</TableHead>
              <TableHead>√k</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {v.rows?.map((r, i) => (
              <TableRow key={i}>
                {r.map((c, j) => (
                  <TableCell key={j}>{c}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {v.z && (
          <Plot x={x} y={y} height={260}>
            <Raster x={v.z[0].map((_, j) => j)} y={v.z.map((_, i) => i)} z={v.z} scale="diverging" />
          </Plot>
        )}
      </div>
      {/* endregion */}
    </Figure>
  )
}
