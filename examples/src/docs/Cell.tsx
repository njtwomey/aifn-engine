import { Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button, CodeEditor, cn } from 'aifn-render'
import type { DocExample, DocPackage } from './data'
import { importsOf, runExample, scopeOf, type ExampleResult, type Scope } from './run'

/** One runnable example: the imports it stands for, the code (editable), and the values its expressions give. */
export function Cell({ pkg, path, example }: { pkg: DocPackage; path: string; example: DocExample }) {
  const [code, setCode] = useState(example.code)
  const [scope, setScope] = useState<Scope | null>(null)
  const [result, setResult] = useState<ExampleResult | null>(null)
  useEffect(() => {
    let live = true
    void scopeOf(pkg, path).then((s) => {
      if (!live) return
      setScope(s)
      setResult(runExample(example.code, s))
    })
    return () => {
      live = false
    }
  }, [pkg, path, example.code])
  const run = () => scope && setResult(runExample(code, scope))
  const imports = scope ? importsOf(code, scope, `aifn-${pkg}/${path}`) : []
  return (
    <div className="overflow-hidden rounded-lg border bg-card">
      <div className="border-b px-3 py-2 text-sm font-medium">{example.title || 'Example'}</div>
      {imports.length > 0 && (
        <pre className="overflow-x-auto border-b bg-muted/40 px-3 py-1.5 font-mono text-xs text-muted-foreground">
          {imports.join('\n')}
        </pre>
      )}
      <div className="relative">
        <CodeEditor value={code} onChange={setCode} onRun={run} label={example.title || 'Example'} />
        <Button
          variant="outline"
          size="sm"
          onClick={run}
          disabled={!scope}
          title="Run (⌘↵)"
          className="absolute right-2 bottom-2 bg-card"
        >
          <Play /> Run
        </Button>
      </div>
      <pre
        aria-live="polite"
        className={cn(
          'max-h-64 overflow-auto border-t px-3 py-2 font-mono text-xs whitespace-pre-wrap',
          result && !result.ok ? 'text-destructive' : 'text-foreground',
        )}
      >
        {!result
          ? 'running…'
          : result.ok
            ? [...result.output, result.value].filter((l) => l !== '').join('\n') || '(no value)'
            : [...result.output, result.error].join('\n')}
      </pre>
    </div>
  )
}
