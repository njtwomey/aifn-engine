import { useEffect } from 'react'
import { CodeBlock } from 'aifn-render'
import { Tooltip, TooltipContent, TooltipTrigger } from 'aifn-render/ui/tooltip'
import { hrefOf, onLink } from '../shell/paths'
import { Cell } from './Cell'
import {
  ancestorsOf,
  filePath,
  nodePath,
  PACKAGES,
  TREE,
  typeRef,
  useContent,
  type DocExport,
  type DocFile,
  type DocPackage,
  type DocTreeNode,
} from './data'
import { Inline, Markdown } from './Markdown'

const count = (nodes: readonly DocTreeNode[]): number => nodes.reduce((n, x) => n + 1 + count(x.children), 0)

function NodeCards({ nodes }: { nodes: readonly DocTreeNode[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {nodes.map((n) => (
        <a
          key={n.path}
          href={hrefOf(nodePath(n))}
          onClick={onLink(nodePath(n))}
          className="space-y-1 rounded-lg border bg-card p-4 transition-colors hover:border-foreground/30"
        >
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-sm font-medium">{n.name}</span>
            <span className="text-xs text-muted-foreground">
              {n.children.length ? `${count(n.children)} modules` : ''}
              {n.examples ? `${n.children.length ? ' · ' : ''}${n.examples} examples` : ''}
            </span>
          </div>
          <p className="line-clamp-3 text-sm text-muted-foreground">{n.summary || 'Not yet described.'}</p>
        </a>
      ))}
    </div>
  )
}

/** A package's front page: what it is and its top-level nodes. */
export function PackagePage({ pkg }: { pkg: DocPackage }) {
  const p = PACKAGES[pkg]
  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <div className="font-mono text-xs text-muted-foreground">{p.name}</div>
        <h1 className="text-2xl font-semibold tracking-tight">{p.title}</h1>
        <p className="text-base text-muted-foreground">{p.blurb}</p>
      </header>
      <NodeCards nodes={TREE[pkg]} />
    </div>
  )
}

function Crumbs({ node, last = false }: { node: DocTreeNode; last?: boolean }) {
  const trail = [
    { label: PACKAGES[node.pkg].title, path: node.pkg },
    ...[...ancestorsOf(node), ...(last ? [node] : [])].map((a) => ({ label: a.name, path: nodePath(a) })),
  ]
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
      {trail.map((t) => (
        <span key={t.path} className="flex items-center gap-1">
          <a href={hrefOf(t.path)} onClick={onLink(t.path)} className="hover:text-foreground">
            {t.label}
          </a>
          <span aria-hidden>/</span>
        </span>
      ))}
    </div>
  )
}

/** A documented type's name as a link to its definition, which a hover shows; plain text for any other word. */
function TypeLink({ text, node, e }: { text: string; node: DocTreeNode; e: DocExport }) {
  const r = text === e.name ? null : typeRef(text, nodePath(node), e.file)
  if (!r) return text
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <a
            href={`${hrefOf(r.path)}#${r.name}`}
            onClick={onLink(r.path, r.name)}
            className="underline decoration-dotted underline-offset-4 hover:decoration-solid"
          />
        }
      >
        {text}
      </TooltipTrigger>
      {/* A card, not the inverted tooltip: the definition keeps its code colours (the arrow, the last child, is hidden). */}
      <TooltipContent
        side="bottom"
        align="start"
        className="block max-w-2xl space-y-2 border bg-popover p-2 text-popover-foreground shadow-lg [&>:last-child]:hidden"
      >
        <CodeBlock code={r.signature} className="max-h-96 overflow-y-auto" />
        {r.summary && <p className="px-1 text-xs text-muted-foreground">{r.summary}</p>}
      </TooltipContent>
    </Tooltip>
  )
}

/** A signature whose documented type names are links to their definitions, shown on hover. */
function Signature({ node, e }: { node: DocTreeNode; e: DocExport }) {
  return (
    <CodeBlock
      code={e.signature}
      renderToken={(text) =>
        typeRef(text, nodePath(node), e.file) ? <TypeLink text={text} node={node} e={e} /> : undefined
      }
    />
  )
}

/** A type as written in a signature, its documented type names linked. */
function TypeText({ type, node, e }: { type: string; node: DocTreeNode; e: DocExport }) {
  return type
    .split(/([A-Za-z_$][\w$]*)/)
    .map((part, i) => (i % 2 ? <TypeLink key={i} text={part} node={node} e={e} /> : part))
}

const VISIBILITY = {
  internal: 'internal: exported to the module’s other files only',
  local: 'local to this file',
} as const

/** A function's parameters (name, type, default, what it is for) and what it returns. */
function Parameters({ node, e }: { node: DocTreeNode; e: DocExport }) {
  if (!e.params.length && !e.returns) return null
  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <tbody>
          {e.params.map((p) => {
            const field = p.name.startsWith('options.')
            return (
              <tr key={p.name} className="border-b align-top last:border-b-0">
                <td className={`px-3 py-2 font-mono text-xs whitespace-nowrap ${field ? 'pl-7' : 'font-medium'}`}>
                  {field ? p.name.slice('options.'.length) : p.name}
                  {p.optional && <span className="text-muted-foreground">?</span>}
                </td>
                <td className="max-w-64 px-3 py-2 font-mono text-xs break-words text-muted-foreground">
                  <TypeText type={p.type} node={node} e={e} />
                  {p.default && p.default !== '{}' && <span className="block">= {p.default}</span>}
                </td>
                <td className="w-full px-3 py-2 text-foreground/85">
                  {p.doc ? (
                    <Inline text={p.doc} />
                  ) : (
                    <span className="text-muted-foreground italic">Not described.</span>
                  )}
                </td>
              </tr>
            )
          })}
          {e.returns && (
            <tr className="align-top">
              <td className="px-3 py-2 text-xs font-medium whitespace-nowrap text-muted-foreground uppercase">
                Returns
              </td>
              <td />
              <td className="w-full px-3 py-2 text-foreground/85">
                <Inline text={e.returns} />
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

function Export({ node, e }: { node: DocTreeNode; e: DocExport }) {
  // The section's own address: the file's page at this name.
  const at = `${nodePath(node)}/${e.file.replace(/^.*\//, '').replace(/\.ts$/, '')}`
  return (
    <div id={e.name} className="scroll-mt-8 space-y-3 pt-4">
      <div className="flex items-baseline gap-2">
        <h3 className="font-mono text-lg font-semibold tracking-tight">
          <a href={`${hrefOf(at)}#${e.name}`} onClick={onLink(at, e.name)} className="group/anchor hover:underline">
            {e.name}
            <span
              aria-hidden
              className="ml-1.5 font-normal text-muted-foreground opacity-0 group-hover/anchor:opacity-100"
            >
              #
            </span>
          </a>
        </h3>
        <span className="text-xs text-muted-foreground">
          {e.kind}
          {e.visibility !== 'public' && ` · ${VISIBILITY[e.visibility]}`}
        </span>
        <span className="ml-auto truncate font-mono text-[11px] text-muted-foreground">
          {e.file.replace(/^packages\//, '')}:{e.line}
        </span>
      </div>
      <Signature node={node} e={e} />
      {e.doc && <Markdown text={e.doc} className="space-y-2 text-base text-foreground/85" />}
      <Parameters node={node} e={e} />
      {e.examples.map((x, i) =>
        x.title ? <Cell key={i} pkg={node.pkg} path={node.path} example={x} /> : <CodeBlock key={i} code={x.code} />,
      )}
    </div>
  )
}

function FileCards({ node }: { node: DocTreeNode }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {node.files.map((f) => (
        <a
          key={f.name}
          href={hrefOf(filePath(node, f))}
          onClick={onLink(filePath(node, f))}
          className="space-y-1.5 rounded-lg border bg-card p-4 transition-colors hover:border-foreground/30"
        >
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-sm font-medium">{f.name}.ts</span>
            <span className="text-xs text-muted-foreground">
              {[
                f.key.length ? `${f.key.length} key` : '',
                f.supporting.length ? `${f.supporting.length} supporting` : '',
                f.types.length ? `${f.types.length} types` : '',
                f.examples ? `${f.examples} examples` : '',
              ]
                .filter(Boolean)
                .join(' · ')}
            </span>
          </div>
          <p className="line-clamp-3 font-mono text-xs text-muted-foreground">
            {(f.key.length ? f.key : f.supporting.length ? f.supporting : f.types).join(' · ')}
          </p>
        </a>
      ))}
    </div>
  )
}

/** A node's page: its description and examples from the source, then its children or its source files. */
export function NodePage({ node }: { node: DocTreeNode }) {
  const content = useContent(nodePath(node))
  return (
    <div className="flex flex-col gap-6" key={nodePath(node)}>
      <header className="space-y-1.5">
        <Crumbs node={node} />
        <h1 className="font-mono text-xl font-semibold tracking-tight">
          aifn-{node.pkg}/{node.path}
        </h1>
        {!content && <p className="text-base text-muted-foreground">{node.summary}</p>}
      </header>
      {content &&
        (content.doc ? (
          <Markdown text={content.doc} className="space-y-3 text-base text-foreground/85" />
        ) : (
          <p className="text-sm text-muted-foreground">Not yet described.</p>
        ))}
      {content && content.examples.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight">Examples</h2>
          {content.examples.map((x, i) =>
            x.title ? (
              <Cell key={i} pkg={node.pkg} path={node.path} example={x} />
            ) : (
              <CodeBlock key={i} code={x.code} />
            ),
          )}
        </section>
      )}
      {node.children.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight">Modules</h2>
          <NodeCards nodes={node.children} />
        </section>
      )}
      {node.files.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight">Source files</h2>
          <FileCards node={node} />
        </section>
      )}
    </div>
  )
}

/**
 * Opening a page at `#name` directly: the section exists only once the content has loaded, so scroll to it then, and
 * once more after the examples above it have run and taken their height.
 */
function useHashScroll(ready: boolean) {
  useEffect(() => {
    if (!ready || !location.hash) return
    const scroll = () =>
      document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView({ block: 'start' })
    scroll()
    const later = setTimeout(scroll, 500)
    return () => clearTimeout(later)
  }, [ready])
}

/**
 * A source file's page: an index of everything it declares, then each with its signature, doc and examples. Key
 * functions (the module's public ones) come first, then the supporting ones behind them, then the types.
 */
export function FilePage({ node, file }: { node: DocTreeNode; file: DocFile }) {
  const content = useContent(nodePath(node))
  useHashScroll(content !== null)
  const here = filePath(node, file)
  const mine = content?.exports.filter((e) => e.file.endsWith(`/${file.name}.ts`)) ?? []
  const key = mine.filter((e) => e.kind !== 'type' && e.visibility === 'public')
  const supporting = mine.filter((e) => e.kind !== 'type' && e.visibility !== 'public')
  const types = mine.filter((e) => e.kind === 'type')
  const index = (names: readonly string[], className: string) =>
    names.map((name) => (
      <a key={name} href={`${hrefOf(here)}#${name}`} onClick={onLink(here, name)} className={className}>
        {name}
      </a>
    ))
  const heading = (id: string, title: string, note?: string) => (
    <div className="space-y-1">
      <h2 id={id} className="scroll-mt-8 text-xl font-semibold tracking-tight">
        <a href={`${hrefOf(here)}#${id}`} onClick={onLink(here, id)} className="hover:underline">
          {title}
        </a>
      </h2>
      {note && <p className="text-sm text-muted-foreground">{note}</p>}
    </div>
  )
  return (
    <div className="flex flex-col gap-6" key={here}>
      <header className="space-y-2">
        <Crumbs node={node} last />
        <h1 className="font-mono text-xl font-semibold tracking-tight">{file.name}.ts</h1>
        {file.key.length > 0 && (
          <p className="font-mono text-xs text-muted-foreground">
            import {'{ … }'} from 'aifn-{node.pkg}/{node.path}'
          </p>
        )}
        <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs">
          {index(file.key, 'font-medium text-foreground hover:underline')}
          {index(file.supporting, 'text-muted-foreground hover:text-foreground')}
          {index(file.types, 'text-muted-foreground italic hover:text-foreground')}
        </div>
      </header>
      {!content && <p className="text-sm text-muted-foreground">Loading…</p>}
      {content?.fileDocs[file.name] && (
        <Markdown text={content.fileDocs[file.name]} className="space-y-3 text-base text-foreground/85" />
      )}
      {key.length > 0 && (
        <section className="space-y-12">
          {key.map((e) => (
            <Export key={e.name} node={node} e={e} />
          ))}
        </section>
      )}
      {supporting.length > 0 && (
        <section className="space-y-12 pt-6">
          {heading(
            'supporting',
            'Supporting functions',
            'Behind the functions above: not exported by the module, so not importable from outside it.',
          )}
          {supporting.map((e) => (
            <Export key={e.name} node={node} e={e} />
          ))}
        </section>
      )}
      {types.length > 0 && (
        <section className="space-y-12 pt-6">
          {heading('types', 'Types')}
          {types.map((e) => (
            <Export key={e.name} node={node} e={e} />
          ))}
        </section>
      )}
    </div>
  )
}
