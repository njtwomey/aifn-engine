declare module 'virtual:aifn-docs/tree' {
  const tree: import('../../plugins/docs').DocTree
  export default tree
}
declare module 'virtual:aifn-docs/content' {
  const content: Record<string, import('../../plugins/docs').DocContent>
  export default content
}
