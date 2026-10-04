import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Providers } from 'aifn-render'
import { Shell } from './shell/Shell'
import './examples.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Providers>
      <Shell />
    </Providers>
  </StrictMode>,
)
