// main.jsx - Vite/React entry point. Mounts <App /> (App.jsx, the whole application)
// into the #root div in index.html. Nothing app-specific lives here - nothing to
// change in this file when adding features; it just wires up React itself.
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
