import { Routes, Route, NavLink, useNavigate, useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api } from './api'
import DatasetList from './pages/DatasetList'
import DatasetView from './pages/DatasetView'
import ChunkPage from './pages/ChunkPage'
import DocumentPage from './pages/DocumentPage'
import SimilarityPage from './pages/SimilarityPage'
import styles from './App.module.css'

export default function App() {
  const [health, setHealth] = useState(null)
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'error' }))
  }, [])

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>⬡</span>
          <span className={styles.logoText}>Chunker<strong>Analyzer</strong></span>
        </div>
        <div className={styles.headerRight}>
          {health && (
            <span className={`tag ${health.status === 'ok' ? 'tag-green' : 'tag-red'}`}>
              {health.status === 'ok' ? `● backend` : '✕ backend offline'}
            </span>
          )}
        </div>
      </header>
      <main className={styles.main}>
        <Routes>
          <Route path="/" element={<DatasetList />} />
          <Route path="/dataset/:slug" element={<DatasetView />} />
          <Route path="/chunk/:exp/:chunkId" element={<ChunkPage />} />
          <Route path="/document/:slug/:docId" element={<DocumentPage />} />
          <Route path="/similarity/:exp/:queryId" element={<SimilarityPage />} />
        </Routes>
      </main>
    </div>
  )
}
