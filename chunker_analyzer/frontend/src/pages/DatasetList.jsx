import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import styles from './DatasetList.module.css'

export default function DatasetList() {
  const [datasets, setDatasets] = useState(null)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    api.datasets()
      .then(setDatasets)
      .catch(e => setError(e.message))
  }, [])

  if (error) return (
    <div className="page-center">
      <span className="tag tag-red">✕ {error}</span>
    </div>
  )

  if (!datasets) return (
    <div className="page-center">
      <div className="spinner" />
      <span style={{ color: 'var(--text2)' }}>Loading datasets…</span>
    </div>
  )

  if (!datasets.length) return (
    <div className="page-center">
      <p style={{ color: 'var(--text2)' }}>
        No datasets found. Check your <code className="mono">DATA_ROOT</code> setting.
      </p>
    </div>
  )

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Datasets</h1>
        <p className={styles.subtitle}>{datasets.length} dataset{datasets.length !== 1 ? 's' : ''} found</p>
      </div>

      <div className={styles.grid}>
        {datasets.map(ds => (
          <button
            key={ds.dataset_slug}
            className={styles.card}
            onClick={() => navigate(`/dataset/${encodeURIComponent(ds.dataset_slug)}`)}
          >
            <div className={styles.cardSlug}>{ds.dataset_slug}</div>
            <div className={styles.cardMeta}>
              <span className="tag tag-blue">{ds.chunkers.length} chunker{ds.chunkers.length !== 1 ? 's' : ''}</span>
            </div>
            <ul className={styles.chunkerList}>
              {ds.chunkers.map(c => (
                <li key={c.exp} className={styles.chunkerItem}>
                  <span className={styles.chunkerName}>{c.chunker_name}</span>
                  <span className={styles.chunkerStats}>
                    {c.chunk_count != null && <span>{c.chunk_count.toLocaleString()} chunks</span>}
                    {c.query_count != null && <span>{c.query_count.toLocaleString()} queries</span>}
                  </span>
                </li>
              ))}
            </ul>
          </button>
        ))}
      </div>
    </div>
  )
}
