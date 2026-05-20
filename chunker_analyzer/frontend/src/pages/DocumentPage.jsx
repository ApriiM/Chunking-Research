import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api'
import styles from './DocumentPage.module.css'
import { ArrowLeft } from 'lucide-react'

export default function DocumentPage() {
  const { slug, docId } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [wordCount, setWordCount] = useState(0)

  useEffect(() => {
    api.document(slug, docId)
      .then(d => {
        setData(d)
        setWordCount(d.contents?.split(/\s+/).filter(Boolean).length ?? 0)
      })
      .catch(e => setError(e.message))
  }, [slug, docId])

  if (error) return (
    <div className="page-center">
      <span className="tag tag-red">✕ {error}</span>
    </div>
  )

  if (!data) return (
    <div className="page-center">
      <div className="spinner" />
      <span style={{ color: 'var(--text2)' }}>Loading document…</span>
    </div>
  )

  return (
    <div className={styles.container}>
      <div className={styles.topBar}>
        <button className={styles.back} onClick={() => window.close()}>
          <ArrowLeft size={13} /> Close tab
        </button>
      </div>

      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.labelRow}>
            <span className={styles.typeLabel}>DOCUMENT</span>
            <span className="tag tag-blue mono">{docId}</span>
          </div>
          <div className={styles.metaRow}>
            <span className={styles.meta}>
              dataset: <span className="mono">{slug}</span>
            </span>
            <span className={styles.meta}>
              <span className="mono">{wordCount.toLocaleString()}</span> words
            </span>
            <span className={styles.meta}>
              <span className="mono">{data.contents?.length?.toLocaleString()}</span> chars
            </span>
          </div>
        </div>

        <div className={styles.divider} />

        <div className={styles.docText}>
          {data.contents || <em className={styles.empty}>No content</em>}
        </div>
      </div>
    </div>
  )
}
