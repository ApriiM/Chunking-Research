import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api'
import styles from './ChunkPage.module.css'
import { ExternalLink, ArrowLeft, ChevronDown, ChevronRight, BookOpen, HelpCircle } from 'lucide-react'

export default function ChunkPage() {
  const { exp, chunkId } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.chunk(exp, chunkId)
      .then(setData)
      .catch(e => setError(e.message))
  }, [exp, chunkId])

  if (error) return (
    <div className="page-center">
      <span className="tag tag-red">✕ {error}</span>
    </div>
  )

  if (!data) return (
    <div className="page-center">
      <div className="spinner" />
      <span style={{ color: 'var(--text2)' }}>Loading chunk…</span>
    </div>
  )

  const docUrl = data.parent_id
    ? `/document/${encodeURIComponent(data.dataset_slug)}/${encodeURIComponent(data.parent_id)}`
    : null

  return (
    <div className={styles.container}>
      <div className={styles.topBar}>
        <button className={styles.back} onClick={() => window.close()}>
          <ArrowLeft size={13} /> Close tab
        </button>
        <span className={styles.expBadge}>{exp}</span>
      </div>

      {/* ── chunk content ── */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.labelRow}>
            <span className={styles.typeLabel}>CHUNK</span>
            <span className="tag tag-blue mono">{chunkId}</span>
          </div>
          <div className={styles.metaRow}>
            {data.parent_id && (
              <span className={styles.meta}>
                parent: <span className="mono">{data.parent_id}</span>
              </span>
            )}
            {data.original_id && (
              <span className={styles.meta}>
                original_id: <span className="mono">{data.original_id}</span>
              </span>
            )}
            {data.dataset_slug && (
              <span className={styles.meta}>
                dataset: <span className="mono">{data.dataset_slug}</span>
              </span>
            )}
          </div>
          {docUrl && (
            <a href={docUrl} target="_blank" rel="noopener noreferrer" className={styles.docLink}>
              <ExternalLink size={13} />
              Open full document in new tab
            </a>
          )}
        </div>
        <div className={styles.divider} />
        <div className={styles.chunkText}>
          {data.contents || <em className={styles.empty}>No content</em>}
        </div>
      </div>

      {/* ── parent document (lazy full load) ── */}
      {data.parent_id && (
        <ParentDocumentCard
          exp={exp}
          chunkId={chunkId}
          parentId={data.parent_id}
          chunkText={data.contents}
          preview={data.document_preview}
          totalLen={data.document_total_len}
          truncated={data.document_truncated}
        />
      )}

      {/* ── relevant queries (lazy load) ── */}
      <RelevantQueriesCard exp={exp} chunkId={chunkId} chunkContents={data?.contents} />
    </div>
  )
}

// ── ParentDocumentCard ────────────────────────────────────────────────────────

function ParentDocumentCard({ exp, chunkId, parentId, chunkText, preview, totalLen, truncated }) {
  const [fullText, setFullText]     = useState(null)
  const [loadingFull, setLoadingFull] = useState(false)
  const [fullError, setFullError]   = useState(null)

  const docText = fullText ?? preview

  const loadFull = async () => {
    setLoadingFull(true)
    setFullError(null)
    try {
      const res = await api.chunkFullDocument(exp, chunkId)
      setFullText(res.contents)
    } catch (e) {
      setFullError(e.message)
    } finally {
      setLoadingFull(false)
    }
  }

  if (!preview && !fullText) return null

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.labelRow}>
          <BookOpen size={14} className={styles.cardIcon} />
          <span className={styles.typeLabel}>PARENT DOCUMENT</span>
          <span className="tag tag-blue mono">{parentId}</span>
        </div>
        <div className={styles.docMeta}>
          {totalLen != null && (
            <span className={styles.meta}>
              <span className="mono">{totalLen.toLocaleString()}</span> chars total
            </span>
          )}
          {truncated && !fullText && (
            <span className={styles.previewNote}>
              showing first 3 000 chars
            </span>
          )}
        </div>
      </div>
      <div className={styles.divider} />

      <ChunkHighlight docText={docText} chunkText={chunkText} />

      {truncated && !fullText && (
        <div className={styles.loadMoreBar}>
          {fullError && <span className={styles.loadError}>✕ {fullError}</span>}
          <button
            className={styles.loadMoreBtn}
            onClick={loadFull}
            disabled={loadingFull}
          >
            {loadingFull
              ? <><div className="spinner" style={{width:13,height:13}} /> Loading full document…</>
              : <>↓ Show full document ({totalLen?.toLocaleString()} chars)</>
            }
          </button>
        </div>
      )}
    </div>
  )
}

// ── RelevantQueriesCard ───────────────────────────────────────────────────────

function RelevantQueriesCard({ exp, chunkId, chunkContents }) {
  const [open, setOpen]         = useState(false)
  const [data, setData]         = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  const load = async () => {
    if (data || loading) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.chunkRelevantQueries(exp, chunkId)
      setData(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const toggle = () => {
    setOpen(v => !v)
    if (!open) load()
  }

  return (
    <div className={styles.card}>
      <button className={styles.lazyHeader} onClick={toggle}>
        <HelpCircle size={14} className={styles.cardIcon} />
        <span className={styles.typeLabel}>RELEVANT QUERIES</span>
        {data && (
          <span className="tag tag-blue" style={{marginLeft:8}}>
            {data.count} {data.count === 1 ? 'query' : 'queries'}
          </span>
        )}
        <span className={styles.lazyChevron}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        {!open && !data && (
          <span className={styles.lazyHint}>click to load</span>
        )}
      </button>

      {open && (
        <>
          <div className={styles.divider} />
          <div className={styles.queriesBody}>
            {loading && (
              <div className={styles.lazyLoading}>
                <div className="spinner" /> Loading relevant queries…
              </div>
            )}
            {error && <div className={styles.loadError}>✕ {error}</div>}
            {data && data.queries.length === 0 && (
              <div className={styles.emptyQueries}>
                No queries list this chunk as relevant in experiment <span className="mono">{exp}</span>.
              </div>
            )}
            {data && data.queries.map(q => (
              <QueryItemWithSimilarity
                key={q.id}
                query={q}
                exp={exp}
                chunkId={chunkId}
                chunkContents={chunkContents}
                styles={styles}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ── QueryItemWithSimilarity ──────────────────────────────────────────────────

function QueryItemWithSimilarity({ query, exp, chunkId, chunkContents, styles }) {
  const [sim, setSim]           = useState(null)   // {score_retrieval, score_reranker}
  const [simLoading, setSimLoading] = useState(false)
  const [simErr, setSimErr]         = useState(null)

  // Compute inline score for THIS chunk vs this query
  const computeInline = async (e) => {
    e.stopPropagation()
    if (simLoading || !chunkContents) return
    setSimLoading(true)
    setSimErr(null)
    try {
      const result = await api.similarity(query.contents, [chunkContents])
      const r = result.results[0]
      setSim({ score_retrieval: r.score_before, score_reranker: r.score_after })
    } catch (err) {
      setSimErr(err.message)
    } finally {
      setSimLoading(false)
    }
  }

  const openFullPage = (e) => {
    e.stopPropagation()
    const url = `/similarity/${encodeURIComponent(exp)}/${encodeURIComponent(query.id)}`
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  return (
    <div className={styles.queryItem}>
      <div className={styles.queryItemMeta}>
        <span className="mono tag tag-blue" style={{fontSize:10}}>{query.id}</span>
        {query.extra_meta?.aspect != null && (
          <span className="tag tag-blue" style={{fontSize:10}}>{query.extra_meta.aspect}</span>
        )}
        {query.extra_meta?.complexity != null && (
          <span className="tag tag-yellow" style={{fontSize:10}}>{query.extra_meta.complexity}</span>
        )}
        <span className={styles.simBtnGroup}>
          <button
            className={styles.simBtn}
            onClick={computeInline}
            disabled={simLoading || !chunkContents}
            title={sim ? 'Recompute similarity for this chunk' : 'Compute similarity: this chunk vs this query'}
          >
            {simLoading
              ? <><div className="spinner" style={{width:10,height:10,borderWidth:1.5}} /> computing…</>
              : sim ? '↻ this chunk' : '▶ this chunk'
            }
          </button>
          <button
            className={`${styles.simBtn} ${styles.simBtnSecondary}`}
            onClick={openFullPage}
            title="Open similarity page for all relevant chunks of this query"
          >
            ⊞ all chunks
          </button>
        </span>
      </div>

      <div className={styles.queryItemText}>{query.contents}</div>

      {/* inline sim result */}
      {sim && (
        <div className={styles.simInlineResult}>
          <span className={styles.simInlineItem}>
            <span className={styles.simInlineLabel}>Score Retrieval</span>
            <span className={`mono ${styles.simInlineVal}`}>{sim.score_retrieval.toFixed(4)}</span>
          </span>
          <span className={styles.simInlineSep} />
          <span className={styles.simInlineItem}>
            <span className={styles.simInlineLabel}>Score Reranker</span>
            <span className={`mono ${styles.simInlineVal} ${styles.simInlineValGreen}`}>{sim.score_reranker.toFixed(4)}</span>
          </span>
        </div>
      )}
      {simErr && <div style={{fontSize:11,color:'var(--red)',fontFamily:'var(--mono)',marginTop:4}}>✕ {simErr}</div>}

      {query.free_text_answer && (
        <div className={styles.queryItemAnswer}>
          <span className={styles.queryItemAnswerLabel}>answer:</span>
          {query.free_text_answer}
        </div>
      )}
    </div>
  )
}


// ── ChunkHighlight ────────────────────────────────────────────────────────────

function ChunkHighlight({ docText, chunkText }) {
  if (!chunkText || !docText) return <div className={styles.docText}>{docText}</div>

  const idx = docText.indexOf(chunkText)
  if (idx === -1) return <div className={styles.docText}>{docText}</div>

  return (
    <div className={styles.docText}>
      {docText.slice(0, idx)}
      <mark className={styles.highlight}>{docText.slice(idx, idx + chunkText.length)}</mark>
      {docText.slice(idx + chunkText.length)}
    </div>
  )
}
