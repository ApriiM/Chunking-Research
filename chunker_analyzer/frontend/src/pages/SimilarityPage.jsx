import { useEffect, useState, useMemo, useCallback } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import styles from './SimilarityPage.module.css'
import { ArrowLeft, ArrowUpDown, ExternalLink } from 'lucide-react'

export default function SimilarityPage() {
  const { exp, queryId } = useParams()
  const [searchParams] = useSearchParams()
  const chunkerName = searchParams.get('chunker') || exp

  const [chunkData, setChunkData] = useState(null)
  const [chunkErr, setChunkErr]   = useState(null)

  // scores: { chunkId: { score_retrieval, score_reranker, loading, error } }
  const [scores, setScores]       = useState({})
  const [allLoading, setAllLoading] = useState(false)
  const [allErr, setAllErr]         = useState(null)

  const [sortBy, setSortBy]   = useState('score_reranker')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    api.relevantChunkTexts(exp, queryId)
      .then(setChunkData)
      .catch(e => setChunkErr(e.message))
  }, [exp, queryId])

  // ── compute ALL chunks at once ─────────────────────────────────────────────
  const runAll = async () => {
    if (!chunkData || allLoading) return
    setAllLoading(true)
    setAllErr(null)

    // mark all as loading
    setScores(prev => {
      const next = { ...prev }
      chunkData.chunks.forEach(c => { next[c.id] = { ...next[c.id], loading: true, error: null } })
      return next
    })

    try {
      const documents = chunkData.chunks.map(c => c.contents)
      const result = await api.similarity(chunkData.query_text, documents)
      setScores(prev => {
        const next = { ...prev }
        result.results.forEach((r, i) => {
          const chunk = chunkData.chunks[i]
          if (chunk) next[chunk.id] = {
            score_retrieval: r.score_before,
            score_reranker:  r.score_after,
            loading: false, error: null,
          }
        })
        return next
      })
    } catch (e) {
      setAllErr(e.message)
      setScores(prev => {
        const next = { ...prev }
        chunkData.chunks.forEach(c => { next[c.id] = { ...next[c.id], loading: false } })
        return next
      })
    } finally {
      setAllLoading(false)
    }
  }

  // ── compute ONE chunk ──────────────────────────────────────────────────────
  const runOne = useCallback(async (chunk) => {
    if (!chunkData) return
    setScores(prev => ({ ...prev, [chunk.id]: { ...prev[chunk.id], loading: true, error: null } }))
    try {
      const result = await api.similarity(chunkData.query_text, [chunk.contents])
      const r = result.results[0]
      setScores(prev => ({
        ...prev,
        [chunk.id]: { score_retrieval: r.score_before, score_reranker: r.score_after, loading: false, error: null },
      }))
    } catch (e) {
      setScores(prev => ({ ...prev, [chunk.id]: { ...prev[chunk.id], loading: false, error: e.message } }))
    }
  }, [chunkData])

  const toggleSort = (col) => {
    if (sortBy === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortBy(col); setSortDir('desc') }
  }

  const sortedChunks = useMemo(() => {
    if (!chunkData) return []
    return [...chunkData.chunks].sort((a, b) => {
      let va, vb
      if (sortBy === 'score_retrieval') {
        va = scores[a.id]?.score_retrieval ?? -Infinity
        vb = scores[b.id]?.score_retrieval ?? -Infinity
      } else if (sortBy === 'score_reranker') {
        va = scores[a.id]?.score_reranker ?? -Infinity
        vb = scores[b.id]?.score_reranker ?? -Infinity
      } else {
        va = a.was_retrieved ? (a.retrieval_score ?? -Infinity) : -Infinity
        vb = b.was_retrieved ? (b.retrieval_score ?? -Infinity) : -Infinity
      }
      return sortDir === 'desc' ? vb - va : va - vb
    })
  }, [chunkData, scores, sortBy, sortDir])

  const maxRetrieval = useMemo(() =>
    Math.max(1, ...Object.values(scores).map(s => s.score_retrieval ?? -Infinity).filter(v => isFinite(v))),
    [scores])
  const maxReranker = useMemo(() =>
    Math.max(1, ...Object.values(scores).map(s => s.score_reranker ?? -Infinity).filter(v => isFinite(v))),
    [scores])

  const anyScored = Object.values(scores).some(s => s.score_retrieval != null)

  if (chunkErr) return <div className="page-center"><span className="tag tag-red">✕ {chunkErr}</span></div>
  if (!chunkData) return <div className="page-center"><div className="spinner" /><span style={{color:'var(--text2)'}}>Loading…</span></div>

  return (
    <div className={styles.container}>
      <div className={styles.topBar}>
        <button className={styles.back} onClick={() => window.close()}>
          <ArrowLeft size={13} /> Close tab
        </button>
        <span className={styles.expBadge}>{chunkerName}</span>
      </div>

      {/* query */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.typeLabel}>QUERY</span>
          <span className="tag tag-blue mono" style={{fontSize:10}}>{queryId}</span>
        </div>
        <div className={styles.divider} />
        <div className={styles.queryText}>{chunkData.query_text}</div>
      </div>

      {/* controls */}
      <div className={styles.controlsBar}>
        <div className={styles.statsRow}>
          <span className={styles.statItem}>
            <span className={styles.statVal}>{chunkData.chunks.length}</span> relevant chunks
          </span>
          <span className={styles.statItem}>
            <span className={styles.statVal}>{chunkData.chunks.filter(c => c.was_retrieved).length}</span> were retrieved
          </span>
          {anyScored && (
            <span className={styles.statItem}>
              <span className={styles.statVal}>{Object.values(scores).filter(s => s.score_retrieval != null).length}</span> scored
            </span>
          )}
        </div>
        <div className={styles.actionRow}>
          {allErr && <span className={styles.simErr}>✕ {allErr}</span>}
          <button className={styles.runBtn} onClick={runAll} disabled={allLoading}>
            {allLoading
              ? <><div className="spinner" style={{width:13,height:13}} /> Computing all…</>
              : anyScored ? '↻ Recompute all' : '▶ Compute all similarity scores'
            }
          </button>
        </div>
      </div>

      {/* table */}
      <div className={styles.card}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.thChunk}>Chunk</th>
              <th className={styles.thRetrieved}>Retrieved</th>
              <SortTh label="Retrieval score" col="rank"            sortBy={sortBy} sortDir={sortDir} onSort={toggleSort} />
              <SortTh label="Score Retrieval" col="score_retrieval" sortBy={sortBy} sortDir={sortDir} onSort={toggleSort} active={anyScored} />
              <SortTh label="Score Reranker"  col="score_reranker"  sortBy={sortBy} sortDir={sortDir} onSort={toggleSort} active={anyScored} />
              <th className={styles.thAction} />
            </tr>
          </thead>
          <tbody>
            {sortedChunks.map(chunk => {
              const s = scores[chunk.id]
              const chunkUrl = `/chunk/${encodeURIComponent(exp)}/${encodeURIComponent(chunk.id)}`
              const isLoading = s?.loading
              const hasScore  = s?.score_retrieval != null

              return (
                <tr key={chunk.id} className={styles.tr}>
                  <td className={styles.tdChunk}>
                    <a href={chunkUrl} target="_blank" rel="noopener noreferrer" className={styles.chunkLink}>
                      <ExternalLink size={10} />
                      <span className="mono">{chunk.id}</span>
                    </a>
                    {chunk.contents && (
                      <div className={styles.chunkPreview}>
                        {chunk.contents.slice(0, 180)}{chunk.contents.length > 180 ? '…' : ''}
                      </div>
                    )}
                  </td>
                  <td className={styles.tdRetrieved}>
                    {chunk.was_retrieved
                      ? <span className="tag tag-green">✓ yes</span>
                      : <span className="tag tag-red">✕ no</span>
                    }
                  </td>
                  <td className={styles.tdScore}>
                    {chunk.was_retrieved && chunk.retrieval_score != null
                      ? <span className="mono">{chunk.retrieval_score.toFixed(4)}</span>
                      : <span className={styles.na}>—</span>
                    }
                  </td>
                  <td className={styles.tdScore}>
                    {isLoading ? <div className="spinner" style={{width:12,height:12}} />
                      : hasScore ? <ScoreBar value={s.score_retrieval} max={maxRetrieval} color="#5c7cfa" />
                      : s?.error  ? <span className={styles.scoreErr} title={s.error}>✕</span>
                      : <span className={styles.na}>—</span>
                    }
                  </td>
                  <td className={styles.tdScore}>
                    {isLoading ? <div className="spinner" style={{width:12,height:12}} />
                      : hasScore ? <ScoreBar value={s.score_reranker} max={maxReranker} color="#63e6be" />
                      : <span className={styles.na}>—</span>
                    }
                  </td>
                  <td className={styles.tdAction}>
                    {!isLoading && (
                      <button
                        className={styles.rowRunBtn}
                        onClick={() => runOne(chunk)}
                        title={hasScore ? 'Recompute for this chunk' : 'Compute similarity for this chunk'}
                      >
                        {hasScore ? '↻' : '▶'}
                      </button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function SortTh({ label, col, sortBy, sortDir, onSort, active = true }) {
  const isCurrent = sortBy === col
  return (
    <th
      className={`${styles.thSort} ${isCurrent ? styles.thSortActive : ''} ${!active ? styles.thDisabled : ''}`}
      onClick={() => active && onSort(col)}
      title={active ? `Sort by ${label}` : 'Run similarity first'}
    >
      <span>{label}</span>
      <ArrowUpDown size={11} className={styles.sortIcon} style={{opacity: isCurrent ? 1 : 0.35}} />
      {isCurrent && <span className={styles.sortDir}>{sortDir === 'desc' ? '↓' : '↑'}</span>}
    </th>
  )
}

function ScoreBar({ value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className={styles.scoreBarWrap}>
      <span className={`mono ${styles.scoreVal}`}>{value.toFixed(4)}</span>
      <div className={styles.scoreBarTrack}>
        <div className={styles.scoreBarFill} style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}
