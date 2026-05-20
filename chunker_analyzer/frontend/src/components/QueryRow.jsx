import { useState, useCallback, useMemo } from 'react'
import { api } from '../api'
import styles from './QueryRow.module.css'
import { ChevronDown, ChevronRight, ExternalLink } from 'lucide-react'

// ── pair comparison helpers ───────────────────────────────────────────────────

/**
 * Given the current query's chunkers and the partner dataset's query map,
 * compute aggregate pair status for the whole query:
 *   'same'    – every chunker behaved identically in both datasets
 *   'better'  – partner did better overall (more hits)
 *   'worse'   – partner did worse overall (fewer hits)
 *   'mixed'   – some better, some worse
 *   null      – no partner data
 */
function calcQueryPairStatus(query, exps, slugExpMap, partnerQueryData) {
  if (!partnerQueryData) return null
  let better = 0, worse = 0
  for (const exp of exps) {
    const chunkerName = slugExpMap[exp]?.chunker_name || exp
    const mine = query.chunkers[exp]?.retrieved_relevant
    const theirs = partnerQueryData[chunkerName]
    if (mine === undefined || theirs === undefined) continue
    if (!mine && theirs)  better++
    if (mine && !theirs)  worse++
  }
  if (better === 0 && worse === 0) return 'same'
  if (better > 0  && worse === 0) return 'better'
  if (better === 0 && worse > 0)  return 'worse'
  return 'mixed'
}

/** For a single chunker: null | 'same' | 'better' | 'worse' */
function calcChunkerPairStatus(myResult, partnerQueryData, chunkerName) {
  if (!partnerQueryData) return null
  const theirs = partnerQueryData[chunkerName]
  if (theirs === undefined) return null
  if (myResult === theirs) return 'same'
  if (!myResult && theirs)  return 'better'  // partner found, I missed
  return 'worse'                             // I found, partner missed
}

// ── pair status icons ─────────────────────────────────────────────────────────

function PairQueryBadge({ status, partnerSlug }) {
  if (!status || status === 'same') return null
  const label = {
    better: '↑ pair better',
    worse:  '↓ pair worse',
    mixed:  '↕ pair mixed',
  }[status]
  const cls = {
    better: styles.pairBetter,
    worse:  styles.pairWorse,
    mixed:  styles.pairMixed,
  }[status]
  return (
    <span className={`${styles.pairBadge} ${cls}`} title={`Partner: ${partnerSlug}`}>
      {label}
    </span>
  )
}

function PairChunkerBadge({ status, partnerSlug }) {
  if (!status || status === 'same') return null
  const { icon, label, cls } = {
    better: { icon: '↑', label: `found in ${partnerSlug}`,  cls: styles.pairBetter },
    worse:  { icon: '↓', label: `missed in ${partnerSlug}`, cls: styles.pairWorse  },
  }[status] ?? {}
  if (!label) return null
  return (
    <span className={`${styles.pairChunkerBadge} ${cls}`} title={label}>
      {icon} {label}
    </span>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function QueryRow({ query, exps, slugExpMap, pairInfo }) {
  const [open, setOpen] = useState(false)
  const [details, setDetails] = useState({})
  const [loading, setLoading] = useState({})

  const partnerQueryData = pairInfo?.partner_queries?.[query.id] ?? null
  const partnerSlug      = pairInfo?.partner_slug ?? null

  const allRelevant = exps.every(exp => query.chunkers[exp]?.retrieved_relevant)
  const anyRelevant = exps.some(exp => query.chunkers[exp]?.retrieved_relevant)
  const statusClass = allRelevant ? styles.allGood : anyRelevant ? styles.partial : styles.allBad

  const pairStatus = useMemo(
    () => calcQueryPairStatus(query, exps, slugExpMap, partnerQueryData),
    [query, exps, slugExpMap, partnerQueryData]
  )

  const loadDetail = useCallback(async (exp) => {
    if (details[exp] || loading[exp]) return
    setLoading(l => ({ ...l, [exp]: true }))
    try {
      const d = await api.queryDetail(exp, query.id)
      setDetails(prev => ({ ...prev, [exp]: d }))
    } catch (e) {
      setDetails(prev => ({ ...prev, [exp]: { error: e.message } }))
    } finally {
      setLoading(l => ({ ...l, [exp]: false }))
    }
  }, [details, loading, query.id])

  const toggle = () => {
    setOpen(v => !v)
    if (!open) exps.forEach(exp => loadDetail(exp))
  }

  return (
    <div className={`${styles.row} ${statusClass}`}>
      <div className={styles.header} onClick={toggle} role="button" tabIndex={0} onKeyDown={e => e.key === "Enter" && toggle()}>
        <span className={styles.chevron}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <span className={styles.queryText}>{query.contents}</span>

        {/* aspect / complexity tags */}
        {query.extra_meta && Object.keys(query.extra_meta).length > 0 && (
          <span className={styles.extraMetaTags}>
            {query.extra_meta.aspect != null && (
              <span className={`tag tag-blue ${styles.metaTag}`} title="aspect">
                {query.extra_meta.aspect}
              </span>
            )}
            {query.extra_meta.complexity != null && (
              <span className={`tag tag-yellow ${styles.metaTag}`} title="complexity">
                {query.extra_meta.complexity}
              </span>
            )}
          </span>
        )}

        <span className={styles.badges}>
          {/* pair query-level badge */}
          <PairQueryBadge status={pairStatus} partnerSlug={partnerSlug} />

          {exps.map(exp => {
            const c = query.chunkers[exp]
            if (!c) return null
            const ok = c.retrieved_relevant
            return (
              <span
                key={exp}
                className={`tag ${ok ? 'tag-green' : 'tag-red'}`}
                title={c.chunker_name}
              >
                {ok ? '✓' : '✕'} {c.chunker_name}
              </span>
            )
          })}
        </span>
      </div>

      {open && (
        <div className={styles.body}>
          {query.free_text_answer && (
            <div className={styles.answer}>
              <span className={styles.answerLabel}>Expected answer:</span>
              <span>{query.free_text_answer}</span>
            </div>
          )}

          {query.extra_meta && Object.keys(query.extra_meta).length > 0 && (
            <div className={styles.extraMetaRow}>
              {Object.entries(query.extra_meta).map(([k, v]) =>
                v != null ? (
                  <span key={k} className={styles.extraMetaItem}>
                    <span className={styles.extraMetaKey}>{k}</span>
                    <span className={styles.extraMetaVal}>{String(v)}</span>
                  </span>
                ) : null
              )}
            </div>
          )}

          {/* pair summary banner */}
          {pairStatus && pairStatus !== 'same' && partnerSlug && (
            <div className={`${styles.pairBanner} ${
              pairStatus === 'better' ? styles.pairBannerBetter :
              pairStatus === 'worse'  ? styles.pairBannerWorse  :
                                        styles.pairBannerMixed
            }`}>
              <span className={styles.pairBannerIcon}>
                {pairStatus === 'better' ? '↑' : pairStatus === 'worse' ? '↓' : '↕'}
              </span>
              Compared to <strong>{partnerSlug}</strong>:{' '}
              {pairStatus === 'better' && 'partner dataset scored better on this query'}
              {pairStatus === 'worse'  && 'partner dataset scored worse on this query'}
              {pairStatus === 'mixed'  && 'chunkers performed differently across datasets'}
            </div>
          )}

          <div className={styles.chunkerPanels}>
            {exps.map(exp => {
              const meta = slugExpMap[exp]
              const d = details[exp]
              const isLoading = loading[exp]
              const summaryData = query.chunkers[exp]
              const chunkerName = meta?.chunker_name || exp

              const chunkerPairStatus = calcChunkerPairStatus(
                summaryData?.retrieved_relevant,
                partnerQueryData,
                chunkerName
              )

              return (
                <div key={exp} className={styles.panel}>
                  <div className={styles.panelHeader}>
                    <span className={styles.panelTitle}>{chunkerName}</span>
                    <span className={styles.panelHeaderRight}>
                      {/* per-chunker pair diff badge */}
                      <PairChunkerBadge status={chunkerPairStatus} partnerSlug={partnerSlug} />
                      {summaryData && (
                        <span className={`tag ${summaryData.retrieved_relevant ? 'tag-green' : 'tag-red'}`}>
                          {summaryData.retrieved_relevant ? '✓ found' : '✕ missed'}
                        </span>
                      )}
                    </span>
                  </div>

                  {summaryData && (
                    <RelevantStats chunkerData={summaryData} exp={exp} queryId={query.id} />
                  )}

                  {isLoading && (
                    <div className={styles.loading}>
                      <div className="spinner" style={{ width: 14, height: 14 }} />
                    </div>
                  )}
                  {d?.error && <div className={styles.error}>{d.error}</div>}
                  {d && !d.error && (
                    <div className={styles.chunkList}>
                      {d.chunks.map(chunk => (
                        <ChunkItem key={chunk.id} chunk={chunk} exp={exp} datasetSlug={d.dataset_slug} />
                      ))}
                      {d.chunks.length === 0 && <div className={styles.empty}>No chunks retrieved</div>}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// ── RelevantStats ─────────────────────────────────────────────────────────────

function RelevantStats({ chunkerData, exp, queryId }) {
  const [showIds, setShowIds] = useState(false)
  const { relevant, relevant_count, chunk_count, relevant_pct } = chunkerData

  const openSimilarity = (e) => {
    e.stopPropagation()
    const url = `/similarity/${encodeURIComponent(exp)}/${encodeURIComponent(queryId)}?chunker=${encodeURIComponent(chunkerData.chunker_name || exp)}`
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  return (
    <div className={styles.relevantBlock}>
      <div className={styles.relevantRow}>
        <span className={styles.relevantLabel}>relevant chunks</span>
        <button className={styles.relevantCount} onClick={() => setShowIds(v => !v)} title="Toggle chunk IDs">
          <span className="mono">{relevant_count}</span>
          <span className={styles.relevantSep}>/</span>
          <span className="mono">{chunk_count?.toLocaleString() ?? '?'}</span>
          <span className={styles.relevantPct}>
            ({relevant_pct != null ? `${relevant_pct.toFixed(4)}%` : '—'})
          </span>
          <span className={styles.relevantToggle}>{showIds ? '▲' : '▼'}</span>
        </button>
        {relevant_count > 0 && (
          <button
            className={styles.similarityBtn}
            onClick={openSimilarity}
            title="Open similarity scores in new tab"
          >
            ⟳ similarity
          </button>
        )}
      </div>
      {showIds && relevant?.length > 0 && (
        <div className={styles.relevantIds}>
          {relevant.map(id => (
            <a key={id} href={`/chunk/${encodeURIComponent(exp)}/${encodeURIComponent(id)}`}
               target="_blank" rel="noopener noreferrer" className={styles.relevantId}>
              {id}
            </a>
          ))}
        </div>
      )}
      {showIds && (!relevant || relevant.length === 0) && (
        <div className={styles.relevantEmpty}>No relevant IDs available</div>
      )}
    </div>
  )
}

// ── ChunkItem ─────────────────────────────────────────────────────────────────

function ChunkItem({ chunk, exp, datasetSlug }) {
  const isRel = chunk.is_relevant
  const chunkUrl = `/chunk/${encodeURIComponent(exp)}/${encodeURIComponent(chunk.id)}`
  const docUrl = chunk.parent_id
    ? `/document/${encodeURIComponent(datasetSlug)}/${encodeURIComponent(chunk.parent_id)}`
    : null

  return (
    <div className={`${styles.chunk} ${isRel ? styles.chunkRel : styles.chunkNotRel}`}>
      <div className={styles.chunkMeta}>
        <span className={styles.chunkRank}>#{chunk.rank}</span>
        <span className={`tag ${isRel ? 'tag-green' : 'tag-yellow'}`}>
          {isRel ? '✓ relevant' : 'not relevant'}
        </span>
        <span className={styles.chunkScore}>
          score: <span className="mono">{chunk.score?.toFixed(4) ?? '—'}</span>
        </span>
        <span className={styles.chunkId + ' mono'}>{chunk.id}</span>
        <div className={styles.chunkLinks}>
          <a href={chunkUrl} target="_blank" rel="noopener noreferrer" className={styles.chunkLink}>
            <ExternalLink size={11} /> chunk
          </a>
          {docUrl && (
            <a href={docUrl} target="_blank" rel="noopener noreferrer" className={styles.chunkLink}>
              <ExternalLink size={11} /> document
            </a>
          )}
        </div>
      </div>
      {chunk.contents && <div className={styles.chunkText}>{chunk.contents}</div>}
    </div>
  )
}
