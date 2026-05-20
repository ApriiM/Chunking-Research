import { useEffect, useState, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../api'
import MetricsChart from '../components/MetricsChart'
import QueryRow from '../components/QueryRow'
import AspectComplexityChart from '../components/AspectComplexityChart'
import styles from './DatasetView.module.css'
import { ArrowLeft, Search } from 'lucide-react'

// pair status helper (mirrors QueryRow logic, kept in sync)
function calcQueryPairStatus(query, exps, slugExpMap, partnerQueries) {
  if (!partnerQueries) return null
  let better = 0, worse = 0
  for (const exp of exps) {
    const chunkerName = slugExpMap[exp]?.chunker_name || exp
    const mine = query.chunkers[exp]?.retrieved_relevant
    const theirs = partnerQueries[query.id]?.[chunkerName]
    if (mine === undefined || theirs === undefined) continue
    if (!mine && theirs) better++
    if (mine && !theirs) worse++
  }
  if (better === 0 && worse === 0) return 'same'
  if (better > 0  && worse === 0) return 'better'
  if (better === 0 && worse > 0)  return 'worse'
  return 'mixed'
}

export default function DatasetView() {
  const { slug } = useParams()
  const [metrics, setMetrics] = useState(null)
  const [queries, setQueries] = useState(null)
  const [metricsErr, setMetricsErr] = useState(null)
  const [queriesErr, setQueriesErr] = useState(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [pairInfo, setPairInfo] = useState(null)

  useEffect(() => {
    api.datasetMetrics(slug)
      .then(setMetrics)
      .catch(e => setMetricsErr(e.message))
    api.datasetQueries(slug)
      .then(setQueries)
      .catch(e => setQueriesErr(e.message))
    api.datasetPairInfo(slug)
      .then(setPairInfo)
      .catch(() => setPairInfo(null))
  }, [slug])

  const exps = useMemo(() => metrics?.map(m => m.exp) ?? [], [metrics])

  // slug → chunker_name map for QueryRow
  const slugExpMap = useMemo(() => {
    const m = {}
    metrics?.forEach(c => { m[c.exp] = c })
    return m
  }, [metrics])

  const filteredQueries = useMemo(() => {
    if (!queries) return []
    return queries.filter(q => {
      // text search
      if (search) {
        const s = search.toLowerCase()
        if (!q.contents.toLowerCase().includes(s) &&
            !(q.free_text_answer || '').toLowerCase().includes(s)) return false
      }
      // status filter
      if (filter !== 'all') {
        const hits = exps.filter(exp => q.chunkers[exp]?.retrieved_relevant)
        const total = exps.filter(exp => q.chunkers[exp] !== undefined).length
        if (filter === 'all_good'  && hits.length !== total) return false
        if (filter === 'all_bad'   && hits.length !== 0) return false
        if (filter === 'partial'   && (hits.length === 0 || hits.length === total)) return false
      }
      return true
    })
  }, [queries, search, filter, exps])

  const stats = useMemo(() => {
    if (!queries || !exps.length) return null
    const partnerQueries = pairInfo?.partner_queries ?? null
    const hasPair = !!partnerQueries

    // per-bucket counters: { count, better, worse, mixed, same }
    const empty = () => ({ count: 0, better: 0, worse: 0, mixed: 0, same: 0 })
    const buckets = { all: empty(), all_good: empty(), partial: empty(), all_bad: empty() }

    queries.forEach(q => {
      const total = exps.filter(exp => q.chunkers[exp] !== undefined).length
      if (!total) return
      const hits = exps.filter(exp => q.chunkers[exp]?.retrieved_relevant).length

      let bucket
      if (hits === total) bucket = 'all_good'
      else if (hits === 0) bucket = 'all_bad'
      else bucket = 'partial'

      const pairStatus = hasPair
        ? calcQueryPairStatus(q, exps, slugExpMap, partnerQueries)
        : null

      for (const b of [bucket, 'all']) {
        buckets[b].count++
        if (pairStatus) {
          if      (pairStatus === 'better') buckets[b].better++
          else if (pairStatus === 'worse')  buckets[b].worse++
          else if (pairStatus === 'mixed')  buckets[b].mixed++
          else                               buckets[b].same++
        }
      }
    })

    return {
      total: queries.length,
      allGood: buckets.all_good.count,
      partial: buckets.partial.count,
      allBad:  buckets.all_bad.count,
      hasPair,
      buckets,
    }
  }, [queries, exps, pairInfo, slugExpMap])

  return (
    <div className={styles.container}>
      {/* breadcrumb */}
      <div className={styles.breadcrumb}>
        <Link to="/" className={styles.backLink}>
          <ArrowLeft size={14} /> All datasets
        </Link>
        <span className={styles.sep}>/</span>
        <span className={styles.current}>{slug}</span>
      </div>

      <h1 className={styles.title}>{slug}</h1>

      {/* ── METRICS ── */}
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>Metrics</h2>
        {metricsErr && <div className={styles.error}>{metricsErr}</div>}
        {!metrics && !metricsErr && (
          <div className={styles.loading}><div className="spinner" /> Loading metrics…</div>
        )}
        {metrics && (
          <>
            {/* summary table */}
            <div className={styles.metricsTable}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Chunker</th>
                    <th>Acc@1</th>
                    <th>MRR@10</th>
                    <th>NDCG@10</th>
                    <th>Recall@10</th>
                    <th>Chunks</th>
                    <th>Queries</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map(m => (
                    <tr key={m.exp}>
                      <td className={styles.chunkerCell}>{m.chunker_name}</td>
                      <td className="mono">{m.metrics?.['Accuracy@1']?.toFixed(2) ?? '—'}</td>
                      <td className="mono">{m.metrics?.['MRR@10']?.toFixed(2) ?? '—'}</td>
                      <td className="mono">{m.metrics?.['NDCG@10']?.toFixed(2) ?? '—'}</td>
                      <td className="mono">{m.metrics?.['Recall@10']?.toFixed(2) ?? '—'}</td>
                      <td className="mono">{m.chunk_count?.toLocaleString() ?? '—'}</td>
                      <td className="mono">{m.query_count?.toLocaleString() ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <MetricsChart chunkers={metrics} />
          </>
        )}
      </section>

      {/* ── ASPECT / COMPLEXITY BREAKDOWN ── */}
      {queries && exps.length > 0 && (
        <AspectComplexityChart
          queries={queries}
          exps={exps}
          slugExpMap={slugExpMap}
        />
      )}

      {/* ── QUERIES ── */}
      <section className={styles.section}>
        <div className={styles.queriesHeader}>
          <h2 className={styles.sectionTitle}>Queries</h2>
          {stats && (
            <div className={styles.statsBadges}>
              <span className="tag tag-green">✓ {stats.allGood}</span>
              <span className="tag tag-yellow">~ {stats.partial}</span>
              <span className="tag tag-red">✕ {stats.allBad}</span>
              <span className="tag" style={{background:'var(--bg3)',color:'var(--text2)',border:'1px solid var(--border)'}}>
                {stats.total} total
              </span>
            </div>
          )}
        </div>

        {/* controls */}
        <div className={styles.controls}>
          <div className={styles.searchBox}>
            <Search size={13} className={styles.searchIcon} />
            <input
              className={styles.searchInput}
              placeholder="Search queries or answers…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div className={styles.filterGroup}>
            {[
              { value: 'all',      label: 'All queries',         icon: null },
              { value: 'all_good', label: 'All chunkers hit',    icon: '✓'  },
              { value: 'partial',  label: 'Partial hit',         icon: '~'  },
              { value: 'all_bad',  label: 'All chunkers missed', icon: '✕'  },
            ].map(opt => (
              <FilterButton
                key={opt.value}
                opt={opt}
                active={filter === opt.value}
                bucket={stats?.buckets?.[opt.value]}
                hasPair={stats?.hasPair}
                onClick={() => setFilter(opt.value)}
              />
            ))}
          </div>
        </div>

        {queriesErr && <div className={styles.error}>{queriesErr}</div>}
        {!queries && !queriesErr && (
          <div className={styles.loading}><div className="spinner" /> Loading queries…</div>
        )}

        {queries && (
          <>
            <div className={styles.resultCount}>
              {filteredQueries.length} of {queries.length} queries
            </div>
            <div className={styles.queryList}>
              {filteredQueries.map(q => (
                <QueryRow
                  key={q.id}
                  query={q}
                  exps={exps}
                  slugExpMap={slugExpMap}
                  pairInfo={pairInfo}
                />
              ))}
              {filteredQueries.length === 0 && (
                <div className={styles.empty}>No queries match your filter.</div>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  )
}

// ── FilterButton ──────────────────────────────────────────────────────────────

function FilterButton({ opt, active, bucket, hasPair, onClick }) {
  const icons = { 'all_good': '✓', 'partial': '~', 'all_bad': '✕', 'all': null }

  return (
    <button
      className={`${styles.filterBtn} ${active ? styles.filterActive : ''}`}
      onClick={onClick}
    >
      <span className={styles.filterBtnTop}>
        {opt.icon && <span className={styles.filterBtnIcon}>{opt.icon}</span>}
        <span className={styles.filterBtnLabel}>{opt.label}</span>
        {bucket && (
          <span className={styles.filterBtnCount}>{bucket.count}</span>
        )}
      </span>

      {hasPair && bucket && bucket.count > 0 && (
        <span className={styles.filterBtnPairRow}>
          {bucket.better > 0 && (
            <span className={styles.pairPillBetter} title="pair better">
              ↑{bucket.better}
            </span>
          )}
          {bucket.worse > 0 && (
            <span className={styles.pairPillWorse} title="pair worse">
              ↓{bucket.worse}
            </span>
          )}
          {bucket.mixed > 0 && (
            <span className={styles.pairPillMixed} title="pair mixed">
              ↕{bucket.mixed}
            </span>
          )}
          {bucket.same > 0 && (
            <span className={styles.pairPillSame} title="same in pair">
              ={bucket.same}
            </span>
          )}
        </span>
      )}
    </button>
  )
}
