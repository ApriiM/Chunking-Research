import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell
} from 'recharts'
import { useMemo } from 'react'
import styles from './AspectComplexityChart.module.css'

const CHUNKER_COLORS = [
  '#5c7cfa', '#f59f00', '#63e6be', '#f03e3e', '#da77f2', '#74c0fc', '#ffa94d'
]

const HIT_ALPHA   = 'cc'  // ~80% opacity hex
const MISS_ALPHA  = '55'  // ~33% opacity hex

/** Build {categoryValue: {exp: {hit, miss}}} from queries */
function buildCategoryStats(queries, exps, categoryKey) {
  const stats = {}  // category → exp → {hit, miss}

  for (const q of queries) {
    const catVal = q.extra_meta?.[categoryKey]
    if (catVal == null) continue

    if (!stats[catVal]) stats[catVal] = {}

    for (const exp of exps) {
      const c = q.chunkers[exp]
      if (!c) continue
      if (!stats[catVal][exp]) stats[catVal][exp] = { hit: 0, miss: 0 }
      if (c.retrieved_relevant) stats[catVal][exp].hit++
      else                       stats[catVal][exp].miss++
    }
  }
  return stats
}

/** Convert stats → recharts data array */
function buildChartData(categoryStats, exps, expNames) {
  return Object.entries(categoryStats).map(([catVal, expStats]) => {
    const row = { category: catVal }
    exps.forEach((exp, i) => {
      const s = expStats[exp] || { hit: 0, miss: 0 }
      row[`${exp}__hit`]  = s.hit
      row[`${exp}__miss`] = s.miss
    })
    return row
  }).sort((a, b) => a.category.localeCompare(b.category))
}

const CustomTooltip = ({ active, payload, label, expNames }) => {
  if (!active || !payload?.length) return null
  // Group by chunker
  const byChunker = {}
  for (const p of payload) {
    const [exp, type] = p.dataKey.split('__')
    if (!byChunker[exp]) byChunker[exp] = { name: expNames[exp] || exp, hit: 0, miss: 0, color: p.fill }
    byChunker[exp][type] = p.value
  }
  return (
    <div className={styles.tooltip}>
      <div className={styles.tooltipLabel}>{label}</div>
      {Object.values(byChunker).map(c => {
        const total = c.hit + c.miss
        const pct = total ? ((c.hit / total) * 100).toFixed(1) : '—'
        return (
          <div key={c.name} className={styles.tooltipRow}>
            <span className={styles.tooltipDot} style={{ background: c.color }} />
            <span className={styles.tooltipName}>{c.name}</span>
            <span className={styles.tooltipHit}>✓{c.hit}</span>
            <span className={styles.tooltipMiss}>✕{c.miss}</span>
            <span className={styles.tooltipPct}>{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function CategoryChart({ title, data, exps, expNames, chunkerColors }) {
  if (!data.length) return null

  // Compute dynamic height based on number of categories
  const chartH = Math.max(220, data.length * 52 + 60)

  return (
    <div className={styles.chartCard}>
      <div className={styles.chartTitle}>{title}</div>
      <div className={styles.legendRow}>
        {exps.map((exp, i) => (
          <span key={exp} className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: chunkerColors[i] }} />
            {expNames[exp] || exp}
            <span className={styles.legendHit} style={{ color: chunkerColors[i] }}>hit</span>
            <span className={styles.legendMiss}>miss</span>
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={chartH}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
          barCategoryGap="20%"
          barGap={2}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: 'var(--text3)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
            allowDecimals={false}
          />
          <YAxis
            type="category"
            dataKey="category"
            width={120}
            tick={{ fill: 'var(--text2)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
          />
          <Tooltip content={<CustomTooltip expNames={expNames} />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          {exps.map((exp, i) => (
            [
              <Bar
                key={`${exp}__hit`}
                dataKey={`${exp}__hit`}
                name={`${expNames[exp] || exp} hit`}
                stackId={exp}
                fill={`${chunkerColors[i]}${HIT_ALPHA}`}
                radius={[0, 0, 0, 0]}
                maxBarSize={20}
              />,
              <Bar
                key={`${exp}__miss`}
                dataKey={`${exp}__miss`}
                name={`${expNames[exp] || exp} miss`}
                stackId={exp}
                fill={`${chunkerColors[i]}${MISS_ALPHA}`}
                radius={[0, 3, 3, 0]}
                maxBarSize={20}
              />
            ]
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function AspectComplexityChart({ queries, exps, slugExpMap }) {
  const hasExtraMeta = useMemo(
    () => queries.some(q => q.extra_meta && Object.keys(q.extra_meta).length > 0),
    [queries]
  )

  const expNames = useMemo(() => {
    const m = {}
    exps.forEach(exp => { m[exp] = slugExpMap[exp]?.chunker_name || exp })
    return m
  }, [exps, slugExpMap])

  const chunkerColors = useMemo(
    () => exps.map((_, i) => CHUNKER_COLORS[i % CHUNKER_COLORS.length]),
    [exps]
  )

  const aspectStats     = useMemo(() => buildCategoryStats(queries, exps, 'aspect'),     [queries, exps])
  const complexityStats = useMemo(() => buildCategoryStats(queries, exps, 'complexity'), [queries, exps])

  const aspectData     = useMemo(() => buildChartData(aspectStats,     exps, expNames), [aspectStats,     exps, expNames])
  const complexityData = useMemo(() => buildChartData(complexityStats, exps, expNames), [complexityStats, exps, expNames])

  if (!hasExtraMeta) return null
  if (!aspectData.length && !complexityData.length) return null

  return (
    <div className={styles.section}>
      <h2 className={styles.sectionTitle}>Breakdown by query properties</h2>
      <div className={styles.chartsGrid}>
        {aspectData.length > 0 && (
          <CategoryChart
            title="By Aspect"
            data={aspectData}
            exps={exps}
            expNames={expNames}
            chunkerColors={chunkerColors}
          />
        )}
        {complexityData.length > 0 && (
          <CategoryChart
            title="By Complexity"
            data={complexityData}
            exps={exps}
            expNames={expNames}
            chunkerColors={chunkerColors}
          />
        )}
      </div>
    </div>
  )
}
