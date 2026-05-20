import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts'
import styles from './MetricsChart.module.css'

const METRIC_KEYS = [
  { key: 'Accuracy@1',  label: 'Acc@1'   },
  { key: 'MRR@10',      label: 'MRR@10'  },
  { key: 'NDCG@10',     label: 'NDCG@10' },
  { key: 'Recall@10',   label: 'Rec@10'  },
]

const CHUNKER_COLORS = [
  '#5c7cfa', '#f59f00', '#63e6be', '#f03e3e', '#da77f2', '#74c0fc', '#ffa94d'
]
const RANDOM_COLOR = '#444c6e'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const real = payload.find(p => p.dataKey === 'value')
  const rand = payload.find(p => p.dataKey === 'random')
  return (
    <div className={styles.tooltip}>
      <div className={styles.tooltipLabel}>{label}</div>
      {real && (
        <div className={styles.tooltipRow}>
          <span style={{ color: real.fill ?? real.color }}>model</span>
          <span className="mono">{real.value?.toFixed(2)}</span>
        </div>
      )}
      {rand && rand.value != null && (
        <div className={styles.tooltipRow}>
          <span style={{ color: RANDOM_COLOR }}>random</span>
          <span className="mono">{rand.value?.toFixed(2)}</span>
        </div>
      )}
    </div>
  )
}

export default function MetricsChart({ chunkers }) {
  if (!chunkers?.length) return null

  const hasRandom = chunkers.some(c => Object.keys(c.metrics_random ?? {}).length > 0)

  return (
    <div className={styles.wrapper}>
      {METRIC_KEYS.map(({ key }) => {
        const data = chunkers.map((c, i) => ({
          name: c.chunker_name,
          value: c.metrics?.[key] ?? null,
          random: c.metrics_random?.[key] ?? null,
          color: CHUNKER_COLORS[i % CHUNKER_COLORS.length],
        })).filter(d => d.value !== null)

        if (!data.length) return null

        const allVals = data.flatMap(d => [d.value, d.random].filter(v => v != null))
        const domainMax = Math.min(100, Math.ceil(Math.max(...allVals) / 10) * 10 + 5)

        return (
          <div key={key} className={styles.chartCard}>
            <div className={styles.chartTitleRow}>
              <span className={styles.chartTitle}>{key}</span>
              {hasRandom && (
                <span className={styles.randomLegend}>
                  <span className={styles.randomDot} /> random baseline
                </span>
              )}
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 36 }} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: 'var(--text2)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
                  angle={-25}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis
                  domain={[0, domainMax]}
                  tick={{ fill: 'var(--text3)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(92,124,250,0.06)' }} />
                {hasRandom && (
                  <Bar dataKey="random" radius={[2, 2, 0, 0]} maxBarSize={48} fill={RANDOM_COLOR} opacity={0.5} />
                )}
                <Bar dataKey="value" radius={[3, 3, 0, 0]} maxBarSize={48}>
                  {data.map(d => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )
      })}

      {chunkers.some(c => c.metrics?.['Accuracy@1-5']?.length) && (
        <AccuracyAtKChart chunkers={chunkers} hasRandom={hasRandom} />
      )}
    </div>
  )
}

function AccuracyAtKChart({ chunkers, hasRandom }) {
  const data = [1, 2, 3, 4, 5].map(k => {
    const row = { k: `@${k}` }
    chunkers.forEach(c => {
      const vals = c.metrics?.['Accuracy@1-5']
      if (vals) row[c.chunker_name] = vals[k - 1]
      if (hasRandom) {
        const rvals = c.metrics_random?.['Accuracy@1-5']
        if (rvals) row[`${c.chunker_name}__rand`] = rvals[k - 1]
      }
    })
    return row
  })

  return (
    <div className={styles.chartCard} style={{ gridColumn: '1 / -1' }}>
      <div className={styles.chartTitleRow}>
        <span className={styles.chartTitle}>Accuracy@1–5 (per rank)</span>
        {hasRandom && (
          <span className={styles.randomLegend}>
            <span className={styles.randomDot} /> random baseline (faded)
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 8 }} barCategoryGap="25%">
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="k" tick={{ fill: 'var(--text2)', fontSize: 12, fontFamily: 'IBM Plex Mono' }} />
          <YAxis domain={[0, 100]} tick={{ fill: 'var(--text3)', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: 'var(--bg3)', border: '1px solid var(--border2)', fontSize: 12, fontFamily: 'IBM Plex Mono' }}
            labelStyle={{ color: 'var(--text2)' }}
            itemStyle={{ color: 'var(--text)' }}
            cursor={{ fill: 'rgba(92,124,250,0.06)' }}
          />
          <Legend
            wrapperStyle={{ paddingTop: '8px', fontSize: '11px', fontFamily: 'IBM Plex Mono', color: 'var(--text2)' }}
            formatter={v => v.replace('__rand', ' (random)')}
          />
          {hasRandom && chunkers.map((c, i) =>
            c.metrics_random?.['Accuracy@1-5'] ? (
              <Bar
                key={`${c.exp}__rand`}
                dataKey={`${c.chunker_name}__rand`}
                fill={CHUNKER_COLORS[i % CHUNKER_COLORS.length]}
                opacity={0.25}
                radius={[2, 2, 0, 0]}
                maxBarSize={36}
              />
            ) : null
          )}
          {chunkers.map((c, i) => (
            <Bar
              key={c.exp}
              dataKey={c.chunker_name}
              fill={CHUNKER_COLORS[i % CHUNKER_COLORS.length]}
              radius={[3, 3, 0, 0]}
              maxBarSize={36}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
