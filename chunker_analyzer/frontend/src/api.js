const BASE = '/api'

async function get(path) {
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json()
}

export const api = {
  datasets: () => get('/datasets'),
  datasetMetrics: (slug) => get(`/datasets/${encodeURIComponent(slug)}/metrics`),
  datasetQueries: (slug) => get(`/datasets/${encodeURIComponent(slug)}/queries`),
  queryDetail: (exp, queryId) =>
    get(`/experiments/${encodeURIComponent(exp)}/query/${encodeURIComponent(queryId)}`),
  chunk: (exp, chunkId) =>
    get(`/chunks/${encodeURIComponent(exp)}/${encodeURIComponent(chunkId)}`),
  document: (slug, docId) =>
    get(`/documents/${encodeURIComponent(slug)}/${encodeURIComponent(docId)}`),
  health: () => get('/health'),
  datasetPairInfo: (slug) => get(`/datasets/${encodeURIComponent(slug)}/pair-info`),
  chunkFullDocument: (exp, chunkId) =>
    get(`/chunks/${encodeURIComponent(exp)}/${encodeURIComponent(chunkId)}/full-document`),
  chunkRelevantQueries: (exp, chunkId) =>
    get(`/chunks/${encodeURIComponent(exp)}/${encodeURIComponent(chunkId)}/relevant-queries`),
  relevantChunkTexts: (exp, queryId) =>
    get(`/experiments/${encodeURIComponent(exp)}/query/${encodeURIComponent(queryId)}/relevant-chunk-texts`),
  similarity: (query, documents) =>
    fetch('/api/similarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, documents }),
    }).then(r => { if (!r.ok) throw new Error(`Similarity error ${r.status}`); return r.json(); }),
}
