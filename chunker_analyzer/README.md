# Chunker Analyzer

Narzędzie do jakościowej analizy chunkerów w systemach RAG. Porównuje wiele chunkerów
działających na tym samym datasecie — metryki, pytania, zwrócone chunki, powiązane dokumenty.

## Wymagania

- Python >= 3.10
- Node.js >= 18
- pip, npm

## Uruchomienie

```bash
# Z katalogu chunker-analyzer/
DATA_ROOT=../data ./start.sh
```

Aplikacja będzie dostępna pod: **http://localhost:5173**

### Zmienne środowiskowe

| Zmienna        | Domyślnie  | Opis                                      |
|----------------|------------|-------------------------------------------|
| `DATA_ROOT`    | `../data`  | Katalog główny z `pirb/` i `processed/`   |
| `BACKEND_PORT` | `8000`     | Port backendu Flask                       |
| `FRONTEND_PORT`| `5173`     | Port frontendu Vite                       |

## Struktura katalogów (oczekiwana)

```
<DATA_ROOT>/
  pirb/
    pirb_data/
      <exp_name>/
        passages/passage.json
        queries/queries.json
        metadata.json
    results/
      <exp_name>           ← plik JSON (bez rozszerzenia)
    retrieved_documents/
      <exp_name>.jsonl
  processed/
    <dataset_slug>/
      documents/documents.jsonl
```

## API backendu

| Endpoint                                     | Opis                                                |
|----------------------------------------------|-----------------------------------------------------|
| `GET /api/datasets`                          | Lista datasetów z chunkerami                        |
| `GET /api/datasets/:slug/metrics`            | Metryki wszystkich chunkerów dla datasetu           |
| `GET /api/datasets/:slug/queries`            | Pytania z flagami retrieved_relevant per chunker    |
| `GET /api/experiments/:exp/query/:id`        | Szczegóły pytania: chunki z score, treść            |
| `GET /api/chunks/:exp/:chunkId`              | Jeden chunk + tekst powiązanego dokumentu           |
| `GET /api/documents/:slug/:docId`            | Pełny dokument                                      |
| `GET /api/health`                            | Status backendu                                     |

## Widoki frontendu

- **/** — lista wszystkich datasetów z liczbą chunkerów
- **/dataset/:slug** — wykresy metryk + lista pytań z filtrowaniem
- **/chunk/:exp/:chunkId** — treść chunka + fragment dokumentu z podświetleniem
- **/document/:slug/:docId** — pełny dokument
