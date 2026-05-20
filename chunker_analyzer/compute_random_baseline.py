"""
compute_random_baseline.py
==========================
Dopisuje klucz `metrics_random` do każdego pliku w pirb/results/.

Metryki losowego retrievalu zakładają, że ranker losowo próbkuje k chunków
bez zwracania (sampling without replacement) z puli N chunków łącznie.

Wzory dla zapytania z R relevant chunków spośród N total:
  P(hit w top-k) = 1 - C(N-R, k) / C(N, k)
  = 1 - prod_{i=0}^{k-1} (N-R-i)/(N-i)  dla R <= N-k, inaczej 1.0

  MRR@10  = sum_{k=1}^{10}  (1/k) * P(pierwszy trafiony na pozycji k)
           gdzie P(1. trafiony na poz. k) = [C(N-R,k-1)*R] / [C(N,k-1)*(N-k+1)]
           = P(hit w top-k) - P(hit w top-(k-1))  ... przez rekurencję
           Dokładniej: E[1/rank pierwszego trafienia] w modelu bez zwracania.

  Recall@10 = E[min(R, 10) / R] * P_hit  ← uproszczone;
             dokładnie: E[min(|S ∩ top10|, R)] / R
             = (1/R) * sum_{j=1}^{min(R,10)} j * P(|S ∩ top10| = j)
  
  NDCG@10 = E[DCG@10] / IDCG@10, gdzie IDCG = sum_{i=1}^{min(R,10)} 1/log2(i+1)

Wszystkie metryki uśredniane po zapytaniach w zbiorze.

Uruchomienie:
  python compute_random_baseline.py [--data-root ../data] [--dry-run]
"""

import argparse
import json
import math
from pathlib import Path


# ── combinatorics ─────────────────────────────────────────────────────────────

def log_comb(n: int, k: int) -> float:
    """log C(n, k) — bezpieczne dla dużych n."""
    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    return sum(math.log(n - i) - math.log(i + 1) for i in range(k))


def prob_at_least_one_in_topk(N: int, R: int, k: int) -> float:
    """P(co najmniej 1 z R relevant w losowym top-k bez zwracania)."""
    if R <= 0 or k <= 0 or N <= 0:
        return 0.0
    if R >= N:
        return 1.0
    k = min(k, N)
    # P = 1 - C(N-R, k) / C(N, k)
    lp_miss = log_comb(N - R, k) - log_comb(N, k)
    if lp_miss == -math.inf:
        return 1.0
    return max(0.0, 1.0 - math.exp(lp_miss))


def expected_hits_in_topk(N: int, R: int, k: int) -> float:
    """E[|relevant ∩ top-k|] = k * R / N  (hypergeometric mean)."""
    if N <= 0 or k <= 0:
        return 0.0
    return min(k, R) * min(k, N) / N  # uproszczenie: k*R/N clipped


def prob_exactly_j_in_topk(N: int, R: int, k: int, j: int) -> float:
    """P(dokładnie j relevant w top-k) — rozkład hipergeometryczny."""
    if j < 0 or j > min(R, k):
        return 0.0
    lp = log_comb(R, j) + log_comb(N - R, k - j) - log_comb(N, k)
    if lp == -math.inf:
        return 0.0
    return max(0.0, math.exp(lp))


# ── per-query metric formulas ─────────────────────────────────────────────────

def random_accuracy_at_k(N: int, R: int, k: int) -> float:
    """P(co najmniej 1 relevant w top-k) * 100."""
    return prob_at_least_one_in_topk(N, R, k) * 100.0


def random_accuracy_at_1_5(N: int, R: int) -> list[float]:
    return [random_accuracy_at_k(N, R, k) for k in range(1, 6)]


def random_recall_at_10(N: int, R: int) -> float:
    """E[|relevant ∩ top10| / R]  (Recall@10)."""
    if R <= 0:
        return 0.0
    k = min(10, N)
    expected = sum(
        j * prob_exactly_j_in_topk(N, R, k, j)
        for j in range(1, min(R, k) + 1)
    )
    return (expected / R) * 100.0


def random_mrr_at_10(N: int, R: int) -> float:
    """E[1 / rank(pierwszego relevant)] dla top-10."""
    if R <= 0 or N <= 0:
        return 0.0
    # P(first relevant at position k) = P(hit in top-k) - P(hit in top-(k-1))
    mrr = 0.0
    prev = 0.0
    for k in range(1, min(11, N + 1)):
        cur = prob_at_least_one_in_topk(N, R, k)
        p_first_at_k = cur - prev
        mrr += (1.0 / k) * p_first_at_k
        prev = cur
    return mrr * 100.0


def random_ndcg_at_10(N: int, R: int) -> float:
    """E[DCG@10] / IDCG@10."""
    if R <= 0 or N <= 0:
        return 0.0
    k = min(10, N)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(R, 10)))
    if idcg == 0:
        return 0.0

    # E[DCG@10] = sum_{pos=1}^{10} (1/log2(pos+1)) * P(relevant at pos)
    # P(relevant at pos i) = R/N  (symmetry of uniform sampling)
    # But with WITHOUT replacement from top-k, position marginals are R/N each.
    expected_dcg = sum(
        (R / N) / math.log2(pos + 2)
        for pos in range(k)
        if N > 0
    )
    return (expected_dcg / idcg) * 100.0


# ── per-experiment computation ────────────────────────────────────────────────

def compute_random_metrics(N: int, relevant_counts: list[int]) -> dict:
    """
    N: total chunk count for this experiment
    relevant_counts: list of R values (one per query)
    """
    M = len(relevant_counts)
    if M == 0:
        return {}

    acc1_sum = 0.0
    acc15_sum = [0.0] * 5
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for R in relevant_counts:
        acc1_sum += random_accuracy_at_k(N, R, 1)
        for i, v in enumerate(random_accuracy_at_1_5(N, R)):
            acc15_sum[i] += v
        recall_sum += random_recall_at_10(N, R)
        mrr_sum += random_mrr_at_10(N, R)
        ndcg_sum += random_ndcg_at_10(N, R)

    return {
        "Accuracy@1":    acc1_sum / M,
        "Accuracy@1-5":  [v / M for v in acc15_sum],
        "Recall@10":     recall_sum / M,
        "MRR@10":        mrr_sum / M,
        "NDCG@10":       ndcg_sum / M,
        "_note": f"Random baseline computed over {M} queries, N={N} chunks",
    }


# ── file I/O ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_relevant_counts(pirb_root: Path, exp: str) -> list[int]:
    """
    Try retrieved_documents/<exp>.jsonl first (has 'relevant' key per query),
    fall back to pirb_data/<exp>/queries/queries.json.
    """
    retrieved_path = pirb_root / "retrieved_documents" / f"{exp}.jsonl"
    if retrieved_path.exists():
        rows = load_jsonl(retrieved_path)
        return [len(r.get("relevant") or []) for r in rows]

    queries_path = pirb_root / "pirb_data" / exp / "queries" / "queries.json"
    if queries_path.exists():
        data = load_json(queries_path)
        rows = data if isinstance(data, list) else list(data.values())
        return [len(r.get("relevant") or []) for r in rows]

    return []


def process_experiment(pirb_root: Path, exp: str, dry_run: bool = False) -> bool:
    # ── locate result file ────────────────────────────────────────────────────
    result_path = None
    for candidate in [
        pirb_root / "results" / exp,
        pirb_root / "results" / f"{exp}.json",
    ]:
        if candidate.exists():
            result_path = candidate
            break

    if result_path is None:
        print(f"  [SKIP] No results file for experiment '{exp}'")
        return False

    # ── metadata → chunk_count ────────────────────────────────────────────────
    meta_path = pirb_root / "pirb_data" / exp / "metadata.json"
    if not meta_path.exists():
        print(f"  [SKIP] No metadata.json for '{exp}'")
        return False

    meta = load_json(meta_path)
    N = meta.get("chunk_count")
    if not N:
        print(f"  [SKIP] chunk_count missing in metadata for '{exp}'")
        return False

    # ── queries → relevant counts ─────────────────────────────────────────────
    relevant_counts = get_relevant_counts(pirb_root, exp)
    if not relevant_counts:
        print(f"  [SKIP] Could not load queries for '{exp}'")
        return False

    # ── compute ───────────────────────────────────────────────────────────────
    random_metrics = compute_random_metrics(N, relevant_counts)

    # ── write back ────────────────────────────────────────────────────────────
    result_data = load_json(result_path)

    if "metrics_random" in result_data:
        print(f"  [UPDATE] {exp}  (overwriting existing metrics_random)")
    else:
        print(f"  [ADD]    {exp}")

    result_data["metrics_random"] = random_metrics

    if not dry_run:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"           → saved to {result_path}")
    else:
        print(f"           → DRY RUN, not saved")
        print(f"           Acc@1={random_metrics['Accuracy@1']:.4f}  "
              f"MRR@10={random_metrics['MRR@10']:.4f}  "
              f"NDCG@10={random_metrics['NDCG@10']:.4f}  "
              f"Recall@10={random_metrics['Recall@10']:.4f}")

    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute random baseline metrics for all experiments")
    parser.add_argument("--data-root", default="../data", help="Root data directory (default: ../data)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    args = parser.parse_args()

    pirb_root = Path(args.data_root) / "pirb"

    if not pirb_root.exists():
        print(f"ERROR: pirb root not found at {pirb_root.resolve()}")
        return

    pirb_data = pirb_root / "pirb_data"
    if not pirb_data.exists():
        print(f"ERROR: pirb_data not found at {pirb_data.resolve()}")
        return

    experiments = sorted([d.name for d in pirb_data.iterdir() if d.is_dir()])
    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiment(s) in {pirb_data}")
    print(f"Data root: {Path(args.data_root).resolve()}")
    if args.dry_run:
        print("DRY RUN — files will not be modified\n")
    print()

    ok = sum(process_experiment(pirb_root, exp, dry_run=args.dry_run) for exp in experiments)
    print(f"\nDone: {ok}/{len(experiments)} experiments processed.")


if __name__ == "__main__":
    main()
