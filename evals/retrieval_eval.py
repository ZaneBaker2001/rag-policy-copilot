import json
from collections import Counter, defaultdict
from pathlib import Path

from app.models import EvalCase, UserContext
from app.retriever import VectorStore


def load_cases(path: str) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in raw]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _reciprocal_rank(results, expected_sources: set[str]) -> float:
    for idx, result in enumerate(results, start=1):
        if result.source in expected_sources:
            return 1.0 / idx
    return 0.0


def run_eval(cases_path: str) -> None:
    store = VectorStore()
    store.load()

    user = UserContext(user_id="admin", is_admin=True, roles=["admin"])
    cases = load_cases(cases_path)

    hits_at_1 = 0
    hits_at_3 = 0
    mrr_total = 0.0
    confident_total = 0
    abstained_total = 0

    per_source = defaultdict(
        lambda: {
            "count": 0,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "mrr_total": 0.0,
            "confident": 0,
            "abstained": 0,
        }
    )
    top1_prediction_counts = Counter()
    misses = []

    for case in cases:
        results, diagnostics = store.search(
            question=case.question,
            top_k=5,
            filters={},
            user=user,
        )

        expected_sources = set(case.expected_sources)
        top_sources = [r.source for r in results]
        top1_sources = set(top_sources[:1])
        top3_sources = set(top_sources[:3])

        hit1 = bool(expected_sources.intersection(top1_sources))
        hit3 = bool(expected_sources.intersection(top3_sources))
        rr = _reciprocal_rank(results, expected_sources)

        hits_at_1 += int(hit1)
        hits_at_3 += int(hit3)
        mrr_total += rr
        confident_total += int(diagnostics.confident)
        abstained_total += int(diagnostics.abstained)

        predicted_top1 = top_sources[0] if top_sources else "NO_RESULT"
        top1_prediction_counts[predicted_top1] += 1

        for source in expected_sources:
            bucket = per_source[source]
            bucket["count"] += 1
            bucket["hit_at_1"] += int(hit1)
            bucket["hit_at_3"] += int(hit3)
            bucket["mrr_total"] += rr
            bucket["confident"] += int(diagnostics.confident)
            bucket["abstained"] += int(diagnostics.abstained)

        row = {
            "question": case.question,
            "expected_sources": sorted(expected_sources),
            "confident": diagnostics.confident,
            "abstained": diagnostics.abstained,
            "top_score": round(diagnostics.top_score, 4),
            "second_score": round(diagnostics.second_score, 4),
            "margin": round(diagnostics.margin, 4),
            "top_sources": top_sources[:5],
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "reciprocal_rank": round(rr, 4),
        }
        print(row)

        if not hit3:
            misses.append(row)

    total = max(len(cases), 1)

    print("\n=== Overall Metrics ===")
    print(f"Cases: {total}")
    print(f"Hit@1: {_safe_div(hits_at_1, total):.2%}")
    print(f"Hit@3: {_safe_div(hits_at_3, total):.2%}")
    print(f"MRR@5: {_safe_div(mrr_total, total):.4f}")
    print(f"Confident Rate: {_safe_div(confident_total, total):.2%}")
    print(f"Abstain Rate: {_safe_div(abstained_total, total):.2%}")

    print("\n=== Per-Source Metrics ===")
    for source, stats in sorted(per_source.items()):
        count = stats["count"]
        print(
            {
                "source": source,
                "cases": count,
                "hit_at_1": f"{_safe_div(stats['hit_at_1'], count):.2%}",
                "hit_at_3": f"{_safe_div(stats['hit_at_3'], count):.2%}",
                "mrr@5": round(_safe_div(stats["mrr_total"], count), 4),
                "confident_rate": f"{_safe_div(stats['confident'], count):.2%}",
                "abstain_rate": f"{_safe_div(stats['abstained'], count):.2%}",
            }
        )

    print("\n=== Top-1 Prediction Distribution ===")
    for source, count in top1_prediction_counts.most_common():
        print(f"{source}: {count}")

    print("\n=== Misses (Hit@3 = False) ===")
    if not misses:
        print("None")
    else:
        for miss in misses:
            print(miss)


if __name__ == "__main__":
    run_eval("evals/eval_cases.json")