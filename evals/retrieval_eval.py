import json
from pathlib import Path

from app.models import EvalCase, UserContext
from app.retriever import VectorStore


def load_cases(path: str) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in raw]


def run_eval(cases_path: str) -> None:
    store = VectorStore()
    store.load()

    user = UserContext(user_id="admin", is_admin=True, roles=["admin"])
    cases = load_cases(cases_path)

    hits_at_1 = 0
    hits_at_3 = 0

    for case in cases:
        results, diagnostics = store.search(
            question=case.question,
            top_k=3,
            filters={},
            user=user,
        )

        top1_sources = {r.source for r in results[:1]}
        top3_sources = {r.source for r in results[:3]}

        expected_sources = set(case.expected_sources)

        if expected_sources.intersection(top1_sources):
            hits_at_1 += 1
        if expected_sources.intersection(top3_sources):
            hits_at_3 += 1

        print(
            {
                "question": case.question,
                "confident": diagnostics.confident,
                "top_sources": [r.source for r in results[:3]],
            }
        )

    total = max(len(cases), 1)
    print(f"Hit@1: {hits_at_1 / total:.2%}")
    print(f"Hit@3: {hits_at_3 / total:.2%}")


if __name__ == "__main__":
    run_eval("evals/eval_cases.json")