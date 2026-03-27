import json
import statistics
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.config import settings
from app.generator import generate_answer
from app.models import UserContext
from app.retriever import VectorStore
from evals.retrieval_eval import load_cases


OUTPUT_DIR = Path('evals/output')


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[94]


def _safe_max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _summarize(values: list[float]) -> dict:
    return {
        'mean_ms': round(_safe_mean(values), 2),
        'median_ms': round(statistics.median(values), 2) if values else 0.0,
        'p95_ms': round(_safe_p95(values), 2),
        'max_ms': round(_safe_max(values), 2),
    }


def _plot_latency_charts(rows: list[dict], output_dir: Path) -> None:
    case_labels = [f"{idx + 1}" for idx, _ in enumerate(rows)]
    retrieval = [row['retrieval_latency_ms'] for row in rows]
    generation = [row['generation_latency_ms'] for row in rows]
    total = [row['total_latency_ms'] for row in rows]

    plt.figure(figsize=(12, 6))
    plt.plot(case_labels, retrieval, marker='o', label='Retrieval')
    plt.plot(case_labels, generation, marker='o', label='Generation')
    plt.plot(case_labels, total, marker='o', label='Total')
    plt.xlabel('Evaluation Case')
    plt.ylabel('Latency (ms)')
    plt.title('Latency by Evaluation Case')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_by_case.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(total, bins=min(10, max(5, len(total))))
    plt.xlabel('Total Latency (ms)')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of End-to-End Latency')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(retrieval, generation)
    plt.xlabel('Retrieval Latency (ms)')
    plt.ylabel('Generation Latency (ms)')
    plt.title('Retrieval vs Generation Latency')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_scatter.png', dpi=200)
    plt.close()


def run_eval(cases_path: str = 'evals/eval_cases.json') -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    store = VectorStore()
    store.load()
    user = UserContext(user_id='admin', is_admin=True, roles=['admin'])
    cases = load_cases(cases_path)

    rows = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    for case in cases:
        started = perf_counter()
        chunks, diagnostics = store.search(
            question=case.question,
            top_k=settings.top_k,
            filters={},
            user=user,
        )

        generation_started = perf_counter()
        answer = generate_answer(case.question, chunks, diagnostics)
        generation_latency_ms = (perf_counter() - generation_started) * 1000
        total_latency_ms = (perf_counter() - started) * 1000

        row = {
            'question': case.question,
            'retrieval_latency_ms': round(diagnostics.retrieval_latency_ms, 2),
            'generation_latency_ms': round(generation_latency_ms, 2),
            'total_latency_ms': round(total_latency_ms, 2),
            'confident': diagnostics.confident,
            'abstained': diagnostics.abstained,
            'answer_preview': answer[:180].replace('\n', ' '),
        }
        rows.append(row)
        retrieval_latencies.append(diagnostics.retrieval_latency_ms)
        generation_latencies.append(generation_latency_ms)
        total_latencies.append(total_latency_ms)
        print(row)

    summary = {
        'cases': len(rows),
        'openai_model': settings.openai_model,
        'using_openai_generation': bool(settings.openai_api_key),
        'retrieval_latency': _summarize(retrieval_latencies),
        'generation_latency': _summarize(generation_latencies),
        'total_latency': _summarize(total_latencies),
        'artifacts': {
            'rows_json': str(OUTPUT_DIR / 'latency_eval_rows.json'),
            'summary_json': str(OUTPUT_DIR / 'latency_eval_summary.json'),
            'latency_by_case_png': str(OUTPUT_DIR / 'latency_by_case.png'),
            'latency_distribution_png': str(OUTPUT_DIR / 'latency_distribution.png'),
            'latency_scatter_png': str(OUTPUT_DIR / 'latency_scatter.png'),
        },
    }

    (OUTPUT_DIR / 'latency_eval_rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'latency_eval_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _plot_latency_charts(rows, OUTPUT_DIR)

    print('\n=== Latency Summary ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run_eval()