import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.config import settings
from app.generator import generate_answer
from app.models import UserContext
from app.retriever import VectorStore
from evals.retrieval_eval import load_cases


OUTPUT_DIR = Path('evals/output')
TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
CITATION_RE = re.compile(r"\[[^\]]+\]")
ABSTAIN_HINTS = (
    'insufficient',
    'not enough context',
    'don\'t have enough',
    'cannot answer',
    'can\'t answer',
    'couldn\'t find',
)


def _normalize(text: str) -> str:
    return ' '.join(TOKEN_RE.findall(text.lower()))


def _token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [part.strip() for part in parts if part.strip()]


def _contains_expected_answer(answer: str, expected: list[str]) -> bool:
    if not expected:
        return True
    normalized_answer = _normalize(answer)
    return all(_normalize(item) in normalized_answer for item in expected)


def _is_abstention(answer: str, expected_should_abstain: bool) -> bool:
    if expected_should_abstain:
        lowered = answer.lower()
        return any(hint in lowered for hint in ABSTAIN_HINTS)
    return False


def _sentence_grounding_score(sentence: str, contexts: list[str]) -> float:
    sentence_tokens = _token_set(CITATION_RE.sub('', sentence))
    if not sentence_tokens:
        return 1.0

    best = 0.0
    for context in contexts:
        context_tokens = _token_set(context)
        if not context_tokens:
            continue
        overlap = len(sentence_tokens & context_tokens)
        score = overlap / len(sentence_tokens)
        if score > best:
            best = score
    return best


def _hallucination_metrics(answer: str, contexts: list[str]) -> tuple[float, float, int]:
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0, 0.0, 0

    scores = [_sentence_grounding_score(sentence, contexts) for sentence in sentences]
    unsupported = sum(1 for score in scores if score < 0.5)
    return mean(scores), unsupported / len(scores), len(sentences)


def _plot(rows: list[dict], source_rows: dict[str, dict], output_dir: Path) -> None:
    labels = [str(i + 1) for i in range(len(rows))]
    unsupported = [row['unsupported_sentence_ratio'] for row in rows]
    grounding = [row['grounding_score'] for row in rows]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, unsupported)
    plt.xlabel('Evaluation Case')
    plt.ylabel('Unsupported Sentence Ratio')
    plt.title('Hallucination Proxy by Evaluation Case')
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_by_case.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(grounding, bins=min(10, max(5, len(grounding))))
    plt.xlabel('Grounding Score')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Answer Grounding Scores')
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_grounding_distribution.png', dpi=200)
    plt.close()

    sources = list(source_rows.keys())
    rates = [source_rows[source]['hallucination_rate'] for source in sources]
    plt.figure(figsize=(10, 6))
    plt.bar(sources, rates)
    plt.xlabel('Expected Source')
    plt.ylabel('Hallucination Rate')
    plt.title('Hallucination Rate by Source')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_by_source.png', dpi=200)
    plt.close()


def run_eval(cases_path: str = 'evals/eval_cases.json') -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    store = VectorStore()
    store.load()
    user = UserContext(user_id='admin', is_admin=True, roles=['admin'])
    cases = load_cases(cases_path)

    rows = []
    per_source = defaultdict(lambda: {'count': 0, 'hallucinations': 0, 'grounding_scores': []})

    hallucinations = 0
    answer_support_hits = 0
    abstention_alignment_hits = 0

    for case in cases:
        chunks, diagnostics = store.search(
            question=case.question,
            top_k=settings.top_k,
            filters={},
            user=user,
        )
        answer = generate_answer(case.question, chunks, diagnostics)
        contexts = [chunk.text for chunk in chunks]

        answer_support = _contains_expected_answer(answer, case.expected_answer_contains)
        abstention_alignment = (diagnostics.abstained == case.should_abstain) or _is_abstention(answer, case.should_abstain)
        grounding_score, unsupported_ratio, sentence_count = _hallucination_metrics(answer, contexts)

        flagged = False
        if case.should_abstain:
            flagged = not abstention_alignment
        else:
            flagged = (not answer_support) or (grounding_score < 0.6) or (unsupported_ratio > 0.5)

        hallucinations += int(flagged)
        answer_support_hits += int(answer_support)
        abstention_alignment_hits += int(abstention_alignment)

        source_key = case.expected_sources[0] if case.expected_sources else 'UNKNOWN'
        per_source[source_key]['count'] += 1
        per_source[source_key]['hallucinations'] += int(flagged)
        per_source[source_key]['grounding_scores'].append(grounding_score)

        row = {
            'question': case.question,
            'expected_sources': case.expected_sources,
            'should_abstain': case.should_abstain,
            'diagnostics_abstained': diagnostics.abstained,
            'answer_support': answer_support,
            'abstention_alignment': abstention_alignment,
            'grounding_score': round(grounding_score, 4),
            'unsupported_sentence_ratio': round(unsupported_ratio, 4),
            'sentence_count': sentence_count,
            'hallucination_flag': flagged,
            'answer_preview': answer[:220].replace('\n', ' '),
        }
        rows.append(row)
        print(row)

    source_rows = {}
    for source, stats in sorted(per_source.items()):
        source_rows[source] = {
            'cases': stats['count'],
            'hallucination_rate': round(stats['hallucinations'] / stats['count'], 4) if stats['count'] else 0.0,
            'avg_grounding_score': round(mean(stats['grounding_scores']), 4) if stats['grounding_scores'] else 0.0,
        }

    summary = {
        'cases': len(rows),
        'openai_model': settings.openai_model,
        'using_openai_generation': bool(settings.openai_api_key),
        'hallucination_rate': round(hallucinations / len(rows), 4) if rows else 0.0,
        'answer_support_rate': round(answer_support_hits / len(rows), 4) if rows else 0.0,
        'abstention_alignment_rate': round(abstention_alignment_hits / len(rows), 4) if rows else 0.0,
        'avg_grounding_score': round(mean([row['grounding_score'] for row in rows]), 4) if rows else 0.0,
        'avg_unsupported_sentence_ratio': round(mean([row['unsupported_sentence_ratio'] for row in rows]), 4) if rows else 0.0,
        'per_source': source_rows,
        'artifacts': {
            'rows_json': str(OUTPUT_DIR / 'hallucination_eval_rows.json'),
            'summary_json': str(OUTPUT_DIR / 'hallucination_eval_summary.json'),
            'hallucination_by_case_png': str(OUTPUT_DIR / 'hallucination_by_case.png'),
            'grounding_distribution_png': str(OUTPUT_DIR / 'hallucination_grounding_distribution.png'),
            'hallucination_by_source_png': str(OUTPUT_DIR / 'hallucination_by_source.png'),
        },
    }

    (OUTPUT_DIR / 'hallucination_eval_rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'hallucination_eval_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _plot(rows, source_rows, OUTPUT_DIR)

    print('\n=== Hallucination Summary ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run_eval()