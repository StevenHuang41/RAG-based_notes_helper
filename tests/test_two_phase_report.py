import pytest

from rag_notes_helper.eval.report import write_readme


def render_report(tmp_path, case, *, summary=None, legacy_metadata=None):
    path = tmp_path / 'README.md'
    write_readme(path, {
        'run_metadata': {'workflow': 'agentic', 'max_answer_chunks': 1, **(legacy_metadata or {})},
        'workflow_degraded_count': 1 if case.get('workflow_degraded') else 0,
        'summary': {'workflows': {'agentic': summary or {}}},
        'cases': [{'workflow': 'agentic', 'case_id': 'pool', 'question': 'Who?', **case}],
    })
    return path.read_text()


@pytest.mark.parametrize('reranked,empty', [(True, False), (False, False), (False, True)])
def test_report_compares_pool_and_selection_without_repeating_raw_text(tmp_path, reranked, empty):
    first = {
        'source': 'launch.md', 'chunk_id': 4, 'start_line': 2, 'end_line': 3,
        'text': 'EXACT_CONTEXT_ONLY_TEXT', 'score': .8,
        'search_provenance': [{'search_id': 1}, {'search_id': 2}],
    }
    second = {'source': 'noise.md', 'chunk_id': 7, 'start_line': 1, 'end_line': 1, 'text': 'NOISE_RAW_TEXT', 'score': .5, 'search_provenance': [{'search_id': 2}]}
    before = [] if empty else [first, second] if reranked else [first]
    selected = [] if empty else [{**first, **({'reranker_score': .99} if reranked else {})}]
    report = render_report(tmp_path, {
        'chunks_pool': before, 'selected_pool': selected,
        'final_selection': {'rerank_performed': reranked, 'max_chunks': 1, 'latency_ms': 3.5 if reranked else None},
        'expected_evidence': [{'source': 'launch.md', 'start_line': 3, 'end_line': 3}],
        'generation_context': selected,
        'retrieval_coverage': {'assessed_question': 'Who?', 'coverage': 'sufficient_evidence', 'missing_information': []},
        'retrieval_stop_reason': 'sufficient_evidence', 'answer_coverage': 'full', 'review_passed': True,
    })
    assert '#### Accumulated chunks pool' in report
    assert '#### Selected chunks pool' in report
    tables = report.split('#### Accumulated chunks pool', 1)[1].split('#### Final generation context', 1)[0]
    assert tables.count('| Order | Chunk ID | Source path | Lines | Search |') == 2
    assert 'EXACT_CONTEXT_ONLY_TEXT' not in tables and 'NOISE_RAW_TEXT' not in tables
    if empty:
        assert tables.count('No chunks were selected or accumulated.') == 2
    else:
        assert 'launch.md | 2–3 | 1, 2 |' in tables
        assert report.count('EXACT_CONTEXT_ONLY_TEXT') == 1
        assert '1/1' in tables
    if reranked:
        assert '| Final pool rerank | 3.5000 |' in report
        assert '0.9900' in tables
    else:
        assert 'Skipped: pool size <= LLM_MAX_CHUNKS' in tables


def test_report_attributes_only_actual_review_calls_and_final_pool_rerank(tmp_path):
    calls = [
        {'call_type': 'controller', 'phase': 'intelligibility', 'search_index': 1, 'latency_ms': 1.5},
        {'call_type': 'controller', 'phase': 'retrieval_review', 'search_index': 1, 'latency_ms': 2.5},
        {'call_type': 'controller', 'phase': 'retrieval_review', 'search_index': 1, 'latency_ms': 3.5},
        {'call_type': 'controller', 'phase': 'answer_review', 'answer_attempt': 1, 'latency_ms': 4.5},
    ]
    report = render_report(tmp_path, {
        'generation_usage': {'calls': calls},
        'retrieval_stages': [{'search_id': 1, 'stages': {'vector': {'latency_ms': 10.0, 'results': []}}}],
        'final_selection': {'rerank_performed': True, 'latency_ms': 5.0},
        'latency': {'retrieval_ms': 10.0, 'controller_ms': 12.0, 'answer_generation_ms': 20.0},
    })
    search = report.split('**Search**', 1)[1].split('**Answer attempts**', 1)[0]
    assert '| Search 1 | Intelligibility check | 1.5000 |' in search
    assert '| Search 1 | Retrieval review | 2.5000 |' in search
    assert '| Search 1 | Retrieval review | 3.5000 |' in search
    assert '| Search 1 | Controller |' not in search
    assert '| Search 2 |' not in search
    assert '| Final pool rerank | 5.0000 | 5.0000 |' in report
    assert '| RAG query | 47.0000 | 47.0000 |' in report
    assert 'Controller call 1 (Intelligibility, Search 1)' in report
    assert '| Answer 1 | Controller review | 4.5000 |' in report


def test_report_exposes_degradation_coverage_and_metric_denominators(tmp_path):
    counts = {name: {'eligible': 2, 'scored': 1, 'unavailable': 1, 'failed': 1} for name in ['faithfulness', 'answer_relevancy', 'answer_correctness', 'context_utilization']}
    report = render_report(tmp_path, {
        'workflow_degraded': True,
        'retrieval_coverage': {'assessed_question': 'Who?', 'coverage': 'unknown', 'missing_information': ['Date']},
        'retrieval_stop_reason': 'review_degraded',
        'retrieval_degradation': {'cause': 'review_context_budget_exceeded', 'search_id': 1},
        'answer_coverage': 'partial', 'review_passed': True,
        'answer_missing_information': ['Date'],
        'judge_metrics': {'faithfulness': {'status': 'failed', 'eligible': True, 'reason': 'Judge failed.'}},
    }, summary={'ragas_metric_status_counts': counts, 'workflow_degraded_case_count': 1, 'workflow_failure_case_count': 0, 'outcome_counts': {'workflow_degradations': {'review_context_budget_exceeded': 1}}}, legacy_metadata={'context_window_tokens': 8192})
    assert '**Workflow degraded:** `1`' in report
    assert '| Pool coverage | unknown |' in report
    assert '| Assessed question | Who? |' in report
    assert '| Retrieval stop reason | review_degraded |' in report
    assert '| Answer coverage | partial |' in report
    assert '| Review passed | true |' in report
    assert '| Workflow degradation | review_context_budget_exceeded | — | 1 |' in report
    assert '| Metric | Eligible | Scored | Unavailable | Failed |' in report
    assert '| Faithfulness | 2 | 1 | 1 | 1 |' in report
    assert '| Faithfulness | — | failed | Judge failed. |' in report
