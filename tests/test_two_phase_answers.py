import json
from types import SimpleNamespace

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag import agentic


def setup_query(monkeypatch, reviews, answers, *, max_attempts=3):
    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=max_attempts,
        llm=SimpleNamespace(max_chunks=5, max_tokens=1024),
    )
    monkeypatch.setattr(agentic, 'get_settings', lambda: settings)
    hits = [{'source': 'release.md', 'chunk_id': 0, 'text': 'Lena owns launch.', 'score': .9}]
    searches = []
    monkeypatch.setattr(agentic, 'search_notes', lambda query, **kwargs: searches.append(query) or hits)
    payloads = []
    review_iter = iter(reviews)

    class Controller:
        provider_name = 'fixture'
        model = 'fixture'

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]['content'])
            if 'chunks_pool' in payload:
                return json.dumps({'coverage': 'sufficient_evidence'})
            payloads.append(payload)
            response = next(review_iter)
            if isinstance(response, Exception):
                raise response
            return json.dumps(response) if isinstance(response, dict) else response

    monkeypatch.setattr(agentic, 'get_llm', Controller)
    answer_iter = iter(answers)
    answer_calls = []

    def answer(query, *, hits, **kwargs):
        answer_calls.append((hits, kwargs))
        response = next(answer_iter)
        if isinstance(response, Exception):
            raise response
        return {'answer': response, 'citations': [], 'generation_context': [
            {**hit, 'prompt_block': hit['text']} for hit in hits
        ]}

    monkeypatch.setattr(agentic, 'rag_answer', answer)
    return searches, payloads, answer_calls


def test_supported_partial_answer_can_pass_without_using_remaining_attempts(monkeypatch):
    searches, payloads, answer_calls = setup_query(monkeypatch, [{
        'action': 'accept', 'answer_coverage': 'partial',
        'missing_information': ['Checkpoint date is not recorded.'],
    }], ['Lena owns launch. The checkpoint date is not recorded.'])
    result = cli.run_query(object(), object(), query='Who owns launch and what is the checkpoint date?', agentic=True)
    assert result['review_passed'] is True
    assert result['answer_coverage'] == 'partial'
    assert result['answer_missing_information'] == ['Checkpoint date is not recorded.']
    assert result['answer_attempts'] == 1
    assert len(searches) == len(answer_calls) == len(payloads) == 1
    assert 'candidate_pool' not in payloads[0]


def test_revisions_keep_context_and_do_not_reopen_search(monkeypatch):
    searches, payloads, answer_calls = setup_query(monkeypatch, [
        {'action': 'revise', 'feedback': 'Remove invented date and disclose the gap.'},
        {'action': 'accept', 'answer_coverage': 'partial', 'missing_information': ['Date']},
    ], ['Lena owns launch on June 1.', 'Lena owns launch. The date is not recorded.'])
    result = cli.run_query(object(), object(), query='Who and when?', agentic=True)
    assert result['review_passed'] is True
    assert result['answer_attempts'] == 2
    assert len(searches) == 1
    assert answer_calls[0][0] == answer_calls[1][0]
    assert answer_calls[1][1]['revision_feedback'] == 'Remove invented date and disclose the gap.'
    assert payloads[0]['generation_context'] == payloads[1]['generation_context']


@pytest.mark.parametrize('reviews,answers,reason,cause', [
    ([], ['   '], 'generation_failure', 'empty_model_output'),
    ([], [RuntimeError('provider secret')], 'generation_failure', 'provider_failure'),
    (['not-json', 'not-json'], ['Invented claim.'], 'answer_review_failure', 'invalid_json'),
    ([RuntimeError('provider secret')], ['Invented claim.'], 'answer_review_failure', 'provider_failure'),
])
def test_generation_and_review_failures_return_safe_warning(monkeypatch, reviews, answers, reason, cause):
    searches, payloads, answer_calls = setup_query(monkeypatch, reviews, answers)
    result = cli.run_query(object(), object(), query='Who owns launch?', agentic=True)
    assert result['reason_code'] == reason
    assert result['workflow_error'] == reason
    assert result['failure_category'] == cause
    assert result['review_passed'] is False
    assert result['response_origin'] == 'deterministic'
    assert result['answer'].strip()
    assert 'Invented' not in result['answer'] and 'secret' not in result['answer']
    assert result['citations'] == []
    assert len(searches) == len(answer_calls) == 1


def test_invalid_review_retries_without_consuming_generation_attempt(monkeypatch):
    searches, payloads, answer_calls = setup_query(monkeypatch, [
        'bad-json', {'action': 'accept', 'answer_coverage': 'full'},
    ], ['Lena owns launch.'])
    result = cli.run_query(object(), object(), query='Who owns launch?', agentic=True)
    assert result['review_passed'] is True
    assert result['answer_attempts'] == 1
    assert len(payloads) == 2 and 'validation_feedback' in payloads[1]
    assert len(searches) == len(answer_calls) == 1


def test_revision_exhaustion_never_returns_rejected_candidate(monkeypatch):
    searches, payloads, answer_calls = setup_query(monkeypatch, [
        {'action': 'revise', 'feedback': 'Remove invented date.', 'answer_coverage': 'partial'},
        {'action': 'revise', 'feedback': 'Still unsupported.', 'answer_coverage': 'partial'},
    ], ['REJECTED first', 'REJECTED second'], max_attempts=2)
    result = cli.run_query(object(), object(), query='Who and when?', agentic=True)
    assert result['reason_code'] == 'answer_attempts_exhausted'
    assert result['review_passed'] is False
    assert result['answer_retry_exhausted'] is True
    assert result['answer_attempts'] == 2
    assert 'REJECTED' not in result['answer']
    assert len(searches) == 1 and len(answer_calls) == 2


def test_answer_model_abstention_is_accepted_with_none_coverage(monkeypatch):
    setup_query(monkeypatch, [{'action': 'accept_abstention'}], ['I cannot find the answers in the notes.'])
    result = cli.run_query(object(), object(), query='What is missing?', agentic=True)
    assert result['reason_code'] == 'insufficient_evidence'
    assert result['answer_coverage'] == 'none'
    assert result['review_passed'] is True


def test_real_generation_preserves_full_selected_context_for_revision(monkeypatch):
    from rag_notes_helper.rag import answer as answer_module

    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=3, line_width=80,
        llm=SimpleNamespace(max_chunks=3, max_tokens=512),
    )
    review_settings = SimpleNamespace(agentic_max_search_attempts=3, agentic_max_answer_attempts=3, llm=SimpleNamespace(max_chunks=3, max_tokens=512))
    monkeypatch.setattr(agentic, 'get_settings', lambda: review_settings)
    monkeypatch.setattr(answer_module, 'get_settings', lambda: settings)
    chunks = [{'source': 'release.md', 'chunk_id': i, 'text': 'Lena owns launch. ' + 'x' * 4000, 'score': .9} for i in range(3)]
    monkeypatch.setattr(agentic, 'search_notes', lambda query, **kwargs: chunks)
    reviews = []

    class Controller:
        provider_name = 'fixture'
        model = 'fixture'

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]['content'])
            if 'chunks_pool' in payload:
                return json.dumps({'coverage': 'sufficient_evidence'})
            reviews.append(payload['generation_context'])
            return json.dumps({'action': 'revise', 'feedback': 'Disclose missing date.'}) if len(reviews) == 1 else json.dumps({'action': 'accept', 'answer_coverage': 'partial'})

    prompts = []

    class AnswerModel:
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return 'Lena owns launch. The date is not recorded.'

    monkeypatch.setattr(agentic, 'get_llm', Controller)
    monkeypatch.setattr(answer_module, 'get_llm', AnswerModel)
    result = cli.run_query(object(), object(), query='Who and when?', agentic=True)
    assert result['review_passed'] is True, (result.get('reason_code'), result.get('failure_category'), result.get('workflow_error_type'))
    assert len(result['selected_pool']) == 3
    assert len(result['generation_context']) == 3
    assert reviews[0] == reviews[1]
    assert [item['text'] for item in reviews[0]] == [item['text'] for item in result['generation_context']]
    for prompt in prompts:
        assert all(chunk['text'] in prompt[-1]['content'] for chunk in chunks)


def test_provider_failure_preserves_full_attempted_context_without_raw_error(monkeypatch):
    from rag_notes_helper.rag import answer as answer_module

    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=3, line_width=80,
        llm=SimpleNamespace(max_chunks=3, max_tokens=512),
    )
    review_settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=3,
        llm=SimpleNamespace(max_chunks=3, max_tokens=512),
    )
    monkeypatch.setattr(agentic, 'get_settings', lambda: review_settings)
    monkeypatch.setattr(answer_module, 'get_settings', lambda: settings)
    chunks = [{'source': 'release.md', 'chunk_id': i, 'text': f'Fact {i}. ' + 'x' * 4000, 'score': .9} for i in range(3)]
    monkeypatch.setattr(agentic, 'search_notes', lambda query, **kwargs: chunks)

    class Controller:
        provider_name = 'fixture'
        model = 'fixture'

        def generate_raw(self, messages):
            return json.dumps({'coverage': 'sufficient_evidence'})

    prompts = []

    class AnswerModel:
        provider_name = 'fixture'
        model = 'answer'

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            raise RuntimeError('SECRET raw provider output')

    monkeypatch.setattr(agentic, 'get_llm', Controller)
    monkeypatch.setattr(answer_module, 'get_llm', AnswerModel)
    usage = []
    result = cli.run_query(object(), object(), query='Who owns launch?', agentic=True, on_generation=usage.append)
    assert result['reason_code'] == 'generation_failure'
    assert result['failure_category'] == 'provider_failure'
    assert result['workflow_error_type'] == 'RuntimeError'
    assert len(result['selected_pool']) == 3
    assert len(result['generation_context']) == 3
    assert all(item['text'] in prompts[0][-1]['content'] for item in result['generation_context'])
    assert chunks[-1]['text'] in prompts[0][-1]['content']
    assert 'SECRET' not in repr(result)
    assert result['citations'] == []
    assert result['review_passed'] is False
    assert any(call.get('phase') == 'answer_generation' for call in usage)
