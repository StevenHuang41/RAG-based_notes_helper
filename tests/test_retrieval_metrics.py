from rag_notes_helper.eval.retrieval_metrics import calculate_retrieval_metrics


def test_recall_matches_source_lines_and_precision_counts_wrong_source_lines():
    metrics = calculate_retrieval_metrics(
        expected_evidence=[{
            "source": "A.md",
            "start_line": 100,
            "end_line": 150,
            "text": "ground truth",
        }],
        retrieval_lists=[[
            {
                "source": "A.md",
                "chunk_id": 1,
                "start_line": 90,
                "end_line": 120,
                "score": 0.9,
            },
            {
                "source": "B.md",
                "chunk_id": 2,
                "start_line": 100,
                "end_line": 140,
                "score": 0.8,
            },
        ]],
        top_k=5,
    )

    assert metrics == {
        "recall": 21 / 51,
        "precision": 21 / 72,
        "ndcg": 1.0,
    }


def test_overlapping_chunks_count_each_source_line_once():
    metrics = calculate_retrieval_metrics(
        expected_evidence=[{
            "source": "A.md",
            "start_line": 100,
            "end_line": 110,
            "text": "ground truth",
        }],
        retrieval_lists=[[
            {"source": "A.md", "chunk_id": 1, "start_line": 95, "end_line": 105},
            {"source": "A.md", "chunk_id": 2, "start_line": 103, "end_line": 110},
        ]],
        top_k=5,
    )

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 11 / 16
    assert 0.0 <= metrics["ndcg"] < 1.0


def test_single_line_evidence_is_retrievable_and_unretrieved_precision_is_na():
    metrics = calculate_retrieval_metrics(
        expected_evidence=[{
            "source": "A.md",
            "start_line": 7,
            "end_line": 7,
            "text": "one line",
        }],
        retrieval_lists=[[]],
        top_k=5,
    )

    assert metrics == {"recall": 0.0, "precision": None, "ndcg": None}


def test_inclusive_single_line_span_counts_as_one_retrieved_line():
    metrics = calculate_retrieval_metrics(
        expected_evidence=[{
            "source": "A.md",
            "start_line": 7,
            "end_line": 7,
            "text": "one line",
        }],
        retrieval_lists=[[
            {"source": "A.md", "chunk_id": 1, "start_line": 7, "end_line": 7},
        ]],
        top_k=5,
    )

    assert metrics == {"recall": 1.0, "precision": 1.0, "ndcg": 1.0}


def test_no_ground_truth_returns_not_applicable_metrics():
    metrics = calculate_retrieval_metrics(
        expected_evidence=[],
        retrieval_lists=[[
            {"source": "A.md", "chunk_id": 1, "start_line": 1, "end_line": 3},
        ]],
        top_k=5,
    )

    assert metrics == {"recall": None, "precision": None, "ndcg": None}


def test_ndcg_rewards_relevant_chunks_earlier_in_the_ranked_list():
    expected = [{
        "source": "A.md",
        "start_line": 1,
        "end_line": 20,
        "text": "ground truth",
    }]
    large_hit = {
        "source": "A.md",
        "chunk_id": 1,
        "start_line": 1,
        "end_line": 10,
    }
    small_hit = {
        "source": "A.md",
        "chunk_id": 2,
        "start_line": 20,
        "end_line": 20,
    }

    high_ranked_gain = calculate_retrieval_metrics(
        expected_evidence=expected,
        retrieval_lists=[[large_hit, small_hit]],
        top_k=2,
    )["ndcg"]
    low_ranked_gain = calculate_retrieval_metrics(
        expected_evidence=expected,
        retrieval_lists=[[small_hit, large_hit]],
        top_k=2,
    )["ndcg"]

    assert high_ranked_gain == 1.0
    assert low_ranked_gain < high_ranked_gain


def test_agentic_ndcg_averages_search_lists_but_recall_uses_their_union():
    metrics = calculate_retrieval_metrics(
        expected_evidence=[{
            "source": "A.md",
            "start_line": 1,
            "end_line": 4,
            "text": "ground truth",
        }],
        retrieval_lists=[
            [{"source": "A.md", "chunk_id": 1, "start_line": 1, "end_line": 2}],
            [{"source": "A.md", "chunk_id": 2, "start_line": 3, "end_line": 4}],
        ],
        top_k=2,
    )

    assert metrics == {"recall": 1.0, "precision": 1.0, "ndcg": 1.0}
