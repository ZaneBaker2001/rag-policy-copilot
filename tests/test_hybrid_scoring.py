from app.utils import sparse_overlap_score


def test_sparse_overlap_rewards_keyword_matches():
    query = "pto carryover policy"
    doc_a = "The PTO carryover policy allows 40 hours to roll over into next year."
    doc_b = "The cafeteria menu changes every Monday."

    assert sparse_overlap_score(query, doc_a) > sparse_overlap_score(query, doc_b)